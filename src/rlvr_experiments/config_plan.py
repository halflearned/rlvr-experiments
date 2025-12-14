from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml


@dataclass(frozen=True)
class RolePlan:
    kind: str              # "titan" | "vllm"
    config: Dict[str, Any] # raw role config
    world_size: int        # external sync ranks contributed by this role


@dataclass(frozen=True)
class ChannelPlan:
    name: str
    src: str
    dst: str
    world_size: int
    offsets: Dict[str, int]  # role -> base rank offset in this channel
    src_rank: int = 0


@dataclass(frozen=True)
class Plan:
    run: Dict[str, Any]
    model: Dict[str, Any]
    roles: Dict[str, RolePlan]
    channels: Dict[Tuple[str, str], ChannelPlan]  # (src, dst) -> plan
    chunk_mb: int

    def channel(self, src: str, dst: str) -> ChannelPlan:
        return self.channels[(src, dst)]


def load_plan(path: str) -> Plan:
    y = yaml.safe_load(open(path, "r"))

    run = y["run"]
    model = dict(y.get("model", {}))
    roles_in = y["roles"]                 # required
    sync_in = dict(y.get("sync", {}))
    wiring = sync_in["wiring"]            # required
    chunk_mb = int(sync_in.get("chunk_mb", 100))

    def titan_world_size(cfg: Dict[str, Any]) -> int:
        p = cfg.get("parallelism", {})
        dp_rep = int(p.get("data_parallel_replicate_degree", 1))
        dp_shard = int(p.get("data_parallel_shard_degree", 1))
        tp = int(p.get("tensor_parallel_degree", 1))
        # - context_parallel_degree is always 1
        # - pipeline_parallel_degree is always 1 for now at least
        return dp_rep * dp_shard * tp

    def vllm_world_size(cfg: Dict[str, Any]) -> int:
        """
        vLLM can internally spawn multiple worker processes/ranks depending on its
        parallel configuration. For external sync, we should count the number of
        participating worker ranks.

        We keep this simple and extensible: multiply any known degrees if present.
        """
        def get_int(key: str) -> int:
            v = cfg.get(key, 1)
            try:
                return int(v)
            except Exception:
                return 1

        tp = get_int("tensor_parallel_size")
        pp = get_int("pipeline_parallel_size")
        ep = get_int("expert_parallel_size")  # only if you use it; otherwise 1
        return max(1, tp * pp * ep)

    roles: Dict[str, RolePlan] = {}
    for r in roles_in:
        name = r["name"]
        kind = r["kind"]
        cfg = dict(r.get("config", {}) or {})
        if kind == "titan":
            ws = titan_world_size(cfg)
        elif kind == "vllm":
            ws = vllm_world_size(cfg)
        else:
            raise ValueError(f"Unknown role kind: {kind}")
        roles[name] = RolePlan(kind=kind, config=cfg, world_size=ws)

    channels: Dict[Tuple[str, str], ChannelPlan] = {}
    for w in wiring:
        src, dst = w["src"], w["dst"]
        name = w.get("name") or f"{src}_to_{dst}"
        src_rank = int(w.get("src_rank", 0))

        src_ws = roles[src].world_size
        dst_ws = roles[dst].world_size
        world_size = src_ws + dst_ws
        offsets = {src: 0, dst: src_ws}

        key = (src, dst)
        if key in channels:
            raise ValueError(f"Duplicate wiring entry for {src}->{dst}")

        channels[key] = ChannelPlan(
            name=name,
            src=src,
            dst=dst,
            world_size=world_size,
            offsets=offsets,
            src_rank=src_rank,
        )

    return Plan(run=run, model=model, roles=roles, channels=channels, chunk_mb=chunk_mb)

