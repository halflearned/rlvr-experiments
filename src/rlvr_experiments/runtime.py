
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Dict

import ray

from .config_plan import Plan, load_plan
from .titan_actor import create_titan_group
from .vllm_engine_actor import VLLMEngineRank, VLLMHandle
from .rollout_buffer import RolloutBuffer


def _stable_hash_u32(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def _port_map(*, base: int, stride: int, namespace: int, names: list[str]) -> Dict[str, int]:
    start = base + namespace * stride + 10
    return {n: start + i for i, n in enumerate(sorted(names))}

def _max_namespace_for(*, base: int, stride: int, count: int) -> int:
    # Ensure all ports stay within [1, 65535], accounting for the highest index.
    safe_count = max(1, count)
    return (65535 - base - 10 - (safe_count - 1)) // stride


@dataclass
class Runtime:
    plan: Plan
    host: str
    roles: Dict[str, Any]               # role_name -> handle
    titan_ports: Dict[str, int]         # titan role rendezvous ports
    channel_ports: Dict[str, int]       # channel_name -> sync port
    namespace: int
    buffer: Any

    @classmethod
    async def from_plan(cls, plan_path: str) -> "Runtime":
        plan = load_plan(plan_path)
        ray.init(address="auto")
        host = ray.util.get_node_ip_address()

        run_name = plan.run.get("name", "unnamed_run")
        titan_roles = [n for n, r in plan.roles.items() if r.kind == "titan"]
        channel_names = [ch.name for ch in plan.channels.values()]
        max_ns_titan = _max_namespace_for(
            base=29500, stride=200, count=len(titan_roles)
        )
        max_ns_channel = _max_namespace_for(
            base=51200, stride=200, count=len(channel_names)
        )
        max_namespace = min(199, max_ns_titan, max_ns_channel)
        if max_namespace < 0:
            raise ValueError(
                "No valid port namespace available; reduce role/channel counts or adjust base/stride."
            )
        namespace = _stable_hash_u32(run_name) % (max_namespace + 1)

        titan_ports = _port_map(base=29500, stride=200, namespace=namespace, names=titan_roles)
        channel_ports = _port_map(base=51200, stride=200, namespace=namespace, names=channel_names)

        # One-time log of the final wiring decisions.
        print(f"[runtime] run.name={run_name!r} host={host} namespace={namespace}")
        for rn in sorted(titan_ports):
            print(f"[runtime] titan rendezvous port role={rn} port={titan_ports[rn]}")
        for cn in sorted(channel_ports):
            print(f"[runtime] sync channel port channel={cn} port={channel_ports[cn]}")

        buffer = RolloutBuffer(**plan.buffer)

        return cls(
            plan=plan,
            host=host,
            roles={},
            titan_ports=titan_ports,
            channel_ports=channel_ports,
            namespace=namespace,
            buffer=buffer,
        )

    async def start(self, wire=True) -> "Runtime":
        # Spawn Titan roles first (they claim specific GPUs), then vLLM roles.
        # This ensures Ray's scheduler sees accurate GPU availability when
        # placing vLLM actors across the cluster.
        titan_roles = [n for n, r in self.plan.roles.items() if r.kind == "titan"]
        vllm_roles = [n for n, r in self.plan.roles.items() if r.kind == "vllm"]

        # Spawn Titan roles in parallel
        if titan_roles:
            await asyncio.gather(*[self.spawn_role(name) for name in titan_roles])

        # Spawn vLLM roles after Titan has claimed its GPUs
        for name in vllm_roles:
            await self.spawn_role(name)

        if wire:
            # Wire up all channels
            for src, dst in self.plan.channels:
                await self.wire(src, dst)

        return self

    def _find_node_with_free_gpus(self, min_gpus: int) -> str | None:
        """
        Find a node with at least min_gpus available GPUs.
        Prefers non-head nodes since Titan roles typically claim head node GPUs first.
        """
        head_ip = self.host  # Head node IP
        candidates = []

        for node in ray.nodes():
            if not node.get("Alive", False):
                continue
            node_ip = node.get("NodeManagerAddress")
            resources = node.get("Resources", {})
            total_gpu = resources.get("GPU", 0)
            if total_gpu >= min_gpus:
                # Prefer non-head nodes
                is_head = (node_ip == head_ip)
                candidates.append((is_head, node_ip))

        # Sort so non-head nodes come first
        candidates.sort(key=lambda x: x[0])

        if candidates:
            return candidates[0][1]
        return None

    async def spawn_role(self, name: str) -> None:
        if name in self.roles:
            return

        role = self.plan.roles[name]

        if role.kind == "titan":
            port = self.titan_ports[name]
            # Run blocking create_titan_group in thread pool to allow parallel spawning
            loop = asyncio.get_event_loop()
            handle = await loop.run_in_executor(
                None,
                lambda: create_titan_group(
                    config=role.config,
                    name=name,
                    world_size=role.world_size,
                    port=port,
                )
            )
            self.roles[name] = handle
            print(f"[runtime] spawned titan role={name} world_size={role.world_size} rendezvous_port={port}")
            return

        if role.kind == "vllm":
            # Find a node with enough free GPUs for vLLM's TP workers.
            # vLLM creates its own placement group internally, so we just need
            # to schedule the coordinator actor on the right node.
            tp_size = role.config.get("tensor_parallel_size", 1)
            target_node = self._find_node_with_free_gpus(tp_size)

            scheduling_opts = {"num_gpus": 0}
            if target_node:
                scheduling_opts["resources"] = {f"node:{target_node}": 0.001}
                print(f"[runtime] scheduling vllm role={name} on node {target_node}")

            actor = VLLMEngineRank.options(**scheduling_opts).remote(
                engine_kwargs=role.config,
            )
            # Run blocking ray.get in thread pool to allow parallel spawning
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: ray.get(actor.ready.remote()))
            self.roles[name] = handle = VLLMHandle(actor, name=name)
            print(f"[runtime] spawned vllm role={name}")
            return

        raise ValueError(f"Unknown role kind: {role.kind}")

    async def wire(self, src: str, dst: str) -> None:
        await self.spawn_role(src)
        await self.spawn_role(dst)

        ch = self.plan.channel(src, dst)
        port = self.channel_ports[ch.name]

        print(
            f"[runtime] wiring channel={ch.name} src={src} dst={dst} "
            f"port={port} world_size={ch.world_size} offsets={ch.offsets} src_rank={ch.src_rank}"
        )

        await asyncio.gather(
            self._join_role_to_channel(src, ch, port),
            self._join_role_to_channel(dst, ch, port),
        )

        print(f"[runtime] wired channel={ch.name}")

    async def _join_role_to_channel(self, role_name: str, ch, port: int) -> None:
        role = self.plan.roles[role_name]

        if role.kind == "titan":
            handle = self.roles[role_name]
            base = ch.offsets[role_name]
            futs = [
                actor.add_sync_channel.remote(
                    channel_name=ch.name,
                    host=self.host,
                    port=port,
                    world_size=ch.world_size,
                    my_rank=base + local_rank,
                )
                for local_rank, actor in enumerate(handle.actors)
            ]
            await asyncio.gather(*futs)
            return

        # vLLM: join external NCCL sync group via collective RPC to workers
        vllm_actor = self.roles[role_name]._actor
        await vllm_actor.join_sync.remote(
            host=self.host,
            port=port,
            world_size=ch.world_size,
            rank=ch.offsets[role_name],
        )
