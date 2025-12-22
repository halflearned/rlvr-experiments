
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
        namespace = _stable_hash_u32(run_name) % 200  # “200 namespaces” is plenty for most clusters

        titan_roles = [n for n, r in plan.roles.items() if r.kind == "titan"]
        channel_names = [ch.name for ch in plan.channels.values()]

        titan_ports = _port_map(base=29500, stride=200, namespace=namespace, names=titan_roles)
        channel_ports = _port_map(base=51200, stride=200, namespace=namespace, names=channel_names)

        # One-time log of the final wiring decisions.
        print(f"[runtime] run.name={run_name!r} host={host} namespace={namespace}")
        for rn in sorted(titan_ports):
            print(f"[runtime] titan rendezvous port role={rn} port={titan_ports[rn]}")
        for cn in sorted(channel_ports):
            print(f"[runtime] sync channel port channel={cn} port={channel_ports[cn]}")

        # TODO: Change load_plan to load rollout buffer args, if any
        buffer = RolloutBuffer(maxsize=0, max_reads=1)

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
        # Spawn all roles in parallel - each role uses different GPUs so no collision
        await asyncio.gather(*[
            self.spawn_role(name) for name in self.plan.roles.keys()
        ])

        if wire:
            # Wire up all channels
            for src, dst in self.plan.channels:
                await self.wire(src, dst)

        return self

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
            # Note: num_gpus=1 is for the coordinator actor only.
            # vLLM internally spawns additional Ray workers for TP>1.
            actor = VLLMEngineRank.options(num_gpus=1).remote(
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