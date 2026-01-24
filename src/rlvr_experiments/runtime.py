from __future__ import annotations

import asyncio
import os
import socket
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

import ray

from .config_plan import Plan, load_plan
from .buffer import DataBuffer
from .rollout_logger import init_rollout_logger
from .sample_logger import init_sample_logger
from .titan_actor import create_titan_group
from .tracer import init_global_tracer, get_tracer
from .vllm_engine_actor import VLLMEngineRank, VLLMHandle


def _find_free_port(start: int = 29500, end: int = 65535) -> int:
    """Find an available port by actually binding to it."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


def _find_free_ports(count: int, start: int = 29500) -> list[int]:
    """Find multiple consecutive-ish free ports."""
    ports = []
    current = start
    while len(ports) < count:
        try:
            port = _find_free_port(start=current, end=65535)
            ports.append(port)
            current = port + 1
        except RuntimeError:
            break
    if len(ports) < count:
        raise RuntimeError(f"Could only find {len(ports)} free ports, needed {count}")
    return ports


@dataclass
class Runtime:
    plan: Plan
    host: str
    roles: Dict[str, Any]               # role_name -> handle
    titan_ports: Dict[str, int]         # titan role rendezvous ports
    channel_ports: Dict[str, int]       # channel_name -> sync port
    namespace: int
    buffer: Any
    run_name: str = ""                  # run folder name (config + timestamp)
    run_dir: str = ""                   # absolute path to run folder
    run_timestamp: str = ""             # unique timestamp for this run (YYYYMMDD_HHMMSS)

    @property
    def tracer(self):
        """Access the global tracer (if configured)."""
        return get_tracer()

    @property
    def trace_dir(self) -> str:
        """Get directory containing trace files."""
        tracer = self.tracer
        if tracer and tracer.path:
            return os.path.dirname(tracer.path) or "."
        return ""

    @classmethod
    async def from_plan(cls, plan_path: str) -> "Runtime":
        import os

        plan = load_plan(plan_path)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        config_stem = Path(plan_path).stem.replace(" ", "_")
        run_name = f"{config_stem}_{timestamp}"

        results_root = Path(os.environ.get("SM_MODEL_DIR", "results")).resolve()

        # Ensure unique run dir if a collision somehow occurs
        run_dir = results_root / run_name
        if run_dir.exists():
            suffix = 1
            while (results_root / f"{run_name}_{suffix}").exists():
                suffix += 1
            run_name = f"{run_name}_{suffix}"
            run_dir = results_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Standard run subfolders
        traces_dir = run_dir / "traces"
        checkpoints_dir = run_dir / "checkpoints"
        patches_dir = run_dir / "patches"
        traces_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        patches_dir.mkdir(parents=True, exist_ok=True)

        # Persist config and a place for notes
        config_dst = run_dir / "config.yaml"
        try:
            shutil.copyfile(plan_path, config_dst)
        except Exception as e:
            print(f"[runtime] WARNING: failed to copy config to {config_dst}: {e}")
        results_md = run_dir / "RESULTS.md"
        if not results_md.exists():
            results_md.write_text("")

        # Capture repo state for reproducibility
        repo_root = None
        git_info: dict[str, object] = {}
        try:
            repo_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd(), text=True
            ).strip()
            git_info["root"] = repo_root
            git_info["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip()
            git_info["branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, text=True
            ).strip()
            status = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True
            )
            (patches_dir / "git_status.txt").write_text(status)

            diff = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
            (patches_dir / "git_diff.patch").write_text(diff)
            diff_cached = subprocess.check_output(["git", "diff", "--cached"], cwd=repo_root, text=True)
            (patches_dir / "git_diff_cached.patch").write_text(diff_cached)

            untracked = [line[3:].strip() for line in status.splitlines() if line.startswith("?? ")]
            (patches_dir / "git_untracked.txt").write_text("\n".join(untracked))
            if untracked:
                tar_path = patches_dir / "untracked.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    for rel_path in untracked:
                        abs_path = Path(repo_root) / rel_path
                        if abs_path.exists():
                            tar.add(abs_path, arcname=rel_path)
                git_info["untracked_archive"] = str(tar_path)
        except Exception as e:
            print(f"[runtime] WARNING: failed to capture git state: {e}")

        run_meta = {
            "run_name": run_name,
            "timestamp": timestamp,
            "config_path": str(Path(plan_path).resolve()),
            "results_root": str(results_root),
            "run_dir": str(run_dir),
            "host": socket.gethostname(),
            "sm_model_dir": os.environ.get("SM_MODEL_DIR"),
            "git": git_info,
        }
        (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2))

        # Expose run dir to other components (e.g., checkpoint dir resolver)
        os.environ["RLVR_RUN_DIR"] = str(run_dir)

        # Force checkpoints to land under this run directory
        for role in plan.roles.values():
            if role.kind == "titan":
                ckpt_cfg = role.config.get("checkpoint", {})
                ckpt_cfg["folder"] = str(checkpoints_dir)
                role.config["checkpoint"] = ckpt_cfg

        # Initialize tracing with stable filenames inside run_dir
        trace_path = getattr(plan, "trace_path", None) or str(traces_dir / "trace.jsonl")
        if trace_path.endswith(".json"):
            trace_path = trace_path[:-5] + ".jsonl"
        if not os.path.isabs(trace_path):
            trace_path = str(run_dir / trace_path)

        init_global_tracer(trace_path)
        sample_path = str(traces_dir / "samples.jsonl")
        init_sample_logger(sample_path)
        rollout_path = str(traces_dir / "rollouts.jsonl")
        init_rollout_logger(rollout_path)
        print(f"[runtime] tracing to {trace_path}")
        print(f"[runtime] sample logging to {sample_path}")
        print(f"[runtime] rollout logging to {rollout_path}")

        # Emit run configuration metadata for visualization
        tracer = get_tracer()
        if tracer:
            # Extract key config values for the overview panel
            model_path = plan.model.get("path", "") if hasattr(plan, "model") else ""
            dataset = plan.data.get("dataset", "") if hasattr(plan, "data") else ""
            batch_size = plan.data_iter.get("batch_size", 0) if hasattr(plan, "data_iter") else 0
            prompts_per_batch = plan.training.get("prompts_per_batch", 0) if hasattr(plan, "training") else 0
            n_completions = plan.sampling.get("n", 1) if hasattr(plan, "sampling") else 1
            config_file = os.path.basename(plan_path)

            # Extract parallelism info for each Titan role
            titan_parallelism = {}
            for role_name, role_plan in plan.roles.items():
                if role_plan.kind == "titan":
                    p = role_plan.config.get("parallelism", {})
                    titan_parallelism[role_name] = {
                        "dp_replicate": p.get("data_parallel_replicate_degree", 1),
                        "dp_shard": p.get("data_parallel_shard_degree", 1),
                        "tp": p.get("tensor_parallel_degree", 1),
                    }

            tracer.meta(
                config_file=config_file,
                run_name=run_name,
                model_path=model_path,
                dataset=dataset,
                vllm_batch_size=batch_size,
                prompts_per_batch=prompts_per_batch,
                n_completions=n_completions,
                titan_parallelism=titan_parallelism,
                start_time=datetime.now().isoformat(timespec='seconds'),
            )

        if not ray.is_initialized():
            # Set up runtime_env to make rlvr_experiments available to all workers
            # This is needed for SageMaker where source is in /opt/ml/code
            import os
            runtime_env = {}
            src_path = "/opt/ml/code/src"
            if os.path.exists(src_path):
                # SageMaker environment - add src to Python path for all workers
                # Also fix cuDNN version mismatch: PyTorch 2.9+cu128 bundles cuDNN 9.x,
                # but the base AWS DLC image has an older system cuDNN. Prepend the
                # nvidia pip packages to LD_LIBRARY_PATH so PyTorch finds them.
                nvidia_libs = "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"
                existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
                ld_path = f"{nvidia_libs}:{existing_ld}" if existing_ld else nvidia_libs
                runtime_env["env_vars"] = {
                    "PYTHONPATH": src_path,
                    "LD_LIBRARY_PATH": ld_path,
                }
            ray.init(address="auto", runtime_env=runtime_env)
        host = ray.util.get_node_ip_address()

        titan_roles = sorted([n for n, r in plan.roles.items() if r.kind == "titan"])
        channel_names = sorted([ch.name for ch in plan.channels.values()])

        # Dynamically find free ports instead of hash-based allocation
        # This avoids collisions with ports still in TIME_WAIT from crashed runs
        total_ports_needed = len(titan_roles) + len(channel_names)
        all_ports = _find_free_ports(total_ports_needed, start=29500)

        titan_ports = {name: all_ports[i] for i, name in enumerate(titan_roles)}
        channel_ports = {name: all_ports[len(titan_roles) + i] for i, name in enumerate(channel_names)}

        # One-time log of the final wiring decisions.
        print(f"[runtime] run.name={run_name!r} host={host}")
        for rn in sorted(titan_ports):
            print(f"[runtime] titan rendezvous port role={rn} port={titan_ports[rn]}")
        for cn in sorted(channel_ports):
            print(f"[runtime] sync channel port channel={cn} port={channel_ports[cn]}")

        buffer = DataBuffer(**plan.buffer)

        return cls(
            plan=plan,
            host=host,
            roles={},
            titan_ports=titan_ports,
            channel_ports=channel_ports,
            namespace=0,  # No longer used, kept for compatibility
            buffer=buffer,
            run_name=run_name,
            run_dir=str(run_dir),
            run_timestamp=timestamp,
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
        Prefers head node to pack GPUs together and minimize cross-node communication.
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
                # Prefer head node to pack GPUs together
                is_head = (node_ip == head_ip)
                candidates.append((not is_head, node_ip))  # False sorts before True

        # Sort so head node comes first
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
            # Find nodes with enough free GPUs for vLLM's TP workers.
            # vLLM creates its own placement group internally, so we just need
            # to schedule the coordinator actor on the right node.
            tp_size = role.config.get("tensor_parallel_size", 1)
            dp_size = role.data_parallel_size

            # Spawn all actors first, then wait for all to be ready in parallel
            actors = []
            for replica_id in range(dp_size):
                target_node = self._find_node_with_free_gpus(tp_size)

                scheduling_opts = {"num_gpus": 0}
                if target_node:
                    scheduling_opts["resources"] = {f"node:{target_node}": 0.001}
                    print(f"[runtime] scheduling vllm role={name} replica={replica_id} on node {target_node}")

                actor = VLLMEngineRank.options(**scheduling_opts).remote(
                    engine_kwargs=role.config,
                    replica_id=replica_id,
                )
                actors.append(actor)
                print(f"[runtime] spawning vllm role={name} replica={replica_id}")

            # Wait for all replicas to be ready in parallel
            print(f"[runtime] waiting for {len(actors)} vllm replicas to be ready...")
            loop = asyncio.get_event_loop()
            ready_refs = [a.ready.remote() for a in actors]
            await loop.run_in_executor(None, ray.get, ready_refs)
            print(f"[runtime] all {len(actors)} vllm replicas ready for role={name}")

            max_concurrent = role.config.get("max_concurrent_per_replica", 8)
            self.roles[name] = VLLMHandle(actors, name=name, max_concurrent_per_replica=max_concurrent)
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

        # IMPORTANT: Dispatch all add_sync_channel calls FIRST, then await.
        # NCCL init is a collective that blocks until all ranks join.
        # If we await one role before dispatching the other, we deadlock.
        src_refs = self._get_channel_join_futures(src, ch, port)
        dst_refs = self._get_channel_join_futures(dst, ch, port)
        all_refs = src_refs + dst_refs

        # Ray ObjectRefs need ray.get() - run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, ray.get, all_refs)

        print(f"[runtime] wired channel={ch.name}")

    def _get_channel_join_futures(self, role_name: str, ch, port: int) -> list:
        """
        Return list of Ray ObjectRefs for add_sync_channel calls.
        Does NOT await - caller must gather all futures together.
        """
        role = self.plan.roles[role_name]

        if role.kind == "titan":
            handle = self.roles[role_name]
            base = ch.offsets[role_name]
            return [
                actor.add_sync_channel.remote(
                    channel_name=ch.name,
                    host=self.host,
                    port=port,
                    world_size=ch.world_size,
                    rank=base + local_rank,
                )
                for local_rank, actor in enumerate(handle.actors)
            ]

        elif role.kind == "vllm":
            # vLLM: join external NCCL sync group via collective RPC to workers
            # Each replica's workers get consecutive rank ranges
            handle = self.roles[role_name]
            base = ch.offsets[role_name]
            workers_per_replica = role.world_size  # TP workers per instance

            futs = []
            for replica_id, actor in enumerate(handle._actors):
                replica_base = base + (replica_id * workers_per_replica)
                futs.append(
                    actor.add_sync_channel.remote(
                        channel_name=ch.name,
                        host=self.host,
                        port=port,
                        world_size=ch.world_size,
                        rank=replica_base,
                    )
                )
            return futs

        else:
            raise ValueError(f"Unknown role kind: {role.kind}")
