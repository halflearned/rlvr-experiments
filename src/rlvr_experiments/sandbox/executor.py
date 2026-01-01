"""Docker-based code executor with security isolation."""

import asyncio
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

# Counter for unique container names within this process
_container_counter = 0


@dataclass
class ExecutionResult:
    """Result of a code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: float
    error: str | None = None
    # Timing for tracing: when execution actually started (after semaphore acquired)
    actual_start_time: float | None = None  # time.perf_counter() value

    @property
    def success(self) -> bool:
        """True if execution completed successfully (exit code 0, no timeout)."""
        return self.exit_code == 0 and not self.timed_out and self.error is None


@dataclass
class ExecutorConfig:
    """Configuration for the code executor."""

    image: str = "python:3.11-slim"
    memory_limit: str = "256m"
    cpu_limit: float = 1.0
    pids_limit: int = 64
    timeout: float = 10.0
    tmpfs_size: str = "64m"

    # Additional timeout buffer for container startup overhead
    # External timeout = timeout + timeout_buffer
    # Kimi paper reports ~0.12s average startup, so 0.5s should be plenty
    timeout_buffer: float = 0.5

    # Use file-based code injection instead of -c flag
    # More robust for code with special characters/quotes
    use_file_injection: bool = True


class CodeExecutor:
    """Execute Python code in isolated Docker containers."""

    def __init__(self, config: ExecutorConfig | None = None, max_concurrent: int = 1):
        self.config = config or ExecutorConfig()
        self._docker_available: bool | None = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _check_docker(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is not None:
            return self._docker_available

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            self._docker_available = proc.returncode == 0
        except FileNotFoundError:
            self._docker_available = False

        if not self._docker_available:
            logger.warning("Docker not available, falling back to subprocess execution")

        return self._docker_available

    def _build_docker_command(self, code: str, code_file: Path | None = None, container_name: str | None = None) -> list[str]:
        """Build the docker run command with all security flags."""
        cfg = self.config
        # Outer timeout wraps the entire docker command (startup + execution)
        outer_timeout = cfg.timeout + cfg.timeout_buffer

        cmd = [
            "timeout", str(outer_timeout),  # Kill docker if it hangs on startup
            "docker", "run",
            "--rm",                                    # Auto-remove container
            "--network=none",                         # No network access
            "--read-only",                            # Read-only root filesystem
            f"--tmpfs=/tmp:size={cfg.tmpfs_size},mode=1777",  # Writable /tmp
            f"--memory={cfg.memory_limit}",           # Memory limit
            f"--cpus={cfg.cpu_limit}",                # CPU limit
            f"--pids-limit={cfg.pids_limit}",         # Fork bomb protection
            "--cap-drop=ALL",                         # Drop all capabilities
            "--security-opt=no-new-privileges",       # No privilege escalation
            "--user=65534:65534",                     # Run as nobody
        ]

        if container_name:
            cmd.extend(["--name", container_name])

        if code_file is not None:
            # File-based injection: mount the code file read-only
            cmd.extend([
                "-v", f"{code_file}:/code.py:ro",
                cfg.image,
                "timeout", str(cfg.timeout),  # Inner timeout for Python execution
                "python", "/code.py",
            ])
        else:
            # Command-line injection
            cmd.extend([
                cfg.image,
                "timeout", str(cfg.timeout),  # Inner timeout for Python execution
                "python", "-c", code,
            ])

        return cmd

    def _build_subprocess_command(self, code: str, code_file: Path | None = None) -> list[str]:
        """Build a subprocess command (fallback when Docker unavailable)."""
        if code_file is not None:
            return [
                "timeout", str(self.config.timeout),
                "python3", str(code_file),
            ]
        else:
            return [
                "timeout", str(self.config.timeout),
                "python3", "-c", code,
            ]

    async def execute(self, code: str) -> ExecutionResult:
        """Execute Python code and return the result."""
        async with self._semaphore:
            return await self._execute_impl(code)

    def _generate_container_name(self) -> str:
        """Generate a unique container name for this execution."""
        global _container_counter
        _container_counter += 1
        return f"rlvr_{os.getpid()}_{_container_counter}"

    async def _force_kill_container(self, container_name: str):
        """Force kill a container that may be hanging."""
        proc = await asyncio.create_subprocess_exec(
            "docker", "kill", container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    async def _wait_container_gone(self, container_name: str, timeout: float = 5.0):
        """Wait until the container no longer exists."""
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            proc = await asyncio.create_subprocess_exec(
                "docker", "inspect", container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode != 0:
                # Container doesn't exist - we're done
                return
            await asyncio.sleep(0.05)

    async def _execute_impl(self, code: str) -> ExecutionResult:
        """Actual execution logic."""
        start_time = time.perf_counter()
        external_timeout = self.config.timeout + self.config.timeout_buffer
        use_docker = await self._check_docker()
        use_file = self.config.use_file_injection

        container_name = self._generate_container_name() if use_docker else None

        # Create temp file if using file-based injection
        code_file = None
        try:
            if use_file:
                fd = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    delete=False,
                    encoding="utf-8",
                )
                fd.write(code)
                fd.close()
                code_file = Path(fd.name)
                code_file.chmod(0o644)

            if use_docker:
                cmd = self._build_docker_command(code, code_file, container_name)
            else:
                cmd = self._build_subprocess_command(code, code_file)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=external_timeout,
                )
                timed_out = False
                exit_code = proc.returncode or 0

                if exit_code == 124:
                    timed_out = True
                    # timeout command killed docker, but container may still be running
                    if container_name:
                        await self._force_kill_container(container_name)
                        await self._wait_container_gone(container_name, timeout=2.0)

            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                # Forcefully stop the container - it may still be running
                if container_name:
                    await self._force_kill_container(container_name)
                    await self._wait_container_gone(container_name, timeout=2.0)
                stdout_bytes, stderr_bytes = b"", b""
                timed_out = True
                exit_code = -1

            # Note: with --rm, container is already cleaned up when docker run exits.
            # Only wait if we had to force-kill (timeout case).
            # Skipping this for normal execution saves ~50-200ms of polling overhead.

            duration_ms = (time.perf_counter() - start_time) * 1000

            return ExecutionResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                exit_code=exit_code,
                timed_out=timed_out,
                duration_ms=duration_ms,
                actual_start_time=start_time,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=-1,
                timed_out=False,
                duration_ms=duration_ms,
                error=str(e),
                actual_start_time=start_time,
            )

        finally:
            if code_file is not None:
                try:
                    code_file.unlink()
                except OSError:
                    pass

