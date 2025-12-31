"""Docker-based code executor with security isolation.

Each execution runs in a fresh container with:
- No network access
- Read-only filesystem (except tmpfs /tmp)
- Memory, CPU, and PID limits
- Dropped capabilities
- Non-root user

This ensures deterministic, isolated execution suitable for RL training.
"""

import asyncio
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: float
    error: str | None = None

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

    # Additional timeout buffer for container overhead
    # External timeout = timeout + timeout_buffer
    timeout_buffer: float = 2.0

    # Use file-based code injection instead of -c flag
    # More robust for code with special characters/quotes
    use_file_injection: bool = True


class CodeExecutor:
    """Execute Python code in isolated Docker containers.

    Each execution creates a fresh container, ensuring deterministic results.
    Containers are configured with strict security limits.

    Example:
        executor = CodeExecutor()
        result = await executor.execute("print('hello')")
        assert result.success
        assert result.stdout.strip() == "hello"
    """

    def __init__(self, config: ExecutorConfig | None = None):
        self.config = config or ExecutorConfig()
        self._docker_available: bool | None = None

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

    def _build_docker_command(self, code: str, code_file: Path | None = None) -> list[str]:
        """Build the docker run command with all security flags."""
        cfg = self.config

        cmd = [
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

        if code_file is not None:
            # File-based injection: mount the code file read-only
            cmd.extend([
                "-v", f"{code_file}:/code.py:ro",
                cfg.image,
                "timeout", str(cfg.timeout),
                "python", "/code.py",
            ])
        else:
            # Command-line injection
            cmd.extend([
                cfg.image,
                "timeout", str(cfg.timeout),
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
        """Execute Python code and return the result.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with stdout, stderr, exit code, and timing info
        """
        start_time = time.perf_counter()
        external_timeout = self.config.timeout + self.config.timeout_buffer
        use_docker = await self._check_docker()
        use_file = self.config.use_file_injection

        # Create temp file if using file-based injection
        # We use delete=False and clean up manually to avoid issues on Windows
        # and to ensure the file exists when Docker mounts it
        code_file = None
        try:
            if use_file:
                # Create temp file with .py extension
                fd = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    delete=False,
                    encoding="utf-8",
                )
                fd.write(code)
                fd.close()
                code_file = Path(fd.name)
                # Make readable by the container's nobody user (65534)
                code_file.chmod(0o644)

            if use_docker:
                cmd = self._build_docker_command(code, code_file)
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

                # Check if internal timeout triggered (exit code 124)
                if exit_code == 124:
                    timed_out = True

            except asyncio.TimeoutError:
                # External timeout - kill the process
                proc.kill()
                await proc.wait()
                stdout_bytes, stderr_bytes = b"", b""
                timed_out = True
                exit_code = -1

            duration_ms = (time.perf_counter() - start_time) * 1000

            return ExecutionResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                exit_code=exit_code,
                timed_out=timed_out,
                duration_ms=duration_ms,
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
            )

        finally:
            # Clean up temp file
            if code_file is not None:
                try:
                    code_file.unlink()
                except OSError:
                    pass

