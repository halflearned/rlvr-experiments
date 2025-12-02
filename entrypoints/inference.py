#!/usr/bin/env python3
import argparse
import sys
import tomllib
import subprocess


def build_vllm_command(cfg: dict) -> list[str]:
    server_cfg = cfg["vllm_server"]
    model = server_cfg["model"]
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)
    module = server_cfg.get(
        "module", "vllm.entrypoints.openai.api_server"
    )

    cmd: list[str] = [sys.executable, "-m", module]

    # required args
    cmd += ["--model", str(model)]
    cmd += ["--host", str(host)]
    cmd += ["--port", str(port)]

    # other args
    skip_keys = {"model", "host", "port", "module"}
    for key, value in server_cfg.items():
        if key in skip_keys:
            continue
        if value is None:
            continue

        flag = "--" + key.replace("_", "-")

        if isinstance(value, bool):  # booleans
            cmd += [flag, "true" if value else "false"]
        elif isinstance(value, (list, tuple)):  # sequences
            cmd += [flag, ",".join(str(v) for v in value)]
        else:
            cmd += [flag, str(value)]

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    cmd = build_vllm_command(cfg)

    print("Launching vLLM server with command:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
