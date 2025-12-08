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

    cmd = ["python3", "src/vllm_utils/server.py"]  # TODO: better path handling


    # required args
    cmd += ["--model", str(model)]
    cmd += ["--host", str(host)]
    cmd += ["--port", str(port)]

    # other args
    skip_keys = {"model", "host", "port", "module"}
    for key, value in server_cfg.items():  # TODO: fix, this isn't actually looping thru params
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

    from subprocess import run
    run(build_vllm_command(cfg))


if __name__ == "__main__":
    main()
