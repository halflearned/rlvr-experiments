#!/usr/bin/env python3
"""Launch and manage RLVR cluster via CloudFormation."""

import argparse
import subprocess
import json
import time
import sys
from pathlib import Path

HEAD_FILE = Path("/efs/rlvr-experiments/.head_node")

def run(cmd, capture=True):
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if result.returncode != 0 and capture:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip() if capture else None

def get_head():
    """Get saved head node IP (private), or None if not set."""
    if HEAD_FILE.exists():
        return HEAD_FILE.read_text().strip()
    return None

def set_head(private_ip):
    """Save head node IP to shared file."""
    HEAD_FILE.write_text(private_ip)

def get_instances(stack_name):
    out = run(f"""aws ec2 describe-instances \
        --filters "Name=tag:aws:cloudformation:stack-name,Values={stack_name}" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[*].Instances[*].[PublicIpAddress,PrivateIpAddress]' \
        --output json""")
    return [ip for r in json.loads(out) for ip in r if ip[0]]

def cmd_create(args):
    print(f"Creating stack '{args.stack_name}' with {args.instances} instances...")
    params = f"ParameterKey=KeyName,ParameterValue={args.key} ParameterKey=InstanceCount,ParameterValue={args.instances}"
    if args.vpc:
        params += f" ParameterKey=ExistingVpcId,ParameterValue={args.vpc}"
    if args.subnet:
        params += f" ParameterKey=ExistingSubnetId,ParameterValue={args.subnet}"
    run(f"""aws cloudformation create-stack \
        --stack-name {args.stack_name} \
        --template-body file://infra/cloudformation.yaml \
        --capabilities CAPABILITY_IAM \
        --parameters {params}""", capture=False)
    print("Waiting for stack creation...")
    run(f"aws cloudformation wait stack-create-complete --stack-name {args.stack_name}", capture=False)
    print("Stack created. Waiting for instances...")
    for _ in range(30):
        instances = get_instances(args.stack_name)
        if len(instances) >= args.instances:
            break
        time.sleep(10)
    cmd_status(args)

def cmd_status(args):
    instances = get_instances(args.stack_name)
    if not instances:
        print("No running instances found.")
        return
    head = get_head()
    print(f"\n{'Role':<8} {'Public IP':<18} {'Private IP':<18}")
    print("-" * 46)
    for pub, priv in instances:
        role = "HEAD" if priv == head else ""
        print(f"{role:<8} {pub:<18} {priv:<18}")
    if head:
        head_pub = next((pub for pub, priv in instances if priv == head), None)
        if head_pub:
            print(f"\nHead: ssh -i ~/.ssh/{args.key}.pem ubuntu@{head_pub}")
            print(f"Ray:  ray start --address={head}:6379")
    else:
        print(f"\nNo head node set. Run 'ray start --head' on one node, then:")
        print(f"  python infra/launch.py set-head <PRIVATE_IP>")

def cmd_set_head(args):
    set_head(args.ip)
    print(f"Head node set to {args.ip}")

def cmd_scale(args):
    asg_name = f"{args.stack_name}-asg"
    print(f"Scaling to {args.instances} instances...")
    run(f"aws autoscaling set-desired-capacity --auto-scaling-group-name {asg_name} --desired-capacity {args.instances}", capture=False)
    print(f"Waiting for {args.instances} instances...")
    for _ in range(30):
        instances = get_instances(args.stack_name)
        if len(instances) >= args.instances:
            break
        time.sleep(10)
    cmd_status(args)

def cmd_delete(args):
    print(f"Deleting stack '{args.stack_name}'...")
    run(f"aws cloudformation delete-stack --stack-name {args.stack_name}", capture=False)
    print("Stack deletion initiated.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RLVR cluster management")
    p.add_argument("--stack-name", default="rlvr-cluster")
    p.add_argument("--key", default="rlvr-key", help="EC2 key pair name")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="Create cluster")
    c.add_argument("-n", "--instances", type=int, default=2)
    c.add_argument("--vpc", help="Existing VPC ID (skip VPC creation)")
    c.add_argument("--subnet", help="Existing Subnet ID (required if --vpc is set)")

    sub.add_parser("status", help="Show instance IPs")

    h = sub.add_parser("set-head", help="Set head node IP")
    h.add_argument("ip", help="Private IP of head node")

    s = sub.add_parser("scale", help="Scale cluster to N instances")
    s.add_argument("-n", "--instances", type=int, required=True)

    sub.add_parser("delete", help="Delete cluster")

    args = p.parse_args()
    {"create": cmd_create, "status": cmd_status, "set-head": cmd_set_head, "scale": cmd_scale, "delete": cmd_delete}[args.cmd](args)
