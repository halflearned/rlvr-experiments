# RVLR experiments

## Launch cluster

The following will launch n EC2 machines with the right configuration to start experimenting.

```bash
# 0. Clone repo
git clone https://github.com/halflearned/rlvr-experiments.git
cd rlvr-experiments

# 1. Create SSH key (if needed)
aws ec2 create-key-pair --key-name rlvr-key --query 'KeyMaterial' --output text > ~/.ssh/rlvr-key.pem
chmod 400 ~/.ssh/rlvr-key.pem

# 2. Create cluster (waits for instances)
python infra/launch.py create -n 2

# 3. Check status / get IPs
python infra/launch.py status

# 4. SSH and start Ray on head node
ssh -i ~/.ssh/rlvr-key.pem ubuntu@<HEAD_IP>
cd /efs/rlvr-experiments && source .venv/bin/activate
ray start --head
python infra/launch.py set-head $(hostname -I | awk '{print $1}')  # save head IP

# 5. On worker nodes, join Ray (status shows the command)
ray start --address=<HEAD_PRIVATE_IP>:6379

# 6. Run training
python entrypoints/train_grpo.py configs/qwen3-06B-base.yaml

# 7. Cleanup
python infra/launch.py delete
```

## Scale cluster

**Scale up:**
```bash
python infra/launch.py scale -n 4  # set total instances
python infra/launch.py status      # shows head node and ray command

# SSH to new node and join Ray (status shows the exact command)
ssh -i ~/.ssh/rlvr-key.pem ubuntu@<NEW_NODE_IP>
cd /efs/rlvr-experiments && source .venv/bin/activate
ray start --address=<HEAD_PRIVATE_IP>:6379
```

**Scale down:**
```bash
# Stop Ray on ALL worker nodes first (AWS picks which to terminate)
# On each non-head node:
ray stop

# Then scale from local machine
python infra/launch.py scale -n 2

# Remaining workers rejoin Ray
ray start --address=<HEAD_PRIVATE_IP>:6379
```
