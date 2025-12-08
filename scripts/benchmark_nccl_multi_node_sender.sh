export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens32
export NCCL_DEBUG_SUBSYS=INIT,NET
export FI_PROVIDER=efa
export MASTER_ADDR=172.31.17.116
MASTER_PORT=29500

torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  benchmark_nccl_multi_node.py \
  --op all_reduce \
  --size-gb 1.0 \
  --iters 20