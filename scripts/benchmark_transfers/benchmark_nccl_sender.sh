# Run as Sender (Client)
# Replace MASTER_ADDR with the actual IP address of the master node
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens32
export NCCL_DEBUG_SUBSYS=INIT,NET
export FI_PROVIDER=efa
export MASTER_ADDR=172.31.17.116
# export MASTER_ADDR=127.0.0.1
python benchmark_nccl.py \
  --role sender \
  --master-addr $MASTER_ADDR \
  --size-gb 2