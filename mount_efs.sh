#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIGURE THIS VALUE ONCE
########################################
FS_ID="fs-0f0f759ca9c245f18"   # <--- update if you ever change EFS
MOUNT_POINT="/efs"

########################################
# 1. Detect AWS region (NOT AZ)
########################################
echo "[INFO] Detecting region..."
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
echo "[INFO] Region: ${REGION}"

########################################
# 2. Install NFS client
########################################
echo "[INFO] Installing NFS client..."
sudo apt update -y
sudo apt install -y nfs-common

########################################
# 3. Create mount point if missing
########################################
if [ ! -d "${MOUNT_POINT}" ]; then
    echo "[INFO] Creating mount directory at ${MOUNT_POINT}..."
    sudo mkdir -p "${MOUNT_POINT}"
fi

########################################
# 4. Mount EFS
########################################
echo "[INFO] Mounting EFS ${FS_ID} at ${MOUNT_POINT}..."
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 \
    ${FS_ID}.efs.${REGION}.amazonaws.com:/ ${MOUNT_POINT}

########################################
# 5. Fix permissions for ubuntu user
########################################
echo "[INFO] Fixing ownership..."
sudo chown ubuntu:ubuntu "${MOUNT_POINT}"

########################################
# 6. Make mount persistent (fstab)
########################################
echo "[INFO] Updating /etc/fstab..."
grep -q "${FS_ID}.efs.${REGION}.amazonaws.com:/" /etc/fstab || \
echo "${FS_ID}.efs.${REGION}.amazonaws.com:/ ${MOUNT_POINT} nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,_netdev 0 0" \
| sudo tee -a /etc/fstab

echo "[INFO] EFS mount complete and persistent."
df -h | grep ${MOUNT_POINT}


##### Other quality of life improvements
if ! grep -q 'history-search-backward' ~/.bashrc; then
  cat >> ~/.bashrc << 'EOF'

# Up/down arrow command history search
bind '"\e[A": history-search-backward'
bind '"\e[B": history-search-forward'
bind '"\C-p": history-search-backward'
bind '"\C-n": history-search-forward'
EOF
fi


# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install conda
curl -s https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
bash /tmp/miniconda.sh -b -p $HOME/miniconda && \
rm /tmp/miniconda.sh && \
$HOME/miniconda/bin/conda init bash
