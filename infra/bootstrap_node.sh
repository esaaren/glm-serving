#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/6] Checking GPU & Network Interfaces ==="
nvidia-smi -L
ip -br addr show

echo "=== [2/6] Setting up 6TB Local NVMe SSD RAID-0 at /mnt/localssd ==="
if ! mountpoint -q /mnt/localssd; then
  sudo apt-get update -qq && sudo apt-get install -y -qq mdadm jq python3-pip python3-venv nvme-cli
  NVME_DEVS=(/dev/disk/by-id/google-local-nvme-ssd-*)
  echo "Found ${#NVME_DEVS[@]} Local NVMe SSDs: ${NVME_DEVS[*]}"
  if [[ -e "${NVME_DEVS[0]}" ]]; then
    if [[ ! -b /dev/md0 ]]; then
      sudo mdadm --create /dev/md0 --level=0 --raid-devices="${#NVME_DEVS[@]}" "${NVME_DEVS[@]}" --force --run
      sudo mkfs.ext4 -F -m 0 /dev/md0
    fi
    sudo mkdir -p /mnt/localssd
    sudo mount -o discard,defaults,nobarrier /dev/md0 /mnt/localssd
  else
    sudo mkdir -p /mnt/localssd
  fi
fi
sudo chmod 777 /mnt/localssd
mkdir -p /mnt/localssd/hf_cache /mnt/localssd/docker /mnt/localssd/results
df -h /mnt/localssd

echo "=== [3/6] Installing Docker + NVIDIA Container Toolkit & Moving data-root to /mnt/localssd/docker ==="
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq curl gnupg ca-certificates docker.io
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit
fi

sudo mkdir -p /etc/docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo jq '. + {"data-root": "/mnt/localssd/docker"}' /etc/docker/daemon.json > /tmp/daemon.json
sudo mv /tmp/daemon.json /etc/docker/daemon.json
sudo systemctl daemon-reload
sudo systemctl enable --now docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER" || true

echo "=== [4/6] Configuring Sysctl & Multi-NIC MTU/Routing for GPUDirect-TCPXO ==="
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
sudo nvidia-smi -pm 1

echo "=== [5/6] Pulling vllm/vllm-openai:nightly & Installing Python Benchmark Tools ==="
sudo docker pull vllm/vllm-openai:nightly
pip3 install --upgrade --quiet huggingface_hub hf_transfer aiohttp

echo "=== [6/6] Bootstrap Complete ==="
