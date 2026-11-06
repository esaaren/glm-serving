#!/usr/bin/env bash
set -euo pipefail

echo "=== Configuring Google Cloud GPUDirect-TCPXO (Fastrak + RxDM) on A3-Mega ==="
gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin https://us-docker.pkg.dev

sudo mkdir -p /var/lib/tcpxo/lib64 /run/tcpxo

# 1. Install NCCL GPUDirect-TCPXO / Fastrak plugin into /var/lib/tcpxo/lib64
echo "Installing NCCL GPUDirect-TCPXO plugin..."
sudo docker rm -f nccl-tcpxo-installer 2>/dev/null || true
sudo docker run --rm --name nccl-tcpxo-installer \
  --privileged \
  --net=host \
  -v /var/lib/tcpxo/lib64:/var/lib/tcpxo/lib64 \
  us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/nccl-plugin-gpudirecttcpx-dev:latest \
  install --install-nccl

ls -la /var/lib/tcpxo/lib64

# 2. Launch the GPUDirect-TCPXO Receive Data Path Manager (RxDM) daemon
echo "Starting GPUDirect-TCPXO RxDM daemon (tcpgpudmarxd)..."
sudo docker rm -f tcpxo-rxdm 2>/dev/null || true

sudo docker run -d --name tcpxo-rxdm \
  --restart=unless-stopped \
  --privileged \
  --net=host \
  --ipc=host \
  --gpus all \
  -v /var/lib/tcpxo/lib64:/var/lib/tcpxo/lib64 \
  -v /run/tcpxo:/run/tcpxo \
  -v /tmp:/tmp \
  us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/tcpgpudmarxd-dev:latest \
  --num_hops=2 --stderrthreshold=0

sleep 4
sudo docker ps --filter name=tcpxo-rxdm
sudo docker logs --tail 25 tcpxo-rxdm || true
