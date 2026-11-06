#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "ERROR: PROJECT_ID is not set and no active gcloud project was found. Set PROJECT_ID or run 'gcloud config set project <project-id>'." >&2
  exit 1
fi
ZONE="${ZONE:-us-west1-b}"
REGION="${ZONE%-*}"
PREFIX="${PREFIX:-glm-a3mega}"
CLUSTER_NAME="${CLUSTER_NAME:-glm-a3mega-gke}"
GPU_POOL_NAME="${GPU_POOL_NAME:-a3-mega-spot-pool}"
NUM_GPU_NODES="${NUM_GPU_NODES:-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS_DIR="${SCRIPT_DIR}/../manifests"

echo "=== 1. Ensuring 8 GPUDirect-TCPXO VPC Networks, Subnets & Firewall Rules (MTU 8244) Exist ==="
for i in $(seq 1 8); do
  NET_NAME="${PREFIX}-net-${i}"
  SUB_NAME="${PREFIX}-sub-${i}"
  FW_NAME="${PREFIX}-fw-${i}"
  CIDR="192.168.${i}.0/24"

  if ! gcloud compute networks describe "${NET_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "Creating VPC network ${NET_NAME}..."
    gcloud compute networks create "${NET_NAME}" \
      --project="${PROJECT_ID}" \
      --subnet-mode=custom \
      --mtu=8244 \
      --bgp-routing-mode=regional
  fi

  if ! gcloud compute networks subnets describe "${SUB_NAME}" --project="${PROJECT_ID}" --region="${REGION}" >/dev/null 2>&1; then
    echo "Creating subnet ${SUB_NAME} (${CIDR})..."
    gcloud compute networks subnets create "${SUB_NAME}" \
      --project="${PROJECT_ID}" \
      --network="${NET_NAME}" \
      --region="${REGION}" \
      --range="${CIDR}"
  fi

  if ! gcloud compute firewall-rules describe "${FW_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "Creating firewall rule ${FW_NAME}..."
    gcloud compute firewall-rules create "${FW_NAME}" \
      --project="${PROJECT_ID}" \
      --network="${NET_NAME}" \
      --direction=INGRESS \
      --action=ALLOW \
      --rules=tcp:0-65535,udp:0-65535,icmp \
      --source-ranges=192.168.0.0/16,10.0.0.0/8
  fi
done

echo "=== 2. Creating GKE Cluster ${CLUSTER_NAME} (Dataplane V2 + Multi-Networking) in ${ZONE} ==="
if ! gcloud container clusters describe "${CLUSTER_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" >/dev/null 2>&1; then
  gcloud beta container clusters create "${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${ZONE}" \
    --enable-dataplane-v2 \
    --enable-ip-alias \
    --enable-multi-networking \
    --no-enable-autoupgrade \
    --machine-type=e2-standard-8 \
    --num-nodes=2 \
    --scopes="https://www.googleapis.com/auth/cloud-platform"
fi

echo "=== 3. Fetching Cluster Credentials ==="
gcloud container clusters get-credentials "${CLUSTER_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}"

echo "=== 4. Creating Spot A3-Mega GPU Node Pool ${GPU_POOL_NAME} (${NUM_GPU_NODES}x a3-megagpu-8g, 16x Local NVMe SSD) ==="
if ! gcloud container node-pools describe "${GPU_POOL_NAME}" --cluster="${CLUSTER_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" >/dev/null 2>&1; then
  gcloud beta container node-pools create "${GPU_POOL_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${ZONE}" \
    --cluster="${CLUSTER_NAME}" \
    --machine-type=a3-megagpu-8g \
    --num-nodes="${NUM_GPU_NODES}" \
    --spot \
    --enable-gvnic \
    --ephemeral-storage-local-ssd count=16 \
    --accelerator=type=nvidia-h100-mega-80gb,count=8,gpu-driver-version=LATEST \
    --additional-node-network network=${PREFIX}-net-1,subnetwork=${PREFIX}-sub-1 \
    --additional-node-network network=${PREFIX}-net-2,subnetwork=${PREFIX}-sub-2 \
    --additional-node-network network=${PREFIX}-net-3,subnetwork=${PREFIX}-sub-3 \
    --additional-node-network network=${PREFIX}-net-4,subnetwork=${PREFIX}-sub-4 \
    --additional-node-network network=${PREFIX}-net-5,subnetwork=${PREFIX}-sub-5 \
    --additional-node-network network=${PREFIX}-net-6,subnetwork=${PREFIX}-sub-6 \
    --additional-node-network network=${PREFIX}-net-7,subnetwork=${PREFIX}-sub-7 \
    --additional-node-network network=${PREFIX}-net-8,subnetwork=${PREFIX}-sub-8 \
    --no-enable-autoupgrade \
    --scopes="https://www.googleapis.com/auth/cloud-platform"
fi

echo "=== 5. Applying GKE Multi-Network CRDs (vpc1..vpc8), GPUDirect-TCPXO Installer & NRI Device Injector ==="
kubectl apply -f "${MANIFESTS_DIR}/multinetwork-networks.yaml"
kubectl apply -f "${MANIFESTS_DIR}/nccl-tcpxo-installer.yaml"
kubectl apply -f "${MANIFESTS_DIR}/nri-device-injector.yaml"

echo "=== 6. Waiting for GPUDirect-TCPXO Installer DaemonSet to Complete on All GPU Nodes ==="
kubectl rollout status daemonset/nccl-tcpxo-installer -n kube-system --timeout=600s
kubectl rollout status daemonset/device-injector -n kube-system --timeout=300s
kubectl get nodes -o wide
