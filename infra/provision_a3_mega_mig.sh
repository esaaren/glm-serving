#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "ERROR: PROJECT_ID is not set and no active gcloud project was found. Set PROJECT_ID or run 'gcloud config set project <project-id>'." >&2
  exit 1
fi
ZONE="${ZONE:-us-west1-b}"
REGION="${ZONE%-*}"
MACHINE_TYPE="${MACHINE_TYPE:-a3-megagpu-8g}"
NUM_VMS="${NUM_VMS:-2}"
PREFIX="${PREFIX:-glm-a3mega}"
TEMPLATE_NAME="${PREFIX}-tmpl"
MIG_NAME="${PREFIX}-mig"

echo "=== 1. Creating 8 GPUDirect-TCPXO Data VPC Networks, Subnets & Firewall Rules (MTU 8244) in ${REGION} ==="
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

echo "=== 2. Creating Instance Template ${TEMPLATE_NAME} for ${MACHINE_TYPE} (9 gVNICs) ==="
if ! gcloud compute instance-templates describe "${TEMPLATE_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute instance-templates create "${TEMPLATE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --machine-type="${MACHINE_TYPE}" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --maintenance-policy=TERMINATE \
    --image-family=ubuntu-accelerator-2204-amd64-with-nvidia-580 \
    --image-project=ubuntu-os-accelerator-images \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --network-interface=nic-type=GVNIC,network=default,subnet=default \
    --network-interface=nic-type=GVNIC,network=${PREFIX}-net-1,subnet=${PREFIX}-sub-1,no-address \
    --network-interface=nic-type=GVNIC,network=${PREFIX}-net-2,subnet=${PREFIX}-sub-2,no-address \
    --network-interface=nic-type=GVNIC,network=${PREFIX}-net-3,subnet=${PREFIX}-sub-3,no-address \
    --network-interface=nic-type=GVNIC,network=${PREFIX}-net-4,subnet=${PREFIX}-sub-4,no-address \
    --network-interface=nic-type=GVNIC,network=${PREFIX}-net-5,subnet=${PREFIX}-sub-5,no-address \
    --network-interface=nic-type=GVNIC,network=${PREFIX}-net-6,subnet=${PREFIX}-sub-6,no-address \
    --network-interface=nic-type=GVNIC,network=${PREFIX}-net-7,subnet=${PREFIX}-sub-7,no-address \
    --network-interface=nic-type=GVNIC,network=${PREFIX}-net-8,subnet=${PREFIX}-sub-8,no-address
fi

echo "=== 3. Creating Managed Instance Group ${MIG_NAME} (Size=${NUM_VMS}) in ${ZONE} ==="
if ! gcloud compute instance-groups managed describe "${MIG_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" >/dev/null 2>&1; then
  gcloud compute instance-groups managed create "${MIG_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --default-action-on-vm-failure=do-nothing \
    --size="${NUM_VMS}" \
    --template="${TEMPLATE_NAME}"
fi

echo "=== 4. Waiting for MIG ${MIG_NAME} to become stable and RUNNING ==="
gcloud compute instance-groups managed wait-until "${MIG_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --stable \
  --timeout=600

echo "=== 5. Listing provisioned A3-Mega instances ==="
gcloud compute instance-groups managed list-instances "${MIG_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}"
