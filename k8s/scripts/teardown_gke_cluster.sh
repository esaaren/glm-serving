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
DELETE_NETWORKS="${DELETE_NETWORKS:-0}"

echo "=== 1. Deleting GKE Cluster ${CLUSTER_NAME} in ${ZONE} ==="
gcloud container clusters delete "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --quiet || true

if [[ "${DELETE_NETWORKS}" == "1" ]]; then
  echo "=== 2. Deleting 8 GPUDirect-TCPXO VPC Networks (${PREFIX}-net-1..8) ==="
  for i in $(seq 1 8); do
    gcloud compute firewall-rules delete "${PREFIX}-fw-${i}" --project="${PROJECT_ID}" --quiet || true
    gcloud compute networks subnets delete "${PREFIX}-sub-${i}" --project="${PROJECT_ID}" --region="${REGION}" --quiet || true
    gcloud compute networks delete "${PREFIX}-net-${i}" --project="${PROJECT_ID}" --quiet || true
  done
else
  echo "=== 2. Keeping 8 GPUDirect-TCPXO VPC Networks ${PREFIX}-net-1..8 (set DELETE_NETWORKS=1 to remove) ==="
fi

echo "=== 3. Verifying 0 A3 instances remain ==="
gcloud compute instances list --project="${PROJECT_ID}" --filter="machineType ~ a3-"
