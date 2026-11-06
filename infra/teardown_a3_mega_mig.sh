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
TEMPLATE_NAME="${PREFIX}-tmpl"
POLICY_NAME="${PREFIX}-wp"
MIG_NAME="${PREFIX}-mig"
DELETE_TEMPLATE="${DELETE_TEMPLATE:-0}"
DELETE_NETWORKS="${DELETE_NETWORKS:-0}"

echo "=== 1. Deleting Managed Instance Group ${MIG_NAME} in ${ZONE} ==="
gcloud compute instance-groups managed delete "${MIG_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --quiet || true

if [[ "${DELETE_TEMPLATE}" == "1" ]]; then
  echo "=== 2. Deleting Workload Policy ${POLICY_NAME} (if present) ==="
  gcloud compute resource-policies delete "${POLICY_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --quiet || true

  echo "=== 3. Deleting Instance Template ${TEMPLATE_NAME} ==="
  gcloud compute instance-templates delete "${TEMPLATE_NAME}" \
    --project="${PROJECT_ID}" \
    --quiet || true
else
  echo "=== 2. Keeping Instance Template ${TEMPLATE_NAME} (set DELETE_TEMPLATE=1 to remove) ==="
fi

if [[ "${DELETE_NETWORKS}" == "1" ]]; then
  echo "=== 4. Deleting Firewall Rules, Subnets, and VPC Networks (${PREFIX}-net-1..8) ==="
  for i in $(seq 1 8); do
    gcloud compute firewall-rules delete "${PREFIX}-fw-${i}" --project="${PROJECT_ID}" --quiet || true
    gcloud compute networks subnets delete "${PREFIX}-sub-${i}" --project="${PROJECT_ID}" --region="${REGION}" --quiet || true
    gcloud compute networks delete "${PREFIX}-net-${i}" --project="${PROJECT_ID}" --quiet || true
  done
else
  echo "=== 3. Keeping 8 GPU Data VPC Networks ${PREFIX}-net-1..8 (set DELETE_NETWORKS=1 to remove) ==="
fi

echo "=== 4. Verifying 0 A3 instances remain running ==="
gcloud compute instances list --project="${PROJECT_ID}" --filter="machineType ~ a3-"
