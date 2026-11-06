#!/usr/bin/env bash
set -euo pipefail

RELEASE_NAME="${RELEASE_NAME:-glm-vllm}"
NAMESPACE="${NAMESPACE:-default}"
MODE="${MODE:-all}" # single-fp8, multi-bf16, multi-fp8, or all
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="${SCRIPT_DIR}/../helm/glm-vllm-benchmark"
RESULTS_DIR="${SCRIPT_DIR}/../../results"

mkdir -p "${RESULTS_DIR}"

run_helm_benchmark() {
  local values_file="$1"
  local label="$2"

  echo "================================================================================"
  echo "Deploying Helm Release '${RELEASE_NAME}' with ${values_file} (${label})"
  echo "================================================================================"

  helm uninstall "${RELEASE_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1 || true
  kubectl delete job "${RELEASE_NAME}-bench" -n "${NAMESPACE}" --ignore-not-found >/dev/null 2>&1 || true
  kubectl delete statefulset "${RELEASE_NAME}-worker" -n "${NAMESPACE}" --ignore-not-found >/dev/null 2>&1 || true
  sleep 5

  helm install "${RELEASE_NAME}" "${CHART_DIR}" \
    -n "${NAMESPACE}" \
    -f "${CHART_DIR}/${values_file}"

  echo "Waiting for Head Benchmark Pod to enter Running state..."
  for i in $(seq 1 180); do
    PHASE=$(kubectl get pods -n "${NAMESPACE}" -l app=${RELEASE_NAME},role=head -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)
    if [[ "${PHASE}" == "Running" || "${PHASE}" == "Succeeded" ]]; then
      break
    fi
    sleep 5
  done

  HEAD_POD=$(kubectl get pods -n "${NAMESPACE}" -l app=${RELEASE_NAME},role=head -o jsonpath='{.items[0].metadata.name}')
  echo "Streaming logs from Head Pod ${HEAD_POD}..."
  kubectl logs -f "${HEAD_POD}" -n "${NAMESPACE}" -c vllm-head-and-bench || true

  echo "Waiting for Job ${RELEASE_NAME}-bench to complete..."
  kubectl wait --for=condition=complete "job/${RELEASE_NAME}-bench" -n "${NAMESPACE}" --timeout=3600s

  echo "Extracting JSON benchmark outputs from ${HEAD_POD} logs into ${RESULTS_DIR}..."
  LOG_FILE=$(mktemp)
  kubectl logs "${HEAD_POD}" -n "${NAMESPACE}" -c vllm-head-and-bench > "${LOG_FILE}"

  python3 - "${LOG_FILE}" "${RESULTS_DIR}" <<'PYEOF'
import re
import sys
from pathlib import Path

log_text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
out_dir = Path(sys.argv[2])

nccl_match = re.search(r"NCCL_TCPXO_JSON_RESULT_START\s*(\{.*?\})\s*NCCL_TCPXO_JSON_RESULT_END", log_text, re.DOTALL)
if nccl_match:
    p = out_dir / "gke_nccl_tcpxo_allreduce.json"
    p.write_text(nccl_match.group(1).strip() + "\n", encoding="utf-8")
    print(f"Saved {p}")

for m in re.finditer(r"BENCHMARK_JSON_RESULT_START:([^\s]+)\s*(\{.*?\})\s*BENCHMARK_JSON_RESULT_END:\1", log_text, re.DOTALL):
    fname, content = m.group(1), m.group(2)
    p = out_dir / fname
    p.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Saved {p}")
PYEOF
  rm -f "${LOG_FILE}"

  helm uninstall "${RELEASE_NAME}" -n "${NAMESPACE}" || true
}

if [[ "${MODE}" == "single-fp8" || "${MODE}" == "all" ]]; then
  run_helm_benchmark "values-single-node-fp8.yaml" "Single-Node 8xH100 TP=8 FP8"
fi

if [[ "${MODE}" == "multi-bf16" || "${MODE}" == "all" ]]; then
  run_helm_benchmark "values-multi-node-bf16.yaml" "Multi-Node 16xH100 TP=16 BF16 (GPUDirect-TCPXO)"
fi

if [[ "${MODE}" == "multi-fp8" ]]; then
  run_helm_benchmark "values-multi-node-fp8.yaml" "Multi-Node 16xH100 TP=16 FP8 (GPUDirect-TCPXO)"
fi
