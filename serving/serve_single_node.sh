#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-zai-org/GLM-5.3-Flash}"
STRATEGY="${STRATEGY:-tp8}" # tp8 or tep8
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
PORT="${PORT:-8000}"

sudo docker rm -f vllm-server 2>/dev/null || true
sudo mkdir -p /mnt/localssd/vllm_root/.cache/huggingface

EXTRA_FLAGS=()
if [[ "${STRATEGY}" == "tep8" ]]; then
  EXTRA_FLAGS+=(--enable-expert-parallel --tensor-parallel-size 8)
else
  EXTRA_FLAGS+=(--tensor-parallel-size 8)
fi

echo "=== Starting Single-Node (8x H100) vLLM Server: model=${MODEL}, strategy=${STRATEGY}, max_model_len=${MAX_MODEL_LEN}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} ==="
sudo docker run -d --name vllm-server \
  --gpus all \
  --privileged \
  --ipc=host \
  --net=host \
  -v /mnt/localssd/vllm_root:/root \
  -v /mnt/localssd/hf_cache:/root/.cache/huggingface \
  -v /tmp:/tmp \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_USE_RUST_FRONTEND=1 \
  vllm/vllm-openai:nightly "${MODEL}" \
  --served-model-name "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tokenizer-mode glm5_next \
  "${EXTRA_FLAGS[@]}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --disable-flashinfer-autotune \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm47

echo "Launched container vllm-server. Follow logs with: sudo docker logs -f vllm-server"
