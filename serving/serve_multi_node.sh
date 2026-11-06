#!/usr/bin/env bash
set -euo pipefail

NODE_RANK="${NODE_RANK:?Must set NODE_RANK=0 or 1}"
HEAD_IP="${HEAD_IP:?Must set HEAD_IP to Node 0 internal IP}"
MODEL="${MODEL:-zai-org/GLM-5.3-Flash-BF16}"
STRATEGY="${STRATEGY:-tp16}" # tp16, tep16, or tp8_pp2
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
USE_TCPXO="${USE_TCPXO:-0}"
PORT="${PORT:-8000}"

sudo docker rm -f vllm-server 2>/dev/null || true
sudo mkdir -p /mnt/localssd/vllm_root/.cache/huggingface

MGMT_IFACE=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)
DATA_IFACES=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^enp|^eth' | grep -v "^${MGMT_IFACE}$" | paste -sd "," -)

PARALLEL_FLAGS=()
if [[ "${STRATEGY}" == "tp16" ]]; then
  PARALLEL_FLAGS+=(--tensor-parallel-size 16 --nnodes 2 --node-rank "${NODE_RANK}" --master-addr "${HEAD_IP}")
elif [[ "${STRATEGY}" == "tep16" ]]; then
  PARALLEL_FLAGS+=(--enable-expert-parallel --tensor-parallel-size 16 --nnodes 2 --node-rank "${NODE_RANK}" --master-addr "${HEAD_IP}")
elif [[ "${STRATEGY}" == "tp8_pp2" ]]; then
  PARALLEL_FLAGS+=(--tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank "${NODE_RANK}" --master-addr "${HEAD_IP}")
fi

if [[ "${NODE_RANK}" != "0" ]]; then
  PARALLEL_FLAGS+=(--headless)
fi

TCPXO_ENV=()
if [[ "${USE_TCPXO}" == "1" && -f /var/lib/tcpxo/lib64/libnccl-net.so ]]; then
  echo "Enabling GPUDirect-TCPXO / Fastrak NCCL stack across ${DATA_IFACES}..."
  TCPXO_ENV+=(
    -v /var/lib/tcpxo/lib64:/var/lib/tcpxo/lib64
    -v /run/tcpxo:/run/tcpxo
    -v /tmp:/tmp
    -e NCCL_LIB_DIR=/var/lib/tcpxo/lib64
    -e LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
    -e NCCL_FASTRAK_CTRL_DEV="${MGMT_IFACE}"
    -e NCCL_FASTRAK_IFNAME="${DATA_IFACES}"
    -e NCCL_SOCKET_IFNAME="${MGMT_IFACE}"
    -e GLOO_SOCKET_IFNAME="${MGMT_IFACE}"
    -e NCCL_CROSS_NIC=0
    -e NCCL_PROTO=Simple,LL128
    -e NCCL_MIN_NCHANNELS=4
    -e NCCL_NVLSTREE_MAX_CHUNKSIZE=131072
    -e NCCL_P2P_NET_CHUNKSIZE=524288
    -e NCCL_P2P_PCI_CHUNKSIZE=524288
    -e NCCL_P2P_NVL_CHUNKSIZE=1048576
    -e NCCL_FASTRAK_NUM_FLOWS=2
    -e NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
    -e NCCL_BUFFSIZE=8388608
    -e NCCL_FASTRAK_USE_SNAP=1
    -e NCCL_FASTRAK_USE_LLCM=1
    -e NCCL_NET_GDR_LEVEL=PIX
    -e NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
    -e NCCL_TUNER_PLUGIN=libnccl-tuner.so
    -e NCCL_TUNER_CONFIG_PATH=/var/lib/tcpxo/lib64/a3plus_tuner_config.textproto
    -e NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/var/lib/tcpxo/lib64/a3plus_guest_config.textproto
    -e NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
  )
else
  echo "Using 8-Rail Dedicated Multi-NIC Transport (NCCL_CROSS_NIC=0) across ${DATA_IFACES}..."
  TCPXO_ENV+=(
    -e GLOO_SOCKET_IFNAME="${MGMT_IFACE}"
    -e NCCL_SOCKET_IFNAME="${DATA_IFACES}"
    -e NCCL_CROSS_NIC=0
    -e NCCL_SOCKET_NTHREADS=4
    -e NCCL_NSOCKS_PERTHREAD=4
  )
fi

echo "=== Starting Multi-Node (2x8=16x H100) vLLM Server [Node ${NODE_RANK}]: model=${MODEL}, strategy=${STRATEGY}, max_model_len=${MAX_MODEL_LEN}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} ==="
sudo docker run -d --name vllm-server \
  --gpus all \
  --privileged \
  --ipc=host \
  --net=host \
  --ulimit nofile=1048576:1048576 \
  --ulimit memlock=-1:-1 \
  -v /mnt/localssd/vllm_root:/root \
  -v /mnt/localssd/hf_cache:/root/.cache/huggingface \
  -v /tmp:/tmp \
  "${TCPXO_ENV[@]}" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_USE_RUST_FRONTEND=1 \
  vllm/vllm-openai:nightly "${MODEL}" \
  --served-model-name "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tokenizer-mode glm5_next \
  --no-numa-bind \
  "${PARALLEL_FLAGS[@]}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --disable-flashinfer-autotune \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm47

echo "Launched container vllm-server on Node ${NODE_RANK}."
