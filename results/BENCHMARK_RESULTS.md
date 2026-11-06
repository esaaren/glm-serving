# GLM-5.3-Flash vLLM Serving Benchmark Report: Single-Node (8× H100) vs. Multi-Host (2-Node / 16× H100) on GCP A3-Mega

**Zone:** `us-west1-b`  
**Machine Type:** `a3-megagpu-8g` (Spot Managed Instance Group `glm-a3mega-mig` & GKE Node Pool `a3-mega-spot-pool`, 2 Hosts = 16× NVIDIA H100 80GB HBM3 GPUs, 9× gVNICs per host)  
**Serving Engine:** `vllm/vllm-openai:nightly` (vLLM `0.19.1rc1` with DeepSeek-V3.2 / GlmMoeDsa Sparse Multi-head Latent Attention `FLASHINFER_MLA` & DeepGEMM)  
**Models Evaluated:**
- [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) — Native FP8 (`328.4 GB` disk / `312.16 GiB` GPU weights, `321,323,031,390` Total Parameters [~321.3B MoE] with Sparse Indexer MLA)
- [`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) — Unquantized 16-bit BF16 (`642.7 GB` disk / `609.12 GiB` GPU weights, `321,323,031,390` Total Parameters [~321.3B MoE])

---

## 1. Summary & Architectural Conclusions

1. **Why 16-Bit (`BF16`) & Long-Context Workloads Exceed a Single 8×80GB H100 Node (`640 GB` HBM):**
   - `GLM-5.3-Flash` is a **321.3-billion parameter Mixture-of-Experts (MoE)** architecture (`321,323,031,390` parameters, `~30B` active parameters per token, `kv_lora_rank=512`, `index_head_dim=128`).
   - In 16-bit precision (`zai-org/GLM-5.3-Flash-BF16`), model weights alone consume **`74.65 GiB` per GPU (`597.20 GiB` across 8 GPUs)** on a single `a3-megagpu-8g` host. During vLLM's warmup pass (`determine_available_memory()`), forward activation buffers exceed the remaining `~4.9 GiB/GPU`, triggering an immediate **`Cuda failure 2 'out of memory'`** with **zero bytes available for KV cache**.
   - Scaling across **2 Hosts / 16× H100 GPUs (`1,280 GB` aggregate HBM, `TP=16`)** cuts the BF16 weight footprint in half to **`38.07 GiB` per GPU (`609.12 GiB` total)**, freeing **`~29.05 GiB` per GPU (`464.8 GiB` total)** for KV cache (**`2,470,799` KV cache tokens**, supporting **`18.85×` concurrent `131,072`-token requests** in full 16-bit precision).
   - Running **FP8 (`zai-org/GLM-5.3-Flash`) across 2 Hosts (`TP=16`)** reduces model weights to **`20.27 GiB` per GPU (`324.32 GiB` total)**, expanding the cluster KV cache pool to **`45.84 GiB` per GPU (`733.44 GiB` cluster KV cache = `4,043,663` KV cache tokens, `30.85×` max concurrency at `131,072` context length)**.

2. **Throughput Across Input/Output Token Shapes (`4K`–`120K` Input / `1K` Output):**
   - **Single-Node `8× H100` (`TP=8`, FP8)** achieves:
     - **`29,298.16 total tok/s`** (`23,303.72 input tok/s` + **`5,994.44 output decode tok/s`**, **`5.854 req/s`**) on **4K-input / 1K-output** requests at concurrency `c=256`.
     - **`40,893.74 total tok/s`** (**`40,488.01 input prefill tok/s`** + `405.72 output decode tok/s`) on **120K-input / 1K-output** requests at concurrency `c=8`.
   - **Multi-Host `2-Node / 16× H100` (`TP=16`, GPUDirect-TCPXO FP8 & 8-Rail BF16)** achieves:
     - **`23,282.36 total tok/s`** (`18,526.31 input tok/s` + **`4,756.05 output decode tok/s`**, **`4.654 req/s`**) on **4K-input / 1K-output** requests at `c=256` with **GKE GPUDirect-TCPXO** (`NET/FastSocket`), and **`19,707.03 total tok/s`** on **120K-input / 1K-output** requests at `c=8`.
     - **`13,266.58 total tok/s`** (`10,552.22 input tok/s` + **`2,714.36 output decode tok/s`**, **`2.651 req/s`**) on **4K-input / 1K-output** requests at `c=256` in unquantized **16-bit BF16** over 8-rail TCP, and **`12,321.14 total tok/s`** at `120K` input (`c=8`).

![HBM Memory and KV Cache Breakdown](charts/hbm_memory_and_kv_cache_breakdown.png)

---

## 2. HBM Memory Breakdown & KV Cache Capacity Comparison

The table below summarizes the empirically measured per-GPU and cluster-wide HBM allocations across Single-Node (`8× H100 = 640 GB`) and Multi-Host (`16× H100 = 1,280 GB`) configurations (`--gpu-memory-utilization 0.90`, `--max-model-len 131072`, `--max-num-batched-tokens 8192`):

| Serving Configuration | Model Checkpoint | GPUs (Nodes) | Total HBM | Model Weights / GPU (Total) | CUDA Graphs / GPU | Available KV Cache / GPU (Total) | Allocated KV Cache Tokens | Max Concurrency @ 131K Context | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1A: Single-Node BF16 (`TP=8`)** | `zai-org/GLM-5.3-Flash-BF16` | `8× H100` (1 Node) | `640 GB` | `74.65 GiB` (`597.20 GiB`) | — (OOM before graph capture) | **`0.00 GiB`** (OOM in warmup) | **`0`** | **`0.00×`** | **OOM (`Cuda failure 2`)** |
| **Config 1B: Single-Node FP8 (`TP=8`)** | `zai-org/GLM-5.3-Flash` | `8× H100` (1 Node) | `640 GB` | `39.02 GiB` (`312.16 GiB`) | `1.54 GiB` (`40 s`) | **`30.58 GiB`** (`244.64 GiB`) | **`2,434,451`** | **`18.57×`** | **Healthy (`75.1 GiB/GPU`)** |
| **Config 2A: Multi-Host BF16 (`TP=16`)** | `zai-org/GLM-5.3-Flash-BF16` | `16× H100` (2 Nodes) | `1,280 GB` | `38.07 GiB` (`609.12 GiB`) | `3.07 GiB` (`31 s`) | **`29.05 GiB`** (`464.80 GiB`) | **`2,470,799`** | **`18.85×`** | **Healthy (`74.3 GiB/GPU`)** |
| **Config 2B: Multi-Host FP8 (`TP=16`)** | `zai-org/GLM-5.3-Flash` | `16× H100` (2 Nodes) | `1,280 GB` | `20.27 GiB` (`324.32 GiB`) | `3.07 GiB` (`31 s`) | **`45.84 GiB`** (`733.44 GiB`) | **`4,043,663`** | **`30.85×`** | **Healthy (`4.04M` KV Tokens)** |

---

## 3. GCP A3-Mega Infrastructure & 8-NIC Multi-Rail Collective Bandwidth

Each `a3-megagpu-8g` instance provides **9 gVNIC network interfaces**: `nic0` (`enp0s12`, host management VPC) plus **8 dedicated GPU data NICs** (`nic1..nic8`: `enp6s0f0, enp7s0f0, enp13s0f0, enp14s0f0, enp134s0f0, enp135s0f0, enp141s0f0, enp142s0f0`) mapped 1:1 to the PCIe switches of the 8 H100 GPUs across 8 isolated VPC networks (`glm-a3mega-net-1..8`, MTU `8244`).

### 2-Node (16× H100) Inter-Node NCCL AllReduce Benchmark Results

Using 8-rail dedicated GPU-NIC transport (`NCCL_SOCKET_IFNAME=enp6s0f0,enp7s0f0,enp13s0f0,enp14s0f0,enp134s0f0,enp135s0f0,enp141s0f0,enp142s0f0`, `NCCL_CROSS_NIC=0`, `NCCL_NSOCKS_PERTHREAD=4`, `NCCL_SOCKET_NTHREADS=4`) across all 16 H100 GPUs (`glm-a3mega-mig-tw4q` + `glm-a3mega-mig-x5v7`):

| AllReduce Tensor Size | Iteration Latency (`ms`) | Algorithmic Bandwidth (`GB/s`) | NCCL Bus Bandwidth (`GB/s`) | Effective Inter-Node Throughput (`Gbps`) |
| :---: | :---: | :---: | :---: | :---: |
| **`1 MB`** | `0.36 ms` | `2.71 GB/s` | `5.09 GB/s` | `40.7 Gbps` |
| **`16 MB`** | `1.38 ms` | `11.29 GB/s` | `21.17 GB/s` | `169.4 Gbps` |
| **`128 MB`** | `6.02 ms` | `20.75 GB/s` | `38.91 GB/s` | `311.3 Gbps` |
| **`512 MB`** | `21.41 ms` | `23.35 GB/s` | **`43.79 GB/s`** | **`350.3 Gbps`** |

![NCCL Multi-Rail Bandwidth](charts/nccl_multirail_roce_bandwidth.png)

---

## 4. Benchmark Results Across Input Shapes (`4K` to `120K` Tokens)

### 4.1 Complete Benchmark Summary Table (`results/summary_metrics.csv`)

All runs use a procedurally generated synthetic dataset (`benchmarks/generate_legal_dataset.py`) with `reasoning_effort="low"` and `target_output_tokens=1024` across `4K`, `16K`, `64K`, and `120K` input token shapes (`0%` error rate across all runs):

| Configuration | Model | Input Shape | Target Input (`Avg Prompt`) | Target Output | Concurrency (`Prompts`) | Wall Time (`s`) | Req/s | Input Prefill (`tok/s`) | Output Decode (`tok/s`) | **Total Throughput (`tok/s`)** | TTFT Mean / P50 / P99 (`s`) | E2E Mean / P99 (`s`) | Result JSON |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1-Node `8×H100` (`TP=8`, FP8)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=32` (`32`) | `18.35s` | `1.744` | `6,943.31` | `1,785.91` | **`8,729.22`** | `3.73s` / `3.67s` / `5.43s` | `18.27s` / `18.34s` | [single_node_tp8_fp8_c32_4k.json](single_node_tp8_fp8_c32_4k.json) |
| **1-Node `8×H100` (`TP=8`, FP8)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=64` (`64`) | `21.78s` | `2.938` | `11,697.51` | `3,008.57` | **`14,706.09`** | `1.94s` / `1.95s` / `3.71s` | `21.68s` / `21.78s` | [single_node_tp8_fp8_c64_4k.json](single_node_tp8_fp8_c64_4k.json) |
| **1-Node `8×H100` (`TP=8`, FP8)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=128` (`128`) | `28.68s` | `4.463` | `17,765.31` | `4,569.75` | **`22,335.06`** | `4.04s` / `4.05s` / `7.43s` | `28.45s` / `28.67s` | [single_node_tp8_fp8_c128_4k.json](single_node_tp8_fp8_c128_4k.json) |
| **1-Node `8×H100` (`TP=8`, FP8)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=256` (`256`) | `43.73s` | **`5.854`** | `23,303.72` | **`5,994.44`** | **`29,298.16`** | `7.73s` / `7.73s` / `15.79s` | `43.09s` / `43.71s` | [single_node_tp8_fp8_c256_4k.json](single_node_tp8_fp8_c256_4k.json) |
| **1-Node `8×H100` (`TP=8`, FP8)** | `GLM-5.3-Flash` | **16K In / 1K Out** | `16,384` (`14,423`) | `1,024` | `c=64` (`64`) | `38.48s` | `1.663` | `23,989.44` | `1,703.24` | **`25,692.68`** | `11.06s` / `11.31s` / `21.60s` | `37.92s` / `38.46s` | [single_node_tp8_fp8_c64_16k.json](single_node_tp8_fp8_c64_16k.json) |
| **1-Node `8×H100` (`TP=8`, FP8)** | `GLM-5.3-Flash` | **64K In / 1K Out** | `65,536` (`56,085`) | `1,024` | `c=16` (`16`) | `30.23s` | `0.529` | `29,682.58` | `541.94` | **`30,224.53`** | `10.60s` / `11.17s` / `19.75s` | `29.90s` / `30.22s` | [single_node_tp8_fp8_c16_64k.json](single_node_tp8_fp8_c16_64k.json) |
| **1-Node `8×H100` (`TP=8`, FP8)** | `GLM-5.3-Flash` | **120K In / 1K Out** | `120,000` (`102,187`) | `1,024` | `c=8` (`8`) | `20.19s` | `0.396` | **`40,488.01`** | `405.72` | **`40,893.74`** | `6.51s` / `7.22s` / `11.51s` | `20.04s` / `20.19s` | [single_node_tp8_fp8_c8_120k.json](single_node_tp8_fp8_c8_120k.json) |
| **2-Node `16×H100` (`TP=16`, BF16)** | `GLM-5.3-Flash-BF16` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=32` (`32`) | `89.98s` | `0.356` | `1,415.91` | `364.19` | **`1,780.10`** | `4.07s` / `4.33s` / `7.34s` | `89.63s` / `89.97s` | [multi_node_tp16_bf16_c32_4k.json](multi_node_tp16_bf16_c32_4k.json) |
| **2-Node `16×H100` (`TP=16`, BF16)** | `GLM-5.3-Flash-BF16` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=64` (`64`) | `98.25s` | `0.651` | `2,593.42` | `667.02` | **`3,260.44`** | `3.64s` / `3.82s` / `7.22s` | `97.81s` / `98.23s` | [multi_node_tp16_bf16_c64_4k.json](multi_node_tp16_bf16_c64_4k.json) |
| **2-Node `16×H100` (`TP=16`, BF16)** | `GLM-5.3-Flash-BF16` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=128` (`128`) | `70.65s` | `1.812` | `7,212.79` | `1,855.34` | **`9,068.13`** | `8.09s` / `8.13s` / `14.86s` | `69.63s` / `70.64s` | [multi_node_tp16_bf16_c128_4k.json](multi_node_tp16_bf16_c128_4k.json) |
| **2-Node `16×H100` (`TP=16`, BF16)** | `GLM-5.3-Flash-BF16` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=256` (`256`) | `96.58s` | **`2.651`** | `10,552.22` | **`2,714.36`** | **`13,266.58`** | `14.94s` / `14.87s` / `28.84s` | `94.46s` / `96.55s` | [multi_node_tp16_bf16_c256_4k.json](multi_node_tp16_bf16_c256_4k.json) |
| **2-Node `16×H100` (`TP=16`, BF16)** | `GLM-5.3-Flash-BF16` | **16K In / 1K Out** | `16,384` (`14,423`) | `1,024` | `c=64` (`64`) | `133.42s` | `0.480` | `6,918.23` | `491.19` | **`7,409.42`** | `21.89s` / `22.05s` / `42.25s` | `130.64s` / `133.40s` | [multi_node_tp16_bf16_c64_16k.json](multi_node_tp16_bf16_c64_16k.json) |
| **2-Node `16×H100` (`TP=16`, BF16)** | `GLM-5.3-Flash-BF16` | **64K In / 1K Out** | `65,536` (`56,085`) | `1,024` | `c=16` (`16`) | `108.80s` | `0.147` | `8,248.20` | `150.60` | **`8,398.80`** | `21.75s` / `22.81s` / `41.16s` | `107.52s` / `108.79s` | [multi_node_tp16_bf16_c16_64k.json](multi_node_tp16_bf16_c16_64k.json) |
| **2-Node `16×H100` (`TP=16`, BF16)** | `GLM-5.3-Flash-BF16` | **120K In / 1K Out** | `120,000` (`102,187`) | `1,024` | `c=8` (`8`) | `67.01s` | `0.119` | **`12,198.90`** | `122.24` | **`12,321.14`** | `23.46s` / `25.93s` / `42.22s` | `66.33s` / `67.01s` | [multi_node_tp16_bf16_c8_120k.json](multi_node_tp16_bf16_c8_120k.json) |
| **2-Node `16×H100` (`TP=16`, FP8)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=32` (`32`) | `86.03s` | `0.372` | `1,480.89` | `380.90` | **`1,861.79`** | `3.98s` / `4.25s` / `7.23s` | `85.69s` / `86.02s` | [multi_node_tp16_fp8_c32_4k.json](multi_node_tp16_fp8_c32_4k.json) |
| **2-Node `16×H100` (`TP=16`, FP8)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=64` (`64`) | `95.10s` | `0.673` | `2,679.43` | `689.14` | **`3,368.58`** | `4.31s` / `4.67s` / `8.02s` | `94.62s` / `95.09s` | [multi_node_tp16_fp8_c64_4k.json](multi_node_tp16_fp8_c64_4k.json) |
| **2-Node `16×H100` (`TP=16`, FP8)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=128` (`128`) | `68.29s` | `1.874` | `7,462.10` | `1,919.47` | **`9,381.57`** | `8.77s` / `8.57s` / `16.12s` | `67.32s` / `68.28s` | [multi_node_tp16_fp8_c128_4k.json](multi_node_tp16_fp8_c128_4k.json) |
| **2-Node `16×H100` (`TP=16`, FP8)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=256` (`256`) | `91.07s` | **`2.811`** | `11,190.08` | **`2,878.43`** | **`14,068.51`** | `16.45s` / `16.70s` / `30.54s` | `89.11s` / `91.04s` | [multi_node_tp16_fp8_c256_4k.json](multi_node_tp16_fp8_c256_4k.json) |
| **2-Node `16×H100` (`TP=16`, FP8)** | `GLM-5.3-Flash` | **120K In / 1K Out** | `120,000` (`102,187`) | `1,024` | `c=8` (`8`) | `100.40s` | `0.080` | `8,142.34` | `81.59` | **`8,223.94`** | `41.15s` / `45.25s` / `76.95s` | `99.55s` / `100.39s` | [multi_node_tp16_fp8_c8_120k.json](multi_node_tp16_fp8_c8_120k.json) |
| **GKE 1-Node `8×H100` (`TP=8`, FP8)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=64` (`64`) | `23.64s` | `2.708` | `10,780.60` | `2,772.74` | **`13,553.34`** | `3.76s` / `3.77s` / `7.11s` | `23.43s` / `23.62s` | [gke_single_node_8xH100_tp8_fp8_c64_4k.json](gke_single_node_8xH100_tp8_fp8_c64_4k.json) |
| **GKE 1-Node `8×H100` (`TP=8`, FP8)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=256` (`256`) | `52.24s` | **`4.900`** | `19,506.95` | **`5,017.79`** | **`24,524.75`** | `11.40s` / `11.53s` / `22.41s` | `51.30s` / `52.21s` | [gke_single_node_8xH100_tp8_fp8_c256_4k.json](gke_single_node_8xH100_tp8_fp8_c256_4k.json) |
| **GKE 1-Node `8×H100` (`TP=8`, FP8)** | `GLM-5.3-Flash` | **120K In / 1K Out** | `120,000` (`102,187`) | `1,024` | `c=8` (`8`) | `32.68s` | `0.245` | **`25,014.45`** | `250.67` | **`25,265.12`** | `13.76s` / `15.48s` / `24.09s` | `32.37s` / `32.68s` | [gke_single_node_8xH100_tp8_fp8_c8_120k.json](gke_single_node_8xH100_tp8_fp8_c8_120k.json) |
| **GKE 2-Node `16×H100` (`TP=16`, FP8, TCPXO)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=64` (`64`) | `32.24s` | `1.985` | `7,903.79` | `2,016.89` | **`9,920.68`** | `6.13s` / `6.24s` / `9.36s` | `31.78s` / `32.22s` | [gke_multi_node_16xH100_tp16_fp8_tcpxo_c64_4k.json](gke_multi_node_16xH100_tp16_fp8_tcpxo_c64_4k.json) |
| **GKE 2-Node `16×H100` (`TP=16`, FP8, TCPXO)** | `GLM-5.3-Flash` | **4K In / 1K Out** | `4,096` (`3,981`) | `1,024` | `c=256` (`256`) | `55.01s` | **`4.654`** | `18,526.31` | **`4,756.05`** | **`23,282.36`** | `11.32s` / `11.61s` / `21.16s` | `53.87s` / `54.96s` | [gke_multi_node_16xH100_tp16_fp8_tcpxo_c256_4k.json](gke_multi_node_16xH100_tp16_fp8_tcpxo_c256_4k.json) |
| **GKE 2-Node `16×H100` (`TP=16`, FP8, TCPXO)** | `GLM-5.3-Flash` | **120K In / 1K Out** | `120,000` (`102,187`) | `1,024` | `c=8` (`8`) | `41.90s` | `0.191` | **`19,511.51`** | `195.52` | **`19,707.03`** | `14.27s` / `15.69s` / `25.16s` | `41.25s` / `41.89s` | [gke_multi_node_16xH100_tp16_fp8_tcpxo_c8_120k.json](gke_multi_node_16xH100_tp16_fp8_tcpxo_c8_120k.json) |

---

### 4.2 Concurrency Scaling on 4K Input / 1K Output Legal Redlining (`c=32` to `c=256`)

![4K Legal Concurrency Scaling](charts/throughput_scaling_4k_legal.png)

**Analysis:**
- **Single-Node `8×H100` (`TP=8`, FP8)** scales near-linearly from `8,729.22 tok/s` (`c=32`) to **`29,298.16 tok/s`** (`c=256`), where output decode throughput reaches **`5,994.44 tok/s`** (`5.854` completed 1,024-token legal memoranda per second = **`21,074` legal contract reviews per hour** on a single 8-GPU node).
- **2-Node `16×H100` (`TP=16`, BF16)** exhibits strong batch-size scaling: at lower concurrency (`c=32`), 16-way inter-node AllReduce latency across 78 Transformer layers dominates (`1,780.10 tok/s`). As batch concurrency increases to `c=128` (`9,068.13 tok/s`) and `c=256` (**`13,266.58 tok/s`**, `2,714.36 tok/s` output decode), larger per-step token batches amortize the inter-node collective overhead across the 8-rail 350 Gbps network, achieving **7.45× higher throughput at `c=256` than at `c=32`**.

---

### 4.3 Long-Context Legal Document Scaling (`4K`, `16K`, `64K`, `120K` Input Tokens)

![Long Context Throughput Scaling](charts/long_context_throughput_scaling.png)

**Analysis:**
- Thanks to `GLM-5.3-Flash`'s **DeepSeek-V3.2 Sparse Multi-head Latent Attention (`GlmMoeDsa` with `index_topk=2048` and `kv_lora_rank=512`)**, prefill throughput actually **increases** with document length as FlashInfer MLA saturates H100 Tensor Cores:
  - On Single-Node FP8 (`TP=8`), prefill throughput climbs from `23,303.72 tok/s` (`4K`) $\rightarrow$ `23,989.44 tok/s` (`16K`) $\rightarrow$ `29,682.58 tok/s` (`64K`) $\rightarrow$ **`40,488.01 tok/s` (`120K`)**.
  - On 2-Node BF16 (`TP=16`), prefill throughput climbs from `6,918.23 tok/s` (`16K`) $\rightarrow$ `8,248.20 tok/s` (`64K`) $\rightarrow$ **`12,198.90 tok/s` (`120K`)**.

---

### 4.4 Batch Turnaround Latency Profile (TTFT & End-to-End Memo Generation)

![TTFT and E2E Latency Profile](charts/ttft_and_e2e_latency_profile.png)

---

## 5. Synthetic Benchmark Workload Design

### 5.1 Prompt Generator & Input/Output Token Shapes

Generated by [generate_legal_dataset.py](../benchmarks/generate_legal_dataset.py) and executed by [legal_benchmark.py](../benchmarks/legal_benchmark.py). All documents, entity names, and exhibits are procedurally generated synthetic text designed to produce deterministic input token lengths (`4K`, `16K`, `64K`, `120K` tokens) and `1,024`-token structured completions. The serving configurations and throughput measurements depend only on these input/output token shapes and apply generally to any long-context workload.

- **System Prompt**: Directs the model to act as *Principal Legal Counsel and Head of M&A Due Diligence & Complex Commercial Litigation* at a Tier-1 international law firm reviewing enterprise agreements, M&A data room exhibits, and SEC/GDPR/BIPA regulatory filings.
- **Input Prompts** (`datasets/legal_prompts_{4k,16k,64k,120k}.jsonl`): Realistic multi-exhibit deal packets (*Project NeuralLex — Deal File #MNA-2026-6000*) containing:
  1. *Agreement and Plan of Merger — Indemnification, Survival & Escrow Provisions* (uncapped carve-out traps, `0.35%` de minimis tipping basket, non-reliance clauses).
  2. *Enterprise SaaS MSA & Data Processing Addendum (DPA)* (unnotified 310,000–850,000 record GDPR Art. 33 / Illinois BIPA breach incidents, cross-border SCC gaps).
  3. *GenAI IP Licensing & Training Data Schedule* (copyleft AGPLv3 contamination in core weights, uncapped customer output IP indemnity).
  4. *Senior Secured Credit Facility & Change-of-Control Covenants* (`3.50x` Net Leverage breach at `4.15x`, automatic Event of Default upon Change of Control).
  5. *Active Litigation Docket & Subpoena Disclosures* (Securities Class Action, DOJ Civil Investigative Demand, Trade Secret TRO, and Top-10 Enterprise Customer Change-of-Control Termination Matrix).
- **Output Prompt Specification (`1,024` tokens)**: Requires a 5-section structured legal memorandum including an Executive Risk Matrix, Clause-by-Clause Statutory Analysis, Exact Redline Contract Modifications (`[DELETED: ...]` / `<<INSERTED: ...>>`), Litigation Exposure & Escrow Sizing Table, and Senior Partner Conditions Precedent Checklist.

### 5.2 Verbatim Sample Model Output (`120K`-Token 231-Exhibit Data Room Review)

Excerpt from [multi_node_tp16_bf16_c8_120k.json](multi_node_tp16_bf16_c8_120k.json) (`102,187` prompt tokens $\rightarrow$ `1,024` completion tokens):

```markdown
# LEGAL REVIEW & REDLINE MEMORANDUM

**MATTER:** Project NeuralLex — Acquisition of NeuralLex Enterprise Systems, Inc. ("Target") by Acquirer Corp. ("Buyer")
**DEAL FILE:** #MNA-2026-6000 | **PRIVILEGED & CONFIDENTIAL — ATTORNEY WORK PRODUCT**
**PREPARED BY:** Office of the Principal Legal Counsel, Head of M&A Due Diligence & Complex Commercial Litigation

---

## SECTION 1 — EXECUTIVE RISK SUMMARY & MATERIALITY MATRIX

The data room (Exhibits 001–231, repetitive template exhibits) reveals a Target with severe, systemic exposures across indemnification architecture, data privacy, IP ownership, credit facility cross-defaults, and customer contract change-of-control triggers. Aggregate quantifiable exposure (litigation + breach remediation + contract clawbacks) materially exceeds the aggregate indemnity escrow capacity.

| # | Risk | Severity | Cite | Quantified Exposure |
|---|------|----------|------|---------------------|
| 1 | **Undisclosed GDPR Art. 33 breaches** — 40+ separate security incidents (e.g., Incident #1386, Ex. 002 §2.01(b); #9781, Ex. 008; #9754, Ex. 086 — 850,000 unencrypted records); no notification to data subjects, AGs, or supervisory authorities within 72 hours | **CRITICAL** | Ex. 002, 008, 014, 020, 026, 032, 038, 044, 050, 056, 062, 068, 074, 080 | Up to 4% global annual turnover (GDPR Art. 83(5)) + uncapped DPA liability carve-outs |
| 2 | **Unlimited DPA liability carve-out** — Vendor liability for data/BIPA/GDPR breach uncapped under Customer DPA #MSA-2022-202 (and #MSA-2026-924) | **CRITICAL** | Ex. 002 §2.01(c); Ex. 001 §1.02(b) | Buyer inherits uncapped successor exposure excluded from Seller indemnity cap |
```

---

## 6. Key Systems & vLLM Engineering Findings

1. **FlashInfer SM90 Sparse MLA Work Item Limit (`_FA_MAX_WORK_ITEMS = 16384`)**:
   - In `vllm/vllm-openai:nightly`, `GLM-5.3-Flash` uses `FLASHINFER_MLA` (`FlashInferMLASparseSM90`). FlashInfer's SM90 MLA backend enforces `_FA_MAX_WORK_ITEMS = 16384` in `flashinfer/mla/_batch_mla/_backends/_fa_common.py`. Passing `--max-num-batched-tokens 32768` causes `compile_or_warm_up_model()` to fail with `FA MLA needs at least 32768 work items; native capacity is 16384`. Setting `--max-num-batched-tokens 8192` keeps prefill batches comfortably within FlashInfer SM90's native capacity while saturating H100 Tensor Cores (`>40,000 tok/s` prefill).
2. **Multi-Node `--headless` NUMA Subprocess Race Fix (`--no-numa-bind`)**:
   - In `vllm/vllm-openai:nightly`, worker nodes launched with `--headless` spawn 8 local `MultiprocExecutor` workers concurrently via a `ThreadPoolExecutor` (`_start_workers_concurrently`). Without `--no-numa-bind`, `numa_utils.configure_subprocess` mutates the process-global `multiprocessing.spawn.set_executable` across concurrent threads, causing `AssertionError: Executable was changed during NUMA binding context`. Passing `--no-numa-bind` eliminates this race condition.
3. **Docker File Descriptor Limits (`--ulimit nofile=1048576:1048576`) for 8-Rail NCCL**:
   - Across 16 GPUs and 8 data NICs (`NCCL_NSOCKS_PERTHREAD=4`), NCCL opens thousands of concurrent sockets during `ncclCommInitRank`. Increasing container `--ulimit nofile=1048576:1048576` and `--ulimit memlock=-1:-1` prevents `ncclSystemError: Call to socket failed: Too many open files`.
4. **Persistent Local NVMe JIT Kernel Cache (`6 min` $\rightarrow$ `40 sec` Startup)**:
   - Pre-extracting `/root/.cache/flashinfer`, `/root/.cache/vllm`, `/root/.tilelang`, `/root/.triton`, and `/tmp/torchinductor_root` onto the 5.9 TB 16×NVMe RAID-0 array (`/mnt/localssd/vllm_root`) eliminates the 6-minute DeepGEMM / TileLang / FlashInfer JIT compilation on subsequent container restarts.
