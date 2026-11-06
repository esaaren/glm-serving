# GCP A3-Mega (`a3-megagpu-8g`) Multi-NIC Networking & Spot MIG Setup Guide

This document details the network architecture, Managed Instance Group (MIG) provisioning, host storage initialization, and inter-node NCCL configuration required to run multi-host vLLM inference across `a3-megagpu-8g` instances on Google Cloud.

---

## 1. A3-Mega 9-NIC Hardware & VPC Topology

Each `a3-megagpu-8g` VM attaches **8× NVIDIA H100 80GB HBM3 GPUs** interconnected within the host via **NVSwitch (`900 GB/s` bidirectional NVLink)** and across hosts via **9× Google Virtual NIC (`gVNIC`) interfaces**:

| Guest OS Interface | GCP NIC Index | VPC Network | Subnet CIDR | MTU | Role & PCIe / NUMA Alignment |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `enp0s12` | `nic0` | `default` | `10.138.0.0/20` | `1460` | Host management, SSH, Hugging Face egress, Gloo control plane (`GLOO_SOCKET_IFNAME`), & vLLM HTTP API (`:8000`) |
| `enp6s0f0` | `nic1` | `glm-a3mega-net-1` | `192.168.1.0/24` | `8244` | Dedicated GPU 0 Data Rail (NUMA 0) + `torch.distributed` `MASTER_ADDR` (`192.168.1.x`) |
| `enp7s0f0` | `nic2` | `glm-a3mega-net-2` | `192.168.2.0/24` | `8244` | Dedicated GPU 1 Data Rail (NUMA 0) |
| `enp13s0f0` | `nic3` | `glm-a3mega-net-3` | `192.168.3.0/24` | `8244` | Dedicated GPU 2 Data Rail (NUMA 0) |
| `enp14s0f0` | `nic4` | `glm-a3mega-net-4` | `192.168.4.0/24` | `8244` | Dedicated GPU 3 Data Rail (NUMA 0) |
| `enp134s0f0` | `nic5` | `glm-a3mega-net-5` | `192.168.5.0/24` | `8244` | Dedicated GPU 4 Data Rail (NUMA 1) |
| `enp135s0f0` | `nic6` | `glm-a3mega-net-6` | `192.168.6.0/24` | `8244` | Dedicated GPU 5 Data Rail (NUMA 1) |
| `enp141s0f0` | `nic7` | `glm-a3mega-net-7` | `192.168.7.0/24` | `8244` | Dedicated GPU 6 Data Rail (NUMA 1) |
| `enp142s0f0` | `nic8` | `glm-a3mega-net-8` | `192.168.8.0/24` | `8244` | Dedicated GPU 7 Data Rail (NUMA 1) |

### Key GCP Multi-NIC Rules

1. **One VPC Network per NIC**: Compute Engine requires every network interface on a multi-NIC VM to attach to a distinct VPC network. Attaching two NICs to the same VPC or omitting any of the 8 GPU NICs on `a3-megagpu-8g` is rejected by the Compute Engine API.
2. **Jumbo Frames (`MTU 8244`)**: Each of the 8 GPU data networks (`glm-a3mega-net-1..8`) is created with `--mtu=8244` to minimize packetization overhead during large tensor collectives.
3. **Internal-Only Data NICs (`no-address`)**: Only `nic0` (`default` VPC) receives an external IP for SSH and control access. Interfaces `nic1..nic8` are configured with `no-address` so they carry strictly internal east-west GPU collective traffic (`192.168.1.0/24` .. `192.168.8.0/24`).

---

## 2. Repeatable Provisioning & Lifecycle Commands

All infrastructure scripts reside in [`infra/`](../infra) and accept environment variable overrides (`PROJECT_ID`, `ZONE`, `NUM_VMS`, `PREFIX`).

### 2.1 Create VPC Networks, Instance Template & Spot MIG

Run [`infra/provision_a3_mega_mig.sh`](../infra/provision_a3_mega_mig.sh) (uses the active `gcloud config get-value project` automatically, or override via `PROJECT_ID` / `ZONE` / `NUM_VMS`):

```bash
ZONE=us-west1-b NUM_VMS=2 bash infra/provision_a3_mega_mig.sh
```

What this script executes:
1. Creates 8 custom-mode VPC networks (`glm-a3mega-net-1` .. `glm-a3mega-net-8`, `--mtu=8244`), 8 regional subnets in `us-west1` (`glm-a3mega-sub-1` .. `glm-a3mega-sub-8`, `192.168.1.0/24` .. `192.168.8.0/24`), and 8 ingress firewall rules (`glm-a3mega-fw-1` .. `glm-a3mega-fw-8`) allowing `tcp:0-65535,udp:0-65535,icmp` across `192.168.0.0/16` and `10.0.0.0/8`. If the networks already exist, this step is idempotent and completes immediately.
2. Creates the regional instance template `glm-a3mega-tmpl` (`--machine-type=a3-megagpu-8g`, `--provisioning-model=SPOT`, `--image-family=ubuntu-accelerator-2204-amd64-with-nvidia-580`, 9× `GVNIC` interfaces) if not already present.
3. Creates the zonal Managed Instance Group `glm-a3mega-mig` (`--size=2`, `--default-action-on-vm-failure=do-nothing`) and waits until all instances reach `RUNNING` and `stable`.

### 2.2 Inspect Node IP Assignments

Once the MIG is stable, export the node inventory to [`infra/nodes_inventory.json`](../infra/nodes_inventory.json):

```bash
gcloud compute instances list \
  --project="$(gcloud config get-value project)" \
  --filter="name ~ ^glm-a3mega-mig" \
  --format=json > infra/nodes_inventory.json
```

Determine Node 0's `nic1` IP (`HEAD_IP`, e.g., `192.168.1.3`) for multi-node coordination:

```bash
jq -r '.[] | "\(.name): nic0=\(.networkInterfaces[0].networkIP) nic1=\(.networkInterfaces[1].networkIP) ext=\(.networkInterfaces[0].accessConfigs[0].natIP)"' infra/nodes_inventory.json
```

### 2.3 Tear Down Spot MIG (Preserving VPC Networks)

To stop compute billing while preserving the 8 GPU VPC networks (`glm-a3mega-net-1..8`), subnets, firewall rules, and instance template for rapid re-deployment:

```bash
bash infra/teardown_a3_mega_mig.sh
```

To also remove the instance template and all 8 VPC networks, pass `DELETE_TEMPLATE=1 DELETE_NETWORKS=1`:

```bash
DELETE_TEMPLATE=1 DELETE_NETWORKS=1 bash infra/teardown_a3_mega_mig.sh
```

---

## 3. Host Storage & Environment Bootstrap

Each `a3-megagpu-8g` instance includes **16× 375 GB local NVMe SSDs (`5.9 TB` raw)** attached at `/dev/disk/by-id/google-local-nvme-ssd-*`. Because the boot disk is `200 GB` and the two model checkpoints (`zai-org/GLM-5.3-Flash` FP8 `328.4 GB` + `zai-org/GLM-5.3-Flash-BF16` BF16 `642.7 GB`) plus Docker images exceed `1.1 TB`, all runtime state must reside on the local NVMe array.

Run [`infra/bootstrap_node.sh`](../infra/bootstrap_node.sh) on each node:

```bash
scp infra/bootstrap_node.sh <NODE_EXTERNAL_IP>:~/
ssh <NODE_EXTERNAL_IP> "bash ~/bootstrap_node.sh"
```

`bootstrap_node.sh` performs:
1. **RAID-0 Assembly**: Combines all 16 local NVMe drives into `/dev/md0` (`ext4`, mounted at `/mnt/localssd` with `5.9 TB` usable capacity).
2. **Docker Relocation**: Configures NVIDIA Container Toolkit and moves Docker's `data-root` to `/mnt/localssd/docker`.
3. **TCP Buffer Tuning**: Increases `net.core.rmem_max` and `net.core.wmem_max` to `128 MB` (`134217728` bytes) and enables GPU persistence mode (`nvidia-smi -pm 1`).
4. **Container & Downloader Setup**: Pulls `vllm/vllm-openai:nightly` and installs `huggingface_hub`, `hf_transfer`, and `aiohttp`.

### Persistent JIT Kernel Cache on Local NVMe

On first startup, `vllm/vllm-openai:nightly` compiles architecture-specific DeepGEMM, FlashInfer SM90 Sparse MLA, TileLang, and TorchInductor kernels (`~3.5–6 minutes`). Mounting `/mnt/localssd/vllm_root:/root` and `/tmp:/tmp` (implemented in [`serving/serve_single_node.sh`](../serving/serve_single_node.sh) and [`serving/serve_multi_node.sh`](../serving/serve_multi_node.sh)) persists:
- `/root/.cache/flashinfer`
- `/root/.cache/vllm`
- `/root/.tilelang`
- `/root/.triton`
- `/tmp/torchinductor_root`

Subsequent container restarts load pre-compiled kernels from NVMe and become ready in **35–40 seconds**.

---

## 4. Multi-Node Inter-Host Networking: 8-Rail Multi-NIC vs. GPUDirect-TCPXO

### 4.1 8-Rail Dedicated GPU-NIC Socket Transport (`USE_TCPXO=0`, Default)

When running on Ubuntu 22.04 with Linux kernel `6.8.0-1070-gcp`, binding NCCL directly across the 8 PCIe-aligned GPU data interfaces (`enp6s0f0,enp7s0f0,enp13s0f0,enp14s0f0,enp134s0f0,enp135s0f0,enp141s0f0,enp142s0f0`) with rail-local routing (`NCCL_CROSS_NIC=0`) saturates all 8 data VPCs without requiring host `dmabuf` kernel modules:

```bash
-e GLOO_SOCKET_IFNAME=enp0s12
-e NCCL_SOCKET_IFNAME=enp6s0f0,enp7s0f0,enp13s0f0,enp14s0f0,enp134s0f0,enp135s0f0,enp141s0f0,enp142s0f0
-e NCCL_CROSS_NIC=0
-e NCCL_SOCKET_NTHREADS=4
-e NCCL_NSOCKS_PERTHREAD=4
--ulimit nofile=1048576:1048576
--ulimit memlock=-1:-1
```

* **Why `--ulimit nofile=1048576:1048576` is mandatory**: With 16 ranks × 8 interfaces × 16 sockets per interface, NCCL opens thousands of file descriptors during ring/tree initialization. Docker's default soft limit (`1024`) causes `ncclSystemError: Call to socket failed: Too many open files`.
* **Why `NCCL_CROSS_NIC=0` is mandatory**: Because each NIC (`nic1..nic8`) is isolated in its own `/24` VPC (`192.168.1.0/24` .. `192.168.8.0/24`), NIC `i` on Node 0 can only reach NIC `i` on Node 1. Setting `NCCL_CROSS_NIC=0` instructs NCCL to keep each rail strictly on its matching subnet.

### 4.2 Verifying 2-Node (16-GPU) Multi-Rail Bandwidth

To verify all 8 rails and 16 GPUs prior to launching vLLM, run a 2-node PyTorch distributed AllReduce benchmark across `192.168.1.3` (Node 0) and `192.168.1.2` (Node 1):

```bash
# Run on Node 0 (NODE_RANK=0) and Node 1 (NODE_RANK=1):
sudo docker run --rm --gpus all --net=host --ipc=host \
  --ulimit nofile=1048576:1048576 \
  -e NCCL_SOCKET_IFNAME=enp6s0f0,enp7s0f0,enp13s0f0,enp14s0f0,enp134s0f0,enp135s0f0,enp141s0f0,enp142s0f0 \
  -e GLOO_SOCKET_IFNAME=enp0s12 \
  -e NCCL_CROSS_NIC=0 \
  -e NCCL_SOCKET_NTHREADS=4 \
  -e NCCL_NSOCKS_PERTHREAD=4 \
  --entrypoint torchrun vllm/vllm-openai:nightly \
  --nnodes=2 --nproc_per_node=8 --node_rank=${NODE_RANK} \
  --master_addr=192.168.1.3 --master_port=29500 \
  -c '
import torch, torch.distributed as dist, time
dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank % 8)
for mb in [1, 16, 128, 512]:
    t = torch.ones((mb * 1024 * 256,), dtype=torch.float32, device="cuda")
    for _ in range(5): dist.all_reduce(t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 30
    for _ in range(iters): dist.all_reduce(t)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    algbw = (mb / 1024.0) / dt
    busbw = algbw * (2 * (world - 1) / world)
    if rank == 0:
        print(f"{mb:4d} MB | {dt*1e3:6.2f} ms | AlgBW: {algbw:6.2f} GB/s | BusBW: {busbw:6.2f} GB/s")
dist.destroy_process_group()
'
```

Expected output across 2 `a3-megagpu-8g` nodes (`16× H100`):
```text
   1 MB |   0.36 ms | AlgBW:   2.71 GB/s | BusBW:   5.09 GB/s
  16 MB |   1.38 ms | AlgBW:  11.29 GB/s | BusBW:  21.17 GB/s
 128 MB |   6.02 ms | AlgBW:  20.75 GB/s | BusBW:  38.91 GB/s
 512 MB |  21.41 ms | AlgBW:  23.35 GB/s | BusBW:  43.79 GB/s
```

### 4.3 Optional GPUDirect-TCPXO / Fastrak Stack (`USE_TCPXO=1`)

[`infra/setup_tcpxo.sh`](../infra/setup_tcpxo.sh) installs Google Cloud's `libnccl-net.so` Fastrak plugin into `/var/lib/tcpxo/lib64` and starts the `tcpgpudmarxd` Receive Data Path Manager daemon. Note that `tcpgpudmarxd` requires a guest kernel with compatible `nvidia-peermem` / `dmabuf` importer support (such as Container-Optimized OS or DLVM Debian 12 with kernel `6.1`). On Ubuntu 22.04 kernel `6.8.0-1070-gcp`, `tcpgpudmarxd` rejects page pool initialization (`Failed to create dmabuf importer context`), so `USE_TCPXO=0` (8-rail multi-NIC transport) should be used on Ubuntu `6.8` images.

---

## 5. Multi-Node vLLM Operational Checklist

When launching 2-node (`16× H100`) vLLM servers using [`serving/serve_multi_node.sh`](../serving/serve_multi_node.sh):

1. **Stop Existing Containers on Both Nodes First**:
   Ensure any prior `vllm-server` container on Node 0 is stopped before starting Node 1 so Node 1's headless workers do not attach to a shutting-down `torch.distributed` `TCPStore` on port `29501`:
   ```bash
   ssh <NODE_0_IP> "sudo docker rm -f vllm-server || true"
   ssh <NODE_1_IP> "sudo docker rm -f vllm-server || true"
   ```
2. **Pass `--no-numa-bind` on Multi-Node Launches**:
   Included automatically in [`serving/serve_multi_node.sh`](../serving/serve_multi_node.sh). Prevents a thread-safety race in vLLM's `--headless` worker launcher (`_start_workers_concurrently` mutating `multiprocessing.spawn.set_executable`).
3. **Keep `--max-num-batched-tokens 8192`**:
   Included automatically in both serving scripts. FlashInfer's SM90 Sparse MLA kernel (`FlashInferMLASparseSM90`) enforces a native capacity limit of `16,384` work items (`_FA_MAX_WORK_ITEMS = 16384`).
