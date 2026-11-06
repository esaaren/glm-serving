#!/usr/bin/env python3
"""
2-Node (16x H100) GPUDirect-TCPXO NCCL AllReduce Verification & Bandwidth Benchmark on GKE.
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
    torch.cuda.set_device(local_rank)

    results = []
    sizes_mb = [1, 16, 128, 512, 1024]

    if rank == 0:
        print(f"=== Running {world_size}-GPU Multi-Node GPUDirect-TCPXO NCCL AllReduce Benchmark ===")

    for mb in sizes_mb:
        num_elems = mb * 1024 * 256  # float32 = 4 bytes
        tensor = torch.ones((num_elems,), dtype=torch.float32, device=f"cuda:{local_rank}")

        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        iters = 30
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

        alg_bw_gbs = (mb / 1024.0) / dt
        bus_bw_gbs = alg_bw_gbs * (2.0 * (world_size - 1) / world_size)
        bus_bw_gbps = bus_bw_gbs * 8.0

        if rank == 0:
            print(
                f"{mb:4d} MB | {dt * 1e3:6.2f} ms | "
                f"AlgBW: {alg_bw_gbs:7.2f} GB/s | "
                f"BusBW: {bus_bw_gbs:7.2f} GB/s ({bus_bw_gbps:6.1f} Gbps)"
            )
            results.append({
                "size_mb": mb,
                "latency_ms": round(dt * 1e3, 3),
                "alg_bw_gbs": round(alg_bw_gbs, 2),
                "bus_bw_gbs": round(bus_bw_gbs, 2),
                "bus_bw_gbps": round(bus_bw_gbps, 1),
            })

    if rank == 0:
        out_dir = Path("/mnt/localssd/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "gke_nccl_tcpxo_allreduce.json"
        payload = {
            "world_size": world_size,
            "nccl_fastrak_ifname": os.environ.get("NCCL_FASTRAK_IFNAME", ""),
            "nccl_tuner_plugin": os.environ.get("NCCL_TUNER_PLUGIN", ""),
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("NCCL_TCPXO_JSON_RESULT_START")
        print(json.dumps(payload, indent=2))
        print("NCCL_TCPXO_JSON_RESULT_END")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
