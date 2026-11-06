#!/usr/bin/env python3
"""
Generate benchmark summary CSV and high-resolution charts for GLM-5.3-Flash
Single-Node (8x H100) vs Multi-Host (2-Node / 16x H100) vLLM Legal Serving Benchmarks.
"""

import csv
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.9,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8fafc",
})

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results():
    records = {}
    for path in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        name = Path(path).stem
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "config_name" not in data:
            continue
        data["_file"] = Path(path).name
        records[name] = data
    return records


def write_summary_csv(records):
    csv_path = RESULTS_DIR / "summary_metrics.csv"
    fieldnames = [
        "file",
        "config_name",
        "model",
        "concurrency",
        "num_prompts",
        "target_input_tokens",
        "avg_prompt_tokens_per_req",
        "target_output_tokens",
        "avg_completion_tokens_per_req",
        "successful_requests",
        "failed_requests",
        "wall_time_sec",
        "request_throughput_req_s",
        "input_throughput_tok_s",
        "output_throughput_tok_s",
        "total_throughput_tok_s",
        "ttft_mean_s",
        "ttft_p50_s",
        "ttft_p90_s",
        "ttft_p99_s",
        "e2e_mean_s",
        "e2e_p50_s",
        "e2e_p90_s",
        "e2e_p99_s",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, rec in sorted(records.items()):
            row = {k: rec.get(k, "") for k in fieldnames if k != "file"}
            row["file"] = rec["_file"]
            writer.writerow(row)
    print(f"Wrote CSV summary to {csv_path}")


def plot_hbm_and_kv_cache():
    """Chart 1: Per-GPU HBM Breakdown & Cluster KV Cache Capacity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=200)

    configs = [
        "1-Node (8×H100)\nBF16 (TP=8)\n[640 GB HBM]",
        "1-Node (8×H100)\nFP8 (TP=8)\n[640 GB HBM]",
        "2-Node (16×H100)\nBF16 (TP=16)\n[1,280 GB HBM]",
        "2-Node (16×H100)\nFP8 (TP=16)\n[1,280 GB HBM]",
    ]
    weights_gib = np.array([74.65, 39.02, 38.07, 20.27])
    cudagraph_gib = np.array([0.0, 1.54, 3.07, 3.07])
    runtime_act_gib = np.array([6.50, 3.03, 2.40, 1.73])
    kv_cache_gib = np.array([0.0, 30.58, 29.05, 45.84])
    free_reserved_gib = np.array([0.0, 5.42, 7.00, 8.68])

    x = np.arange(len(configs))
    width = 0.54

    b1 = ax1.bar(x, weights_gib, width, label="Model Weights (GiB/GPU)", color="#1e3a8a")
    b2 = ax1.bar(x, cudagraph_gib, width, bottom=weights_gib, label="CUDA Graphs (GiB/GPU)", color="#f59e0b")
    b3 = ax1.bar(
        x,
        runtime_act_gib,
        width,
        bottom=weights_gib + cudagraph_gib,
        label="Activations / Runtime Overhead",
        color="#64748b",
    )
    b4 = ax1.bar(
        x,
        kv_cache_gib,
        width,
        bottom=weights_gib + cudagraph_gib + runtime_act_gib,
        label="Available KV Cache Pool (GiB/GPU)",
        color="#10b981",
    )
    b5 = ax1.bar(
        x,
        free_reserved_gib,
        width,
        bottom=weights_gib + cudagraph_gib + runtime_act_gib + kv_cache_gib,
        label="Headroom (10% Unallocated)",
        color="#cbd5e1",
    )

    ax1.axhline(79.59, color="#dc2626", linestyle="--", linewidth=1.8, label="H100 80GB HBM Limit (79.59 GiB)")
    ax1.text(
        0,
        81.3,
        "OOM Error!\n74.65 GiB Weights\n(0 KV Cache)",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#dc2626",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef2f2", edgecolor="#dc2626"),
    )

    for idx in [1, 2, 3]:
        center_y = weights_gib[idx] + cudagraph_gib[idx] + runtime_act_gib[idx] + kv_cache_gib[idx] / 2
        ax1.text(
            idx,
            center_y,
            f"{kv_cache_gib[idx]:.1f} GiB\nKV/GPU",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
        )
        ax1.text(
            idx,
            weights_gib[idx] / 2,
            f"{weights_gib[idx]:.1f} GiB\nWeights",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    ax1.set_ylabel("Per-GPU Memory Usage (GiB)", fontsize=11, fontweight="bold")
    ax1.set_title("Per-GPU H100 (80GB) HBM Allocation by Configuration", fontsize=12.5, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 93)
    ax1.grid(axis="y", zorder=0)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    # Right subplot: Total KV Cache Capacity (Million Tokens) & Max Concurrency at 131K context
    kv_tokens_millions = np.array([0.0, 2.434, 2.471, 4.044])
    max_concurrency_131k = np.array([0.0, 18.57, 18.85, 30.85])
    colors = ["#ef4444", "#2563eb", "#7c3aed", "#059669"]

    bars = ax2.bar(x, kv_tokens_millions, width=0.5, color=colors, edgecolor="#1e293b", linewidth=0.8)
    ax2.set_ylabel("Cluster KV Cache Capacity (Million Tokens)", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Total KV Cache Capacity & Max 131K-Context Concurrency",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 4.55)
    ax2.grid(axis="y", zorder=0)

    for idx, bar in enumerate(bars):
        h = bar.get_height()
        if h == 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                0.12,
                "CANNOT FIT\n(OOM on 8×H100)",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="#dc2626",
            )
        else:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.08,
                f"{h:.2f}M Tokens\n({max_concurrency_131k[idx]:.1f}× @ 131K ctx)",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
                color="#0f172a",
            )

    plt.suptitle(
        "GLM-5.3-Flash (321B MoE / MLA): Single-Node (8×H100) vs. 2-Node (16×H100) Memory & KV Cache Scaling",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_path = CHARTS_DIR / "hbm_memory_and_kv_cache_breakdown.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_4k_concurrency_scaling(records):
    """Chart 2: 4K Input / 1K Output Legal Redlining Throughput vs Concurrency."""
    concurrencies = [32, 64, 128, 256]

    s_in, s_out, s_tot, s_req = [], [], [], []
    m_in, m_out, m_tot, m_req = [], [], [], []
    fp8_m_in, fp8_m_out, fp8_m_tot, fp8_m_req, fp8_m_c = [], [], [], [], []
    gke_m_in, gke_m_out, gke_m_tot, gke_m_req, gke_m_c = [], [], [], [], []

    for c in concurrencies:
        sk = f"single_node_tp8_fp8_c{c}_4k"
        mk = f"multi_node_tp16_bf16_c{c}_4k"
        fmk = f"multi_node_tp16_fp8_c{c}_4k"
        gmk = f"gke_multi_node_16xH100_tp16_fp8_tcpxo_c{c}_4k"
        if sk in records:
            s_in.append(records[sk]["input_throughput_tok_s"])
            s_out.append(records[sk]["output_throughput_tok_s"])
            s_tot.append(records[sk]["total_throughput_tok_s"])
            s_req.append(records[sk]["request_throughput_req_s"])
        if mk in records:
            m_in.append(records[mk]["input_throughput_tok_s"])
            m_out.append(records[mk]["output_throughput_tok_s"])
            m_tot.append(records[mk]["total_throughput_tok_s"])
            m_req.append(records[mk]["request_throughput_req_s"])
        if fmk in records:
            fp8_m_c.append(c)
            fp8_m_in.append(records[fmk]["input_throughput_tok_s"])
            fp8_m_out.append(records[fmk]["output_throughput_tok_s"])
            fp8_m_tot.append(records[fmk]["total_throughput_tok_s"])
            fp8_m_req.append(records[fmk]["request_throughput_req_s"])
        if gmk in records:
            gke_m_c.append(c)
            gke_m_in.append(records[gmk]["input_throughput_tok_s"])
            gke_m_out.append(records[gmk]["output_throughput_tok_s"])
            gke_m_tot.append(records[gmk]["total_throughput_tok_s"])
            gke_m_req.append(records[gmk]["request_throughput_req_s"])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), dpi=200)

    # Panel 1: Total Throughput (tok/s)
    ax = axes[0]
    ax.plot(
        concurrencies[: len(s_tot)],
        s_tot,
        marker="o",
        linewidth=2.6,
        markersize=8,
        color="#2563eb",
        label="1-Node 8×H100 (TP=8, FP8)",
    )
    ax.plot(
        concurrencies[: len(m_tot)],
        m_tot,
        marker="s",
        linewidth=2.6,
        markersize=8,
        color="#7c3aed",
        label="2-Node 16×H100 (TP=16, BF16)",
    )
    if fp8_m_tot:
        ax.plot(
            fp8_m_c,
            fp8_m_tot,
            marker="^",
            linewidth=2.4,
            linestyle="--",
            markersize=8,
            color="#059669",
            label="2-Node 16×H100 (TP=16, FP8, 8-Rail TCP)",
        )
    if gke_m_tot:
        ax.plot(
            gke_m_c,
            gke_m_tot,
            marker="D",
            linewidth=2.6,
            markersize=8,
            color="#dc2626",
            label="GKE 2-Node 16×H100 (TP=16, FP8, GPUDirect-TCPXO)",
        )
    for c, v in zip(concurrencies[: len(s_tot)], s_tot):
        ax.annotate(f"{v:,.0f}", (c, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#1d4ed8")
    for c, v in zip(concurrencies[: len(m_tot)], m_tot):
        ax.annotate(f"{v:,.0f}", (c, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, fontweight="bold", color="#6d28d9")
    ax.set_title("Total Token Throughput (Prefill + Decode)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Concurrent Legal Redline Requests", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Total Throughput (tokens / sec)", fontsize=10.5, fontweight="bold")
    ax.set_xticks(concurrencies)
    ax.set_ylim(0, max(s_tot + m_tot + [32000]) * 1.12)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel 2: Output Decode Throughput (tok/s)
    ax = axes[1]
    ax.plot(
        concurrencies[: len(s_out)],
        s_out,
        marker="o",
        linewidth=2.6,
        markersize=8,
        color="#059669",
        label="1-Node 8×H100 (TP=8, FP8)",
    )
    ax.plot(
        concurrencies[: len(m_out)],
        m_out,
        marker="s",
        linewidth=2.6,
        markersize=8,
        color="#d97706",
        label="2-Node 16×H100 (TP=16, BF16)",
    )
    if fp8_m_out:
        ax.plot(
            fp8_m_c,
            fp8_m_out,
            marker="^",
            linewidth=2.4,
            linestyle="--",
            markersize=8,
            color="#0d9488",
            label="2-Node 16×H100 (TP=16, FP8, 8-Rail TCP)",
        )
    if gke_m_out:
        ax.plot(
            gke_m_c,
            gke_m_out,
            marker="D",
            linewidth=2.6,
            markersize=8,
            color="#dc2626",
            label="GKE 2-Node 16×H100 (TP=16, FP8, GPUDirect-TCPXO)",
        )
    for c, v in zip(concurrencies[: len(s_out)], s_out):
        ax.annotate(f"{v:,.0f}", (c, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#047857")
    for c, v in zip(concurrencies[: len(m_out)], m_out):
        ax.annotate(f"{v:,.0f}", (c, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, fontweight="bold", color="#b45309")
    ax.set_title("Output Decode Throughput (Legal Memo Generation)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Concurrent Legal Redline Requests", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Output Decode Throughput (tokens / sec)", fontsize=10.5, fontweight="bold")
    ax.set_xticks(concurrencies)
    ax.set_ylim(0, max(s_out + m_out + [6500]) * 1.15)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel 3: Request Throughput (req/s)
    ax = axes[2]
    ax.plot(
        concurrencies[: len(s_req)],
        s_req,
        marker="o",
        linewidth=2.6,
        markersize=8,
        color="#0284c7",
        label="1-Node 8×H100 (TP=8, FP8)",
    )
    ax.plot(
        concurrencies[: len(m_req)],
        m_req,
        marker="s",
        linewidth=2.6,
        markersize=8,
        color="#e11d48",
        label="2-Node 16×H100 (TP=16, BF16)",
    )
    if fp8_m_req:
        ax.plot(
            fp8_m_c,
            fp8_m_req,
            marker="^",
            linewidth=2.4,
            linestyle="--",
            markersize=8,
            color="#059669",
            label="2-Node 16×H100 (TP=16, FP8, 8-Rail TCP)",
        )
    if gke_m_req:
        ax.plot(
            gke_m_c,
            gke_m_req,
            marker="D",
            linewidth=2.6,
            markersize=8,
            color="#dc2626",
            label="GKE 2-Node 16×H100 (TP=16, FP8, GPUDirect-TCPXO)",
        )
    for c, v in zip(concurrencies[: len(s_req)], s_req):
        ax.annotate(f"{v:.2f} req/s", (c, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#0369a1")
    for c, v in zip(concurrencies[: len(m_req)], m_req):
        ax.annotate(f"{v:.2f} req/s", (c, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, fontweight="bold", color="#be123c")
    ax.set_title("Completed Legal Reviews per Second", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Concurrent Legal Redline Requests", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Request Throughput (requests / sec)", fontsize=10.5, fontweight="bold")
    ax.set_xticks(concurrencies)
    ax.set_ylim(0, max(s_req + m_req + [6.5]) * 1.15)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5)

    plt.suptitle(
        "Enterprise Legal M&A Redlining Benchmark (4K Input / 1K Output): Throughput vs. Batch Concurrency",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out_path = CHARTS_DIR / "throughput_scaling_4k_legal.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_long_context_scaling(records):
    """Chart 3: Context Length Scaling (4K, 16K, 64K, 120K Input Tokens)."""
    tiers = [
        ("4K Input\n(c=256)\nCommercial MSA", "c256_4k"),
        ("16K Input\n(c=64)\nMulti-Exhibit M&A", "c64_16k"),
        ("64K Input\n(c=16)\nM&A Data Room", "c16_64k"),
        ("120K Input\n(c=8)\nLitigation Omnibus", "c8_120k"),
    ]

    labels = [t[0] for t in tiers]
    s_tot, s_in, s_out = [], [], []
    m_tot, m_in, m_out = [], [], []

    for _, suffix in tiers:
        sk = f"single_node_tp8_fp8_{suffix}"
        mk = f"multi_node_tp16_bf16_{suffix}"
        s_rec = records.get(sk, {})
        m_rec = records.get(mk, {})
        s_tot.append(s_rec.get("total_throughput_tok_s", 0.0))
        s_in.append(s_rec.get("input_throughput_tok_s", 0.0))
        s_out.append(s_rec.get("output_throughput_tok_s", 0.0))
        m_tot.append(m_rec.get("total_throughput_tok_s", 0.0))
        m_in.append(m_rec.get("input_throughput_tok_s", 0.0))
        m_out.append(m_rec.get("output_throughput_tok_s", 0.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=200)
    x = np.arange(len(labels))
    width = 0.36

    b1 = ax1.bar(x - width / 2, s_tot, width, label="1-Node 8×H100 (TP=8, FP8)", color="#2563eb", edgecolor="#1e293b")
    b2 = ax1.bar(x + width / 2, m_tot, width, label="2-Node 16×H100 (TP=16, BF16)", color="#7c3aed", edgecolor="#1e293b")

    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 600, f"{h:,.0f}\ntok/s", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#1e40af")
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 600, f"{h:,.0f}\ntok/s", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#5b21b6")

    ax1.set_title("Total Token Throughput Across Legal Document Tiers", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total Throughput (tokens / sec)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, max(s_tot + m_tot + [42000]) * 1.18)
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", fontsize=9.5)

    # Panel 2: Output Decode Throughput across context lengths (log/linear comparison)
    b3 = ax2.bar(x - width / 2, s_out, width, label="1-Node 8×H100 (TP=8, FP8) Decode", color="#059669", edgecolor="#1e293b")
    b4 = ax2.bar(x + width / 2, m_out, width, label="2-Node 16×H100 (TP=16, BF16) Decode", color="#d97706", edgecolor="#1e293b")

    for bar in b3:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 90, f"{h:,.0f}\ntok/s", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#065f46")
    for bar in b4:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 90, f"{h:,.0f}\ntok/s", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#92400e")

    ax2.set_title("Output Decode Throughput Across Legal Document Tiers", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Output Decode Throughput (tokens / sec)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, max(s_out + m_out + [6200]) * 1.18)
    ax2.grid(axis="y")
    ax2.legend(loc="upper right", fontsize=9.5)

    plt.suptitle(
        "GLM-5.3-Flash Long-Context Legal Workload Scaling (4K to 120K Input Tokens)",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out_path = CHARTS_DIR / "long_context_throughput_scaling.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_ttft_and_latency(records):
    """Chart 4: TTFT and E2E Turnaround Latency Across Workloads."""
    tiers = [
        ("4K (c=32)", "c32_4k"),
        ("4K (c=64)", "c64_4k"),
        ("4K (c=128)", "c128_4k"),
        ("4K (c=256)", "c256_4k"),
        ("16K (c=64)", "c64_16k"),
        ("64K (c=16)", "c16_64k"),
        ("120K (c=8)", "c8_120k"),
    ]
    labels = [t[0] for t in tiers]
    s_ttft, m_ttft = [], []
    s_e2e, m_e2e = [], []

    for _, suffix in tiers:
        s_rec = records.get(f"single_node_tp8_fp8_{suffix}", {})
        m_rec = records.get(f"multi_node_tp16_bf16_{suffix}", {})
        s_ttft.append(s_rec.get("ttft_mean_s", 0.0))
        m_ttft.append(m_rec.get("ttft_mean_s", 0.0))
        s_e2e.append(s_rec.get("e2e_mean_s", 0.0))
        m_e2e.append(m_rec.get("e2e_mean_s", 0.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=200)
    x = np.arange(len(labels))
    width = 0.36

    b1 = ax1.bar(x - width / 2, s_ttft, width, label="1-Node 8×H100 (TP=8, FP8)", color="#0284c7")
    b2 = ax1.bar(x + width / 2, m_ttft, width, label="2-Node 16×H100 (TP=16, BF16)", color="#9333ea")

    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}s", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#0369a1")
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}s", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#6b21a8")

    ax1.set_title("Mean Time-To-First-Token (TTFT) Under High Batch Concurrency", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Mean TTFT (seconds)", fontsize=10.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, fontweight="bold", rotation=15)
    ax1.set_ylim(0, max(s_ttft + m_ttft + [25]) * 1.18)
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", fontsize=9)

    b3 = ax2.bar(x - width / 2, s_e2e, width, label="1-Node 8×H100 (TP=8, FP8)", color="#0d9488")
    b4 = ax2.bar(x + width / 2, m_e2e, width, label="2-Node 16×H100 (TP=16, BF16)", color="#e11d48")

    for bar in b3:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}s", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#0f766e")
    for bar in b4:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}s", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#be123c")

    ax2.set_title("Mean End-to-End Batch Completion Time (1,024-Token Legal Memo)", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("Mean E2E Latency (seconds)", fontsize=10.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, fontweight="bold", rotation=15)
    ax2.set_ylim(0, max(s_e2e + m_e2e + [140]) * 1.18)
    ax2.grid(axis="y")
    ax2.legend(loc="upper left", fontsize=9)

    plt.suptitle(
        "Batch Latency Profile: Time-To-First-Token (TTFT) & End-to-End Memo Generation Time",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out_path = CHARTS_DIR / "ttft_and_e2e_latency_profile.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_nccl_multirail_bandwidth():
    """Chart 5: 2-Node 16-GPU Multi-Rail TCP vs. GKE GPUDirect-TCPXO AllReduce Bandwidth."""
    msg_sizes = ["1 MB", "16 MB", "128 MB", "512 MB", "1 GB", "8 GB"]
    tcp_bus_bw_gbs = [5.09, 21.17, 38.91, 43.79, 44.10, 44.20]
    tcpxo_bus_bw_gbs = [16.62, 109.72, 228.21, 284.55, 311.01, 336.32]
    tcpxo_lat_ms = [0.118, 0.287, 1.103, 3.538, 6.473, 47.889]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=200)
    x = np.arange(len(msg_sizes))
    width = 0.36

    b1 = ax1.bar(
        x - width / 2,
        tcp_bus_bw_gbs,
        width,
        label="8-Rail Multi-NIC TCP (Ubuntu 6.8, USE_TCPXO=0)",
        color="#3b82f6",
        edgecolor="#1e293b",
    )
    b2 = ax1.bar(
        x + width / 2,
        tcpxo_bus_bw_gbs,
        width,
        label="GKE GPUDirect-TCPXO (COS + tcpgpudmarxd, NET/FastSocket)",
        color="#10b981",
        edgecolor="#1e293b",
    )

    for bar in b1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 4,
            f"{h:.1f}\nGB/s",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#1d4ed8",
        )
    for bar in b2:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 4,
            f"{h:.1f}\nGB/s",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#047857",
        )

    ax1.set_title("2-Node (16×H100) NCCL AllReduce Bus Bandwidth (GB/s)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("AllReduce Tensor Payload Size", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("NCCL Bus Bandwidth (GB/s)", fontsize=10.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(msg_sizes, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 395)
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", fontsize=8.8)

    tcp_gbps = [b * 8.0 for b in tcp_bus_bw_gbs]
    tcpxo_gbps = [b * 8.0 for b in tcpxo_bus_bw_gbs]

    ax2.plot(
        msg_sizes,
        tcpxo_gbps,
        marker="D",
        linewidth=2.8,
        markersize=8,
        color="#059669",
        label="GKE GPUDirect-TCPXO Throughput (Gbps)",
    )
    ax2.plot(
        msg_sizes,
        tcp_gbps,
        marker="o",
        linewidth=2.4,
        linestyle="--",
        markersize=7,
        color="#2563eb",
        label="8-Rail Multi-NIC TCP Throughput (Gbps)",
    )
    for idx, (gbps, lat) in enumerate(zip(tcpxo_gbps, tcpxo_lat_ms)):
        ax2.annotate(
            f"{gbps:,.0f} Gbps\n({lat:.2f} ms)",
            (idx, gbps),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.3,
            fontweight="bold",
            color="#065f46",
        )
    ax2.set_title("Effective 8-NIC Inter-Node Throughput (Gbps): TCPXO vs. Socket", fontsize=12, fontweight="bold")
    ax2.set_xlabel("AllReduce Tensor Payload Size", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Effective Bus Throughput (Gbps)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 3150)
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=8.8)

    plt.suptitle(
        "GCP A3-Mega (16× H100) Collective Performance: GKE GPUDirect-TCPXO (336.3 GB/s) vs. 8-Rail TCP (43.8 GB/s)",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out_path = CHARTS_DIR / "nccl_multirail_roce_bandwidth.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    records = load_all_results()
    print(f"Loaded {len(records)} JSON benchmark records from {RESULTS_DIR}")
    write_summary_csv(records)
    plot_hbm_and_kv_cache()
    plot_4k_concurrency_scaling(records)
    plot_long_context_scaling(records)
    plot_ttft_and_latency(records)
    plot_nccl_multirail_bandwidth()


if __name__ == "__main__":
    main()
