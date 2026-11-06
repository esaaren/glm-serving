#!/usr/bin/env python3
"""
High-Throughput Legal Context Benchmark Runner for vLLM (zai-org/GLM-5.3-Flash & BF16).
Measures maximum token throughput (input tok/s, output tok/s, total tok/s) on legal M&A due
diligence, contract redlining, and litigation discovery workloads.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
from generate_legal_dataset import LEGAL_SYSTEM_PROMPT, build_legal_document


async def send_legal_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    reasoning_effort: str,
    semaphore: asyncio.Semaphore,
    req_id: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }

    async with semaphore:
        t_start = time.perf_counter()
        t_first_token = None
        prompt_tokens = 0
        completion_tokens = 0
        chunks_text = []
        reasoning_text = []
        error_msg = None

        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=1800)) as resp:
                if resp.status != 200:
                    err_body = await resp.text()
                    return {
                        "req_id": req_id,
                        "success": False,
                        "status": resp.status,
                        "error": err_body[:500],
                        "latency": time.perf_counter() - t_start,
                    }

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if data.get("choices"):
                        delta = data["choices"][0].get("delta", {})
                        content_piece = delta.get("content") or ""
                        reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if (content_piece or reasoning_piece) and t_first_token is None:
                            t_first_token = time.perf_counter()
                        if content_piece:
                            chunks_text.append(content_piece)
                        if reasoning_piece:
                            reasoning_text.append(reasoning_piece)

                    if data.get("usage"):
                        u = data["usage"]
                        prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                        completion_tokens = u.get("completion_tokens", completion_tokens)

        except Exception as e:
            error_msg = str(e)

        t_end = time.perf_counter()
        latency = t_end - t_start
        ttft = (t_first_token - t_start) if t_first_token else latency

        return {
            "req_id": req_id,
            "success": error_msg is None and (completion_tokens > 0 or len(chunks_text) > 0 or len(reasoning_text) > 0),
            "error": error_msg,
            "latency": latency,
            "ttft": ttft,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "sample_output": ("".join(reasoning_text) + "\n" + "".join(chunks_text))[:1500] if req_id == 0 else "",
        }


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    semaphore = asyncio.Semaphore(args.concurrency)

    print(f"Preparing {args.num_prompts} legal due-diligence prompts (~{args.input_tokens} input tokens each)...")
    prompts = [
        build_legal_document(seed=5000 + i, target_input_tokens=args.input_tokens)
        for i in range(args.num_prompts)
    ]

    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup request
        if not args.skip_warmup:
            print("Running 1 warmup legal request...")
            warmup_doc = build_legal_document(seed=9999, target_input_tokens=min(1024, args.input_tokens))
            await send_legal_request(
                session=session,
                url=url,
                model=args.model,
                system_prompt=LEGAL_SYSTEM_PROMPT,
                user_prompt=warmup_doc,
                max_tokens=64,
                reasoning_effort=args.reasoning_effort,
                semaphore=semaphore,
                req_id=-1,
            )

        print(
            f"Starting benchmark: model={args.model}, num_prompts={args.num_prompts}, "
            f"concurrency={args.concurrency}, input_tokens~={args.input_tokens}, "
            f"max_tokens={args.output_tokens}, reasoning_effort={args.reasoning_effort}"
        )
        t0 = time.perf_counter()
        tasks = [
            send_legal_request(
                session=session,
                url=url,
                model=args.model,
                system_prompt=LEGAL_SYSTEM_PROMPT,
                user_prompt=prompts[i],
                max_tokens=args.output_tokens,
                reasoning_effort=args.reasoning_effort,
                semaphore=semaphore,
                req_id=i,
            )
            for i in range(args.num_prompts)
        ]
        results: List[Dict[str, Any]] = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    ok = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    total_prompt_tokens = sum(r["prompt_tokens"] for r in ok)
    total_completion_tokens = sum(r["completion_tokens"] for r in ok)
    total_tokens = total_prompt_tokens + total_completion_tokens

    latencies = sorted(r["latency"] for r in ok) if ok else [0.0]
    ttfts = sorted(r["ttft"] for r in ok) if ok else [0.0]

    def pct(arr: List[float], p: float) -> float:
        if not arr:
            return 0.0
        idx = min(len(arr) - 1, max(0, int(round((p / 100.0) * (len(arr) - 1)))))
        return arr[idx]

    summary = {
        "config_name": args.config_name,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "target_input_tokens": args.input_tokens,
        "target_output_tokens": args.output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "successful_requests": len(ok),
        "failed_requests": len(failed),
        "wall_time_sec": round(wall_time, 3),
        "request_throughput_req_s": round(len(ok) / wall_time, 3) if wall_time > 0 else 0.0,
        "input_throughput_tok_s": round(total_prompt_tokens / wall_time, 2) if wall_time > 0 else 0.0,
        "output_throughput_tok_s": round(total_completion_tokens / wall_time, 2) if wall_time > 0 else 0.0,
        "total_throughput_tok_s": round(total_tokens / wall_time, 2) if wall_time > 0 else 0.0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_prompt_tokens_per_req": round(total_prompt_tokens / len(ok), 1) if ok else 0.0,
        "avg_completion_tokens_per_req": round(total_completion_tokens / len(ok), 1) if ok else 0.0,
        "ttft_mean_s": round(statistics.mean(ttfts), 3) if ok else 0.0,
        "ttft_p50_s": round(pct(ttfts, 50), 3),
        "ttft_p90_s": round(pct(ttfts, 90), 3),
        "ttft_p99_s": round(pct(ttfts, 99), 3),
        "e2e_mean_s": round(statistics.mean(latencies), 3) if ok else 0.0,
        "e2e_p50_s": round(pct(latencies, 50), 3),
        "e2e_p90_s": round(pct(latencies, 90), 3),
        "e2e_p99_s": round(pct(latencies, 99), 3),
        "sample_legal_output": next((r["sample_output"] for r in ok if r.get("sample_output")), ""),
        "errors": [r["error"] for r in failed[:5]],
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "sample_legal_output"}, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved benchmark results to {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal M&A Due Diligence Throughput Benchmark for vLLM")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="zai-org/GLM-5.3-Flash")
    parser.add_argument("--config-name", type=str, default="single-node-8xH100")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", type=str, default="low", choices=["low", "high", "max"])
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))
