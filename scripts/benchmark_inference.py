#!/usr/bin/env python3
"""Inference benchmark baseline, run the same way on Windows and Linux.

Principle 6 of the improvement plan: an optimisation counts only if the benchmark is at or above baseline on BOTH
systems. This records the baseline and checks a later run against it.

    python scripts/benchmark_inference.py run     --url http://127.0.0.1:8090 --model qwen --lane r9700 --out baselines/
    python scripts/benchmark_inference.py compare --baseline baselines/linux-baseline.json --candidate baselines/linux-new.json

`run` sends a fixed prompt set (greedy, fixed seed) `--reps` times through the service, takes prompt-processing and
generation speed from llama-server's own timings, reads the service's hardware log for the same time window (peak
temperature, average and peak watts, energy) and writes one JSON file tagged with OS, llama.cpp servers and git
commit. `compare` exits non-zero if any speed fell by more than `--tolerance` (default 3 %), or tokens per
watt-hour fell by more than that. Standard library only, so it runs unchanged on both systems.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import request as urlrequest

PROMPTS = [
    ("short", "Explain in two sentences why the sky is blue.", 128),
    ("code", "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.", 384),
    ("long-prefill", ("The quick brown fox jumps over the lazy dog. " * 150) + "\nSummarise the text above in one sentence.", 96),
]
HIGHER_IS_BETTER = ("prompt_tokens_per_s", "generation_tokens_per_s", "tokens_per_watt_hour")


def _post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urlrequest.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=timeout) as response:  # noqa: S310 - the user's own service
        return json.loads(response.read().decode("utf-8"))


def _get(url: str, timeout: float = 30) -> Any:
    with urlrequest.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def run_benchmark(url: str, model: str, lane: str | None, reps: int = 3, timeout: float = 600.0, prompts=PROMPTS) -> dict[str, Any]:
    started, rows = time.time(), []
    for name, prompt, max_tokens in prompts:
        for rep in range(reps):
            t0 = time.perf_counter()
            body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_k": 1,
                    "seed": 7, "max_tokens": max_tokens, "stream": False}
            if lane:
                body["lane"] = lane
            reply = _post(f"{url}/v1/chat/completions", body, timeout)
            elapsed = time.perf_counter() - t0
            usage, timings = reply.get("usage") or {}, reply.get("timings") or {}
            generated = usage.get("completion_tokens") or timings.get("predicted_n") or 0
            rows.append({
                "prompt": name, "rep": rep, "seconds": elapsed, "completion_tokens": generated,
                "prompt_tokens": usage.get("prompt_tokens"),
                "prompt_tokens_per_s": timings.get("prompt_per_second"),
                "generation_tokens_per_s": timings.get("predicted_per_second") or (generated / elapsed if elapsed else None),
            })
    finished = time.time()
    summary: dict[str, Any] = {}
    for key in ("prompt_tokens_per_s", "generation_tokens_per_s"):
        values = [r[key] for r in rows if r.get(key)]
        summary[key] = statistics.median(values) if values else None
    hardware: dict[str, Any] = {}
    try:
        hw = _get(f"{url}/api/hardware/summary?start={started}&end={finished}" + (f"&lane={lane}" if lane else ""))
        gpus = hw.get("gpus", {})
        energy = [g["energy_wh"] for g in gpus.values() if g.get("energy_wh") is not None]
        hardware = {"gpus": gpus, "host": hw.get("host")}
        total_tokens = sum(r["completion_tokens"] for r in rows)
        if energy and sum(energy) > 0:
            summary["tokens_per_watt_hour"] = total_tokens / sum(energy)
            summary["energy_wh"] = sum(energy)
        summary["peak_junction_c"] = max(
            (g["temp_junction_c"]["max"] for g in gpus.values() if g["temp_junction_c"]["max"] is not None), default=None)
    except Exception as exc:  # noqa: BLE001 - the speed result is still valid without the hardware log
        hardware = {"error": f"hardware log unavailable: {exc}"}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                                cwd=Path(__file__).resolve().parent.parent).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    try:
        status = _get(f"{url}/api/status")
        servers = [{k: s.get(k) for k in ("target", "model", "lane_keys", "context_window")} for s in status.get("servers", [])]
    except Exception:  # noqa: BLE001
        servers = []
    return {"format": 1, "started_at": started, "finished_at": finished, "host_os": platform.system(), "machine": platform.node(),
            "git_commit": commit, "model": model, "lane": lane, "reps": reps, "servers": servers, "summary": summary,
            "hardware": hardware, "samples": rows}


def compare(baseline: dict[str, Any], candidate: dict[str, Any], tolerance: float = 0.03) -> dict[str, Any]:
    """A candidate passes when no higher-is-better metric dropped by more than `tolerance`. A different OS or model
    is reported as a problem, not silently compared."""
    problems, rows = [], []
    for key in ("host_os", "model"):
        if baseline.get(key) != candidate.get(key):
            problems.append(f"{key} differs: baseline {baseline.get(key)!r} vs candidate {candidate.get(key)!r}")
    for key in HIGHER_IS_BETTER:
        b, c = baseline["summary"].get(key), candidate["summary"].get(key)
        if b is None or c is None:
            rows.append({"metric": key, "baseline": b, "candidate": c, "verdict": "not measured"})
            continue
        change = (c - b) / b if b else 0.0
        verdict = "regression" if change < -tolerance else "ok"
        rows.append({"metric": key, "baseline": b, "candidate": c, "change_pct": round(change * 100, 2), "verdict": verdict})
        if verdict == "regression":
            problems.append(f"{key} fell {abs(change) * 100:.1f}% (baseline {b:.2f}, candidate {c:.2f})")
    return {"ok": not problems, "tolerance": tolerance, "metrics": rows, "problems": problems}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--url", default="http://127.0.0.1:8090")
    run.add_argument("--model", required=True)
    run.add_argument("--lane")
    run.add_argument("--reps", type=int, default=3)
    run.add_argument("--out", default="baselines")
    run.add_argument("--label", default="")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--baseline", required=True)
    cmp_.add_argument("--candidate", required=True)
    cmp_.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_benchmark(args.url, args.model, args.lane, args.reps)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        name = f"{result['host_os'].lower()}-{args.label or time.strftime('%Y%m%dT%H%M%S')}-{args.model.replace('/', '_')}.json"
        (out / name).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result["summary"], indent=1))
        print(f"wrote {out / name}")
        return 0
    verdict = compare(json.loads(Path(args.baseline).read_text("utf-8")), json.loads(Path(args.candidate).read_text("utf-8")), args.tolerance)
    print(json.dumps(verdict, indent=1))
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
