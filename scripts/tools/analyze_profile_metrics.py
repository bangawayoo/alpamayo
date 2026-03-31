#!/usr/bin/env python3
"""Analyze self-play and inner-step profiling JSONL logs.

Usage:
  python scripts/tools/analyze_profile_metrics.py --run-dir outputs/profile_workers4
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def _pctl(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(q * (len(s) - 1))
    return s[idx]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def analyze_selfplay(run_dir: Path) -> None:
    files = sorted((run_dir / "profiles").glob("selfplay_profile_rank*.jsonl"))
    if not files:
        print("No selfplay profile files found.")
        return

    print("\n=== SelfPlay profile summary ===")
    for path in files:
        rows = _load_jsonl(path)
        sections = [r for r in rows if r.get("event") == "section_complete"]
        iters = [r for r in rows if r.get("event") == "iteration_complete"]

        by_stage: dict[tuple[str, str], list[float]] = defaultdict(list)
        bubbles: list[tuple[float, int, str, str]] = []
        for r in sections:
            phase = r.get("phase", "?")
            stage = r.get("stage", "?")
            dur = float(r.get("duration_s", 0.0))
            by_stage[(phase, stage)].append(dur)
            b = r.get("bubble_before_gpu_s")
            if isinstance(b, (int, float)):
                bubbles.append((float(b), int(r.get("iteration", -1)), phase, stage))

        print(f"\n{path.name}: records={len(rows)} sections={len(sections)}")
        top = sorted(((sum(v), len(v), mean(v), k) for k, v in by_stage.items()), reverse=True)[:8]
        print("Top stage totals: total_s n mean phase/stage")
        for total, n, avg, (phase, stage) in top:
            print(f"  {total:8.2f} {n:3d} {avg:6.2f}  {phase}/{stage}")

        if bubbles:
            print("Top bubbles:")
            for b, it, phase, stage in sorted(bubbles, reverse=True)[:6]:
                print(f"  {b:6.3f}s iter={it} {phase}/{stage}")

        if iters:
            print("Iteration growth (rss, vram_reserved):")
            for r in iters:
                g = r.get("mem_growth_vs_iter_start", {})
                print(
                    f"  iter={r.get('iteration')} dur={r.get('duration_s')}s "
                    f"rss={g.get('rss_gb')}GB vram_res={g.get('cuda_reserved_gb')}GB"
                )


def analyze_trainer(run_dir: Path, bubble_threshold: float) -> None:
    files = sorted(run_dir.glob("iter_*/profiles/sft_trainer_profile_rank*.jsonl"))
    if not files:
        print("No trainer profile files found.")
        return

    print("\n=== Trainer inner-step profile summary ===")

    all_rows = []
    spikes_by_iter_rank: dict[tuple[int, int], set[int]] = defaultdict(set)

    for path in files:
        iter_id = int(path.parts[-3].split("_")[1])
        rank = int(path.stem.split("rank")[-1])
        rows = _load_jsonl(path)
        secs = [r for r in rows if r.get("event") == "training_step_section"]

        by_stage: dict[str, list[float]] = defaultdict(list)
        bubble_by_stage: dict[str, list[float]] = defaultdict(list)
        by_step: dict[int, dict[str, dict]] = defaultdict(dict)
        summaries: dict[int, dict] = {}

        for r in rows:
            if r.get("event") == "training_step_summary":
                summaries[int(r.get("step", -1))] = r

        for r in secs:
            step = int(r.get("step", -1))
            stage = r.get("stage", "?")
            dur = float(r.get("duration_s", 0.0))
            by_stage[stage].append(dur)
            by_step[step][stage] = r
            b = r.get("bubble_before_gpu_s")
            if isinstance(b, (int, float)):
                b = float(b)
                bubble_by_stage[stage].append(b)
                all_rows.append((iter_id, rank, step, stage, b))
                if stage == "sft_forward_backward" and step > 1 and b > bubble_threshold:
                    spikes_by_iter_rank[(iter_id, rank)].add(step)

        print(f"\n{path}: sections={len(secs)}")
        for stage in [
            "sft_forward_backward",
            "expert_phaseA_extract_kv",
            "expert_phaseB_forward_backward",
            "expert_phaseB_optimizer_step",
            "expert_cfm_total",
        ]:
            if stage not in by_stage:
                continue
            d = by_stage[stage]
            b = bubble_by_stage.get(stage, [])
            print(
                f"  {stage:30s} n={len(d):3d} mean_dur={mean(d):.3f}s max_dur={max(d):.3f}s"
                f" bubble_med={(median(b) if b else 0):.3f}s bubble_p90={(_pctl(b, 0.9) if b else 0):.3f}s"
            )

        spikes = sorted(spikes_by_iter_rank[(iter_id, rank)])
        if spikes:
            print(f"  sft_forward_backward spikes > {bubble_threshold:.1f}s at steps: {spikes}")
            for st in spikes[:6]:
                sft = by_step[st].get("sft_forward_backward", {})
                a = by_step[st].get("expert_phaseA_extract_kv", {})
                e = by_step[st].get("expert_cfm_total", {})
                summ = summaries.get(st, {})
                print(
                    f"    step={st} bubble={sft.get('bubble_before_gpu_s')}s "
                    f"sft_dur={sft.get('duration_s')}s phaseA_dur={a.get('duration_s')}s "
                    f"expert_total={e.get('duration_s')}s loss={summ.get('sft_loss')}"
                )

    # Consistency summary across all trainer logs
    stage_bubbles: dict[str, list[float]] = defaultdict(list)
    for _, _, step, stage, b in all_rows:
        if step <= 1:
            continue
        stage_bubbles[stage].append(b)

    print("\nConsistent bubble summary across all trainer files (step>1):")
    print("stage,n,median,mean,p90,max,frac>1s")
    for stage, vals in sorted(stage_bubbles.items(), key=lambda kv: -median(kv[1])):
        print(
            f"{stage},{len(vals)},{median(vals):.3f},{mean(vals):.3f},"
            f"{_pctl(vals, 0.9):.3f},{max(vals):.3f},{sum(v>1.0 for v in vals)/len(vals):.1%}"
        )

    # Cross-rank overlap of sft-forward spikes per iteration
    print("\nCross-rank overlap of sft_forward_backward spikes:")
    iter_ids = sorted({it for it, _ in spikes_by_iter_rank})
    for it in iter_ids:
        s0 = spikes_by_iter_rank.get((it, 0), set())
        s1 = spikes_by_iter_rank.get((it, 1), set())
        if not s0 and not s1:
            continue
        overlap = sorted(s0 & s1)
        print(f"  iter={it} rank0={sorted(s0)} rank1={sorted(s1)} overlap={overlap}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="Run output directory")
    ap.add_argument(
        "--bubble-threshold",
        type=float,
        default=1.0,
        help="Threshold (seconds) used to mark spike bubbles",
    )
    args = ap.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(args.run_dir)

    print(f"Analyzing run dir: {args.run_dir}")
    analyze_selfplay(args.run_dir)
    analyze_trainer(args.run_dir, args.bubble_threshold)


if __name__ == "__main__":
    main()
