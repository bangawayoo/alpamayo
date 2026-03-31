"""Shared lightweight JSONL profiler utilities for training loops.

Captures wall-clock section durations, process RSS, CUDA allocator stats,
and optional GPU bubble estimates (gap between GPU-active sections).
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.distributed as dist


class StageProfiler:
    """Structured section profiler with optional JSONL logging.

    Args:
        enabled: If False, all operations are no-ops.
        log_path: JSONL file path for structured records.
        logger: Logger used for optional human-readable summaries.
        capture_memory: If False, skip RSS/CUDA memory sampling.
        emit_human_logs: If True, emits concise PROFILE log lines.
        human_log_level: Logging level used for human logs.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        log_path: str | Path,
        logger: logging.Logger,
        capture_memory: bool = True,
        emit_human_logs: bool = True,
        human_log_level: int = logging.WARNING,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_path = Path(log_path)
        self.logger = logger
        self.capture_memory = bool(capture_memory)
        self.emit_human_logs = bool(emit_human_logs)
        self.human_log_level = human_log_level

        self._prev_snapshot: dict[str, float] | None = None
        self._last_gpu_section_end_ts: float | None = None
        self._baselines: dict[str, dict[str, float]] = {}

        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _rank() -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    def memory_snapshot(self) -> dict[str, float]:
        """Collect RSS + CUDA memory statistics."""
        if not self.capture_memory:
            return {}

        proc = psutil.Process()
        out: dict[str, float] = {
            "rss_gb": proc.memory_info().rss / 1e9,
            "cuda_alloc_gb": 0.0,
            "cuda_reserved_gb": 0.0,
            "cuda_max_alloc_gb": 0.0,
            "cuda_max_reserved_gb": 0.0,
            "cuda_used_gb": 0.0,
            "cuda_total_gb": 0.0,
        }
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            out["cuda_alloc_gb"] = torch.cuda.memory_allocated(device) / 1e9
            out["cuda_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
            out["cuda_max_alloc_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            out["cuda_max_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
            free_b, total_b = torch.cuda.mem_get_info(device)
            out["cuda_used_gb"] = (total_b - free_b) / 1e9
            out["cuda_total_gb"] = total_b / 1e9
        return out

    def set_baseline(self, key: str, snapshot: dict[str, float] | None = None) -> None:
        if not self.enabled:
            return
        self._baselines[key] = snapshot or self.memory_snapshot()

    def diff_from_baseline(
        self,
        key: str,
        snapshot: dict[str, float] | None = None,
        fields: tuple[str, ...] = ("rss_gb", "cuda_reserved_gb"),
    ) -> dict[str, float]:
        if key not in self._baselines:
            return {}
        snap = snapshot or self.memory_snapshot()
        base = self._baselines[key]
        return {k: snap.get(k, 0.0) - base.get(k, 0.0) for k in fields}

    def append_event(self, record: dict[str, Any], *, active: bool = True) -> None:
        if not (self.enabled and active):
            return
        payload = dict(record)
        payload["rank"] = self._rank()
        payload["ts"] = time.time()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    @contextmanager
    def section(
        self,
        *,
        phase: str,
        stage: str,
        gpu_active: bool,
        active: bool = True,
        sync_cuda: bool = False,
        baseline_key: str | None = None,
        event: str = "section_complete",
        extra: dict[str, Any] | None = None,
        **fields: Any,
    ):
        """Profile one code section.

        Additional keyword fields are included in the emitted record.
        """
        if not (self.enabled and active):
            yield
            return

        if sync_cuda and gpu_active and torch.cuda.is_available():
            torch.cuda.synchronize()

        start_ts = time.perf_counter()
        start_mem = self.memory_snapshot()
        bubble_before_s = None
        if gpu_active and self._last_gpu_section_end_ts is not None:
            bubble_before_s = max(0.0, start_ts - self._last_gpu_section_end_ts)

        try:
            yield
        finally:
            if sync_cuda and gpu_active and torch.cuda.is_available():
                torch.cuda.synchronize()

            end_ts = time.perf_counter()
            end_mem = self.memory_snapshot()
            duration_s = end_ts - start_ts

            if gpu_active:
                self._last_gpu_section_end_ts = end_ts

            delta_mem = {
                "rss_gb": round(end_mem.get("rss_gb", 0.0) - start_mem.get("rss_gb", 0.0), 4),
                "cuda_reserved_gb": round(
                    end_mem.get("cuda_reserved_gb", 0.0)
                    - start_mem.get("cuda_reserved_gb", 0.0),
                    4,
                ),
            }

            growth_vs_prev: dict[str, float] = {}
            if self._prev_snapshot is not None:
                growth_vs_prev = {
                    "rss_gb": round(
                        end_mem.get("rss_gb", 0.0) - self._prev_snapshot.get("rss_gb", 0.0), 4
                    ),
                    "cuda_reserved_gb": round(
                        end_mem.get("cuda_reserved_gb", 0.0)
                        - self._prev_snapshot.get("cuda_reserved_gb", 0.0),
                        4,
                    ),
                }
            self._prev_snapshot = end_mem

            growth_vs_baseline = (
                {k: round(v, 4) for k, v in self.diff_from_baseline(baseline_key, end_mem).items()}
                if baseline_key is not None
                else {}
            )

            record: dict[str, Any] = {
                "event": event,
                "phase": phase,
                "stage": stage,
                "gpu_active": gpu_active,
                "duration_s": round(duration_s, 6),
                "bubble_before_gpu_s": None if bubble_before_s is None else round(bubble_before_s, 6),
                "start_mem": start_mem,
                "end_mem": end_mem,
                "delta_mem": delta_mem,
                "growth_vs_prev": growth_vs_prev,
                "growth_vs_baseline": growth_vs_baseline,
                "extra": extra or {},
            }
            record.update(fields)
            self.append_event(record)

            if self.emit_human_logs:
                self.logger.log(
                    self.human_log_level,
                    "PROFILE phase=%s stage=%s dt=%.4fs RSS=%.2fGB VRAM(res)=%.2fGB Δres=%.2fGB bubble=%s",
                    phase,
                    stage,
                    duration_s,
                    end_mem.get("rss_gb", 0.0),
                    end_mem.get("cuda_reserved_gb", 0.0),
                    delta_mem.get("cuda_reserved_gb", 0.0),
                    "-" if bubble_before_s is None else f"{bubble_before_s:.4f}s",
                )
