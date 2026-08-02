"""VYOMAAV Modular Pipeline Latency & Memory Profiler Engine."""

import time
import torch
import psutil
from typing import Dict, Any
from contextlib import contextmanager


class PipelineProfiler:
    """Tracks stage execution latency, peak CUDA VRAM, and CPU RAM consumption."""

    def __init__(self):
        self.stage_durations: Dict[str, float] = {}
        self.vram_peaks_mb: Dict[str, float] = {}

    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager measuring execution time and peak VRAM per stage."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.stage_durations[stage_name] = self.stage_durations.get(stage_name, 0.0) + elapsed

            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
                self.vram_peaks_mb[stage_name] = max(self.vram_peaks_mb.get(stage_name, 0.0), peak_vram)

    def report(self) -> Dict[str, Any]:
        """Generates a complete runtime performance and memory breakdown report."""
        total_time = sum(self.stage_durations.values())
        stages = {
            stage: {
                "duration_sec": round(duration, 4),
                "percentage": round((duration / total_time) * 100, 2) if total_time > 0 else 0.0,
                "peak_vram_mb": round(self.vram_peaks_mb.get(stage, 0.0), 2)
            }
            for stage, duration in self.stage_durations.items()
        }
        
        process = psutil.Process()
        cpu_ram_mb = process.memory_info().rss / (1024 ** 2)

        return {
            "total_latency_sec": round(total_time, 4),
            "cpu_ram_rss_mb": round(cpu_ram_mb, 2),
            "stages": stages
        }
