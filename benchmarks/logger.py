"""VYOMAAV Enriched Benchmark History Logger & Environmental Provenance Tracker."""

import os
import json
import csv
import torch
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional


class BenchmarkLogger:
    """Logs quantitative benchmark runs with complete environmental, model, and git metadata."""

    def __init__(self, output_dir: str = "./benchmarks"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.json_path = os.path.join(self.output_dir, "history.json")
        self.csv_path = os.path.join(self.output_dir, "history.csv")

    def _get_git_metadata(self) -> Dict[str, str]:
        """Retrieves current Git commit hash and active branch name."""
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
            return {"commit": commit, "branch": branch}
        except Exception:
            return {"commit": "unknown", "branch": "unknown"}

    def _get_environment_metadata(self) -> Dict[str, Any]:
        """Captures PyTorch, CUDA runtime, and GPU hardware device specifications."""
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
            "gpu_model": gpu_name
        }

    def log_run(
        self,
        experiment_name: str,
        metrics: Dict[str, Any],
        profiler_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Appends comprehensive benchmark results, system environment, and model metadata."""
        git_meta = self._get_git_metadata()
        env_meta = self._get_environment_metadata()
        run_config = config or {}

        record = {
            "timestamp": datetime.now().isoformat(),
            "git_branch": git_meta["branch"],
            "git_commit": git_meta["commit"],
            "experiment": experiment_name,
            "environment": env_meta,
            "config": run_config,
            "metrics": metrics,
            "performance": profiler_data
        }

        # 1. Update JSON History
        history_data = []
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r") as f:
                    history_data = json.load(f)
            except Exception:
                history_data = []

        history_data.append(record)
        with open(self.json_path, "w") as f:
            json.dump(history_data, f, indent=2)

        # 2. Update CSV History
        file_exists = os.path.exists(self.csv_path)
        flattened = {
            "timestamp": record["timestamp"],
            "git_branch": git_meta["branch"],
            "git_commit": git_meta["commit"],
            "experiment": experiment_name,
            "gpu_model": env_meta["gpu_model"],
            "pytorch_version": env_meta["pytorch_version"],
            "views": run_config.get("num_views", 12),
            "resolution": run_config.get("resolution", "640x480"),
            "abs_rel": metrics.get("abs_rel_error", 0.0),
            "rmse_m": metrics.get("rmse", 0.0),
            "chamfer_m": metrics.get("chamfer_distance", 0.0),
            "hausdorff_m": metrics.get("hausdorff_distance", 0.0),
            "f_score": metrics.get("f_score", 0.0),
            "normal_consistency": metrics.get("normal_consistency", 0.0),
            "occupancy_iou": metrics.get("volumetric_iou", 0.0),
            "total_latency_sec": profiler_data.get("total_latency_sec", 0.0),
            "cpu_ram_mb": profiler_data.get("cpu_ram_rss_mb", 0.0)
        }

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flattened.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(flattened)

        print(f"Logged enriched benchmark record [Branch: {git_meta['branch']} | Commit: {git_meta['commit']} | GPU: {env_meta['gpu_model']}] to '{self.csv_path}'")
