"""VYOMAAV Enriched Benchmark History Logger & Auto-Migrating CSV Exporter."""

import os
import json
import csv
import torch
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional


class BenchmarkLogger:
    """Logs quantitative benchmark runs with complete metadata and auto-migrating CSV schemas."""

    def __init__(self, output_dir: str = "./benchmarks"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.json_path = os.path.join(self.output_dir, "history.json")
        self.csv_path = os.path.join(self.output_dir, "history.csv")

    def _get_git_metadata(self) -> Dict[str, str]:
        """Retrieves active Git commit hash and branch name."""
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
            return {"commit": commit, "branch": branch}
        except Exception:
            return {"commit": "unknown", "branch": "unknown"}

    def _get_environment_metadata(self) -> Dict[str, Any]:
        """Captures hardware and PyTorch runtime environment."""
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
            "gpu_model": gpu_name
        }

    def _rebuild_csv_from_json(self) -> None:
        """Rebuilds history.csv cleanly from history.json to enforce uniform columns."""
        if not os.path.exists(self.json_path):
            return

        try:
            with open(self.json_path, "r") as f:
                records = json.load(f)
        except Exception:
            return

        rows = []
        for r in records:
            env = r.get("environment", {})
            cfg = r.get("config", {})
            m = r.get("metrics", {})
            perf = r.get("performance", {})

            rows.append({
                "timestamp": r.get("timestamp", ""),
                "git_branch": r.get("git_branch", "unknown"),
                "git_commit": r.get("git_commit", "unknown"),
                "experiment": r.get("experiment", "unknown"),
                "gpu_model": env.get("gpu_model", "unknown"),
                "pytorch_version": env.get("pytorch_version", "unknown"),
                "views": cfg.get("num_views", 12),
                "resolution": cfg.get("resolution", "640x480"),
                "abs_rel": m.get("abs_rel_error", 0.0),
                "rmse_m": m.get("rmse", 0.0),
                "chamfer_m": m.get("chamfer_distance", 0.0),
                "hausdorff_m": m.get("hausdorff_distance", 0.0),
                "f_score": m.get("f_score", 0.0),
                "normal_consistency": m.get("normal_consistency", 0.0),
                "occupancy_iou": m.get("volumetric_iou", 0.0),
                "total_latency_sec": perf.get("total_latency_sec", 0.0),
                "cpu_ram_mb": perf.get("cpu_ram_rss_mb", 0.0)
            })

        if rows:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    def log_run(
        self,
        experiment_name: str,
        metrics: Dict[str, Any],
        profiler_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Appends benchmark record to history.json and syncs history.csv."""
        git_meta = self._get_git_metadata()
        env_meta = self._get_environment_metadata()

        record = {
            "timestamp": datetime.now().isoformat(),
            "git_branch": git_meta["branch"],
            "git_commit": git_meta["commit"],
            "experiment": experiment_name,
            "environment": env_meta,
            "config": config or {},
            "metrics": metrics,
            "performance": profiler_data
        }

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

        # Sync CSV cleanly from JSON history
        self._rebuild_csv_from_json()
        print(f"Logged run [Commit: {git_meta['commit']} | GPU: {env_meta['gpu_model']}] to '{self.csv_path}'")
