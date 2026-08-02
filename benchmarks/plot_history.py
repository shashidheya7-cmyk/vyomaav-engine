"""VYOMAAV Automated Benchmark History Visualization & Trend Plotter."""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_benchmark_trend_plots(
    json_path: str = "./benchmarks/history.json",
    csv_path: str = "./benchmarks/history.csv",
    output_png: str = "./benchmarks/history_trends.png"
):
    """Reads benchmark history records and generates multi-panel performance trend graphs."""
    if not os.path.exists(json_path) and not os.path.exists(csv_path):
        print(f"Warning: History files not found in './benchmarks/'. Run benchmark suite first.")
        return

    # Parse directly from structured history.json for complete safety
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            records = json.load(f)
        
        rows = []
        for r in records:
            m = r.get("metrics", {})
            perf = r.get("performance", {})
            rows.append({
                "git_commit": r.get("git_commit", "unknown"),
                "chamfer_m": m.get("chamfer_distance", 0.0),
                "occupancy_iou": m.get("volumetric_iou", 0.0),
                "f_score": m.get("f_score", 0.0),
                "total_latency_sec": perf.get("total_latency_sec", 0.0),
                "cpu_ram_mb": perf.get("cpu_ram_rss_mb", 0.0)
            })
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(csv_path, on_bad_lines="skip")

    if df.empty:
        print("Warning: Benchmark history is empty.")
        return

    df["run_label"] = [f"#{i+1}\n({commit[:6]})" for i, commit in enumerate(df["git_commit"])]

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VYOMAAV Pipeline Performance & Benchmark History Trends", fontsize=16, fontweight="bold")

    # 1. Chamfer Distance (m)
    axes[0, 0].plot(df["run_label"], df["chamfer_m"], marker="o", color="#e74c3c", linewidth=2.5)
    axes[0, 0].set_title("Chamfer Distance (m) ↓", fontweight="bold")
    axes[0, 0].set_ylabel("Meters")

    # 2. Volumetric Coverage & Accuracy (%)
    axes[0, 1].plot(df["run_label"], df["occupancy_iou"] * 100, marker="s", color="#2ecc71", label="Occupancy IoU (%)", linewidth=2.5)
    axes[0, 1].plot(df["run_label"], df["f_score"] * 100, marker="^", color="#3498db", label="F-Score @ 5cm (%)", linewidth=2.5)
    axes[0, 1].set_title("Volumetric Coverage & Accuracy (%) ↑", fontweight="bold")
    axes[0, 1].set_ylabel("Percentage (%)")
    axes[0, 1].legend()

    # 3. Execution Latency (s)
    axes[1, 0].plot(df["run_label"], df["total_latency_sec"], marker="d", color="#f39c12", linewidth=2.5)
    axes[1, 0].set_title("Total Pipeline Latency (sec) ↓", fontweight="bold")
    axes[1, 0].set_ylabel("Seconds")

    # 4. Host CPU RAM (MB)
    axes[1, 1].plot(df["run_label"], df["cpu_ram_mb"], marker="v", color="#9b59b6", linewidth=2.5)
    axes[1, 1].set_title("Host CPU RAM Consumption (MB)", fontweight="bold")
    axes[1, 1].set_ylabel("Megabytes (MB)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_png, dpi=300)
    print(f"Generated benchmark trend visualization plot: '{output_png}'")


if __name__ == "__main__":
    generate_benchmark_trend_plots()
