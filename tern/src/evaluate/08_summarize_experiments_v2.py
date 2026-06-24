#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_summarize_experiments_v2.py

Purpose:
- Read every runs_*/result.json under --runs_root.
- Summarize CNN1D / GRU / LSTM / regularized GRU results across all horizons.
- Compare each model against the naive baseline.
- Save CSV tables and comparison plots for the report.

Outputs:
    summary_results_v2/
        model_comparison_summary_all.csv
        model_mae_pivot_all.csv
        best_model_by_horizon.csv
        model_comparison_mae_full.png
        improvement_vs_naive_full.png
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


HORIZON_ORDER = {
    "3s": 0,
    "6s": 1,
    "30s": 2,
    "60s": 3,
    "120s": 4,
}


def load_result(result_path: Path):
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    run_dir = result_path.parent

    model_type = data.get("model_type", run_dir.name.replace("runs_", "").split("_")[0])
    horizon = data.get("horizon", run_dir.name.replace("runs_", "").split("_")[-1])

    metrics = data.get("test_scaled_metrics", {}) or {}
    naive = data.get("naive_comparison", {}) or {}

    raw_mae = data.get("test_raw_mae_mean", None)
    naive_raw = naive.get("test_naive_mae_raw", data.get("naive_mae_raw_mean", None))
    naive_scaled = naive.get("test_naive_mae_scaled", data.get("naive_mae_scaled", None))

    if raw_mae is not None and naive_raw is not None:
        delta = float(raw_mae) - float(naive_raw)
        improvement = (float(naive_raw) - float(raw_mae)) / float(naive_raw) * 100.0
    else:
        delta = None
        improvement = None

    row = {
        "run_dir": str(run_dir.resolve()),
        "result_json": str(result_path.resolve()),
        "model_type": model_type,
        "horizon": horizon,
        "horizon_steps": data.get("horizon_steps", None),
        "input_len": data.get("input_len", None),
        "input_seconds": data.get("input_seconds", None),
        "test_loss_scaled": metrics.get("loss", None),
        "test_mae_scaled": metrics.get("mae", None),
        "test_mae_raw_mean": raw_mae,
        "naive_mae_raw_mean": naive_raw,
        "naive_mae_scaled": naive_scaled,
        "delta_vs_naive_raw": data.get("delta_vs_naive_raw", delta),
        "improvement_vs_naive_raw_percent": data.get("improvement_vs_naive_raw_percent", improvement),
        "model_path": data.get("model_path", ""),
        "history_csv": data.get("history_csv", ""),
        "sensor_mae_csv": data.get("sensor_mae_csv", ""),
        "horizon_order": HORIZON_ORDER.get(horizon, 999),
    }

    sensor_mae = data.get("test_raw_mae_per_sensor", {}) or {}
    for k, v in sensor_mae.items():
        row[f"mae_{k}"] = v

    return row


def save_model_comparison_plot(df: pd.DataFrame, out_png: Path):
    plt.figure(figsize=(10, 6))

    plot_df = df.dropna(subset=["test_mae_raw_mean"]).copy()
    plot_df = plot_df.sort_values(["model_type", "horizon_order"])

    for model_type, g in plot_df.groupby("model_type"):
        g = g.sort_values("horizon_order")
        plt.plot(g["horizon"], g["test_mae_raw_mean"], marker="o", label=model_type)

    naive_df = (
        plot_df[["horizon", "horizon_order", "naive_mae_raw_mean"]]
        .dropna()
        .drop_duplicates()
        .sort_values("horizon_order")
    )
    if not naive_df.empty:
        plt.plot(
            naive_df["horizon"],
            naive_df["naive_mae_raw_mean"],
            marker="o",
            linestyle="--",
            label="naive",
        )

    plt.title("Model comparison by prediction horizon")
    plt.xlabel("Prediction horizon")
    plt.ylabel("Test raw MAE mean")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_improvement_plot(df: pd.DataFrame, out_png: Path):
    plt.figure(figsize=(10, 6))

    plot_df = df.dropna(subset=["improvement_vs_naive_raw_percent"]).copy()
    plot_df = plot_df.sort_values(["model_type", "horizon_order"])

    for model_type, g in plot_df.groupby("model_type"):
        g = g.sort_values("horizon_order")
        plt.plot(g["horizon"], g["improvement_vs_naive_raw_percent"], marker="o", label=model_type)

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title("Improvement over naive baseline")
    plt.xlabel("Prediction horizon")
    plt.ylabel("Improvement over naive raw MAE (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default=".")
    parser.add_argument("--out_dir", default="./summary_results_v2")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result_paths = sorted(runs_root.glob("runs_*/result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No result.json found under {runs_root}/runs_*")

    rows = []
    for p in result_paths:
        try:
            rows.append(load_result(p))
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["horizon_order", "model_type"]).reset_index(drop=True)

    summary_csv = out_dir / "model_comparison_summary_all.csv"
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    pivot = df.pivot_table(
        index="horizon",
        columns="model_type",
        values="test_mae_raw_mean",
        aggfunc="min",
    )
    pivot = pivot.reindex(sorted(pivot.index, key=lambda x: HORIZON_ORDER.get(x, 999)))
    pivot_csv = out_dir / "model_mae_pivot_all.csv"
    pivot.to_csv(pivot_csv, encoding="utf-8-sig")

    valid = df.dropna(subset=["test_mae_raw_mean"]).copy()
    idx = valid.groupby("horizon")["test_mae_raw_mean"].idxmin()
    best = valid.loc[idx].sort_values("horizon_order")
    best_csv = out_dir / "best_model_by_horizon.csv"
    best.to_csv(best_csv, index=False, encoding="utf-8-sig")

    mae_png = out_dir / "model_comparison_mae_full.png"
    improve_png = out_dir / "improvement_vs_naive_full.png"
    save_model_comparison_plot(df, mae_png)
    save_improvement_plot(df, improve_png)

    print("saved files")
    print(summary_csv)
    print(pivot_csv)
    print(best_csv)
    print(mae_png)
    print(improve_png)
    print()
    print("best model by horizon")
    cols = [
        "horizon",
        "model_type",
        "test_mae_raw_mean",
        "naive_mae_raw_mean",
        "improvement_vs_naive_raw_percent",
        "model_path",
    ]
    print(best[cols].to_string(index=False))
    print()
    print("pivot")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
