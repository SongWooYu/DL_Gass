#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_summarize_experiments.py

목적
- 여러 모델 학습 결과 result.json을 모아 비교표를 만든다.
- naive_baseline_mae.csv와 결합하여 개선율을 계산한다.

실행 예
/usr/bin/python 08_summarize_experiments.py \
  --runs_root . \
  --assets_dir ./training_assets \
  --out_dir ./summary_results
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


HORIZON_ORDER = {"3s": 0, "6s": 1, "30s": 2, "60s": 3, "120s": 4}


def load_results(runs_root: Path):
    rows = []
    for p in sorted(runs_root.rglob("result.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        rows.append({
            "run_dir": str(p.parent),
            "model_type": data.get("model_type", "lstm" if "lstm" in str(p.parent).lower() else "unknown"),
            "horizon": data.get("horizon"),
            "horizon_steps": data.get("horizon_steps"),
            "test_mae_scaled": data.get("test_scaled_metrics", {}).get("mae", data.get("test_scaled_metrics", {}).get("compile_metrics")),
            "test_loss_scaled": data.get("test_scaled_metrics", {}).get("loss"),
            "test_mae_raw_mean": data.get("test_raw_mae_mean"),
            "model_path": data.get("model_path"),
        })
    return pd.DataFrame(rows)


def load_naive(assets_dir: Path):
    p = assets_dir / "naive_baseline_mae.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df[(df["split"] == "test") & (df["sensor"] == "ALL_MEAN")].copy()
    return df[["horizon_name", "mae_raw", "mae_scaled"]].rename(columns={
        "horizon_name": "horizon",
        "mae_raw": "naive_mae_raw_mean",
        "mae_scaled": "naive_mae_scaled",
    })


def make_plot(df, out_path: Path):
    if df.empty:
        return
    plot_df = df.dropna(subset=["test_mae_raw_mean"]).copy()
    if plot_df.empty:
        return
    plot_df["horizon_order"] = plot_df["horizon"].map(HORIZON_ORDER)
    plot_df = plot_df.sort_values(["model_type", "horizon_order"])

    plt.figure(figsize=(10, 6))
    for model_type, g in plot_df.groupby("model_type"):
        g = g.sort_values("horizon_order")
        plt.plot(g["horizon"], g["test_mae_raw_mean"], marker="o", label=model_type)
    if "naive_mae_raw_mean" in plot_df.columns:
        naive = plot_df[["horizon", "horizon_order", "naive_mae_raw_mean"]].drop_duplicates().sort_values("horizon_order")
        if not naive.empty:
            plt.plot(naive["horizon"], naive["naive_mae_raw_mean"], marker="o", linestyle="--", label="naive")
    plt.xlabel("Prediction horizon")
    plt.ylabel("Test raw MAE mean")
    plt.title("Model comparison by horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default=".")
    parser.add_argument("--assets_dir", default="./training_assets")
    parser.add_argument("--out_dir", default="./summary_results")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(runs_root)
    naive = load_naive(assets_dir)

    if results.empty:
        raise ValueError(f"no result.json found under {runs_root}")

    summary = results.merge(naive, on="horizon", how="left")
    summary["delta_vs_naive_raw"] = summary["test_mae_raw_mean"] - summary["naive_mae_raw_mean"]
    summary["improvement_vs_naive_raw_percent"] = (
        (summary["naive_mae_raw_mean"] - summary["test_mae_raw_mean"]) / summary["naive_mae_raw_mean"] * 100.0
    )
    summary["horizon_order"] = summary["horizon"].map(HORIZON_ORDER)
    summary = summary.sort_values(["horizon_order", "model_type", "run_dir"])

    summary_csv = out_dir / "model_comparison_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    pivot = summary.pivot_table(index="horizon", columns="model_type", values="test_mae_raw_mean", aggfunc="min")
    pivot = pivot.reindex(sorted(pivot.index, key=lambda x: HORIZON_ORDER.get(x, 999)))
    pivot_csv = out_dir / "model_mae_pivot.csv"
    pivot.to_csv(pivot_csv, encoding="utf-8-sig")

    plot_path = out_dir / "model_comparison_mae.png"
    make_plot(summary, plot_path)

    print("saved files")
    print(summary_csv)
    print(pivot_csv)
    print(plot_path)
    print("\nsummary")
    print(summary[["model_type", "horizon", "test_mae_raw_mean", "naive_mae_raw_mean", "improvement_vs_naive_raw_percent", "run_dir"]].to_string(index=False))


if __name__ == "__main__":
    main()
