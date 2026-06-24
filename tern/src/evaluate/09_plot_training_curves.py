#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_run_name(run_dir: Path):
    name = run_dir.name
    if not name.startswith("runs_"):
        return None, None

    parts = name.replace("runs_", "").split("_")
    if len(parts) < 2:
        return None, None

    model_type = parts[0]
    horizon = parts[1]
    return model_type, horizon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default=".")
    parser.add_argument("--out_dir", default="./summary_results/training_curves")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in runs_root.glob("runs_*") if p.is_dir()])

    rows = []

    for run_dir in run_dirs:
        history_path = run_dir / "history.csv"
        if not history_path.exists():
            continue

        model_type, horizon = parse_run_name(run_dir)
        if model_type is None:
            continue

        df = pd.read_csv(history_path)

        if "mae" not in df.columns or "val_mae" not in df.columns:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["mae"], label="train_mae")
        plt.plot(df["epoch"], df["val_mae"], label="val_mae")
        plt.xlabel("Epoch")
        plt.ylabel("MAE (scaled)")
        plt.title(f"Training curve - {model_type.upper()} {horizon}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_png = out_dir / f"curve_{model_type}_{horizon}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()

        best_idx = int(df["val_mae"].idxmin())
        rows.append({
            "run_dir": str(run_dir),
            "model_type": model_type,
            "horizon": horizon,
            "best_epoch": int(df.loc[best_idx, "epoch"]),
            "best_val_mae": float(df.loc[best_idx, "val_mae"]),
            "last_train_mae": float(df.iloc[-1]["mae"]),
            "last_val_mae": float(df.iloc[-1]["val_mae"]),
            "curve_png": str(out_png),
        })

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "training_curve_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("saved:", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
