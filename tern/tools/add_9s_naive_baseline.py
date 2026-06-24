#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = [
    "Accelerometer",
    "GasLeak",
    "Pressure_1",
    "Pressure_2",
    "Temperature_1",
    "Temperature_2",
]

def resolve_npz_path(row, processed_dir: Path) -> Path:
    p = Path(str(row["npz_path"]))
    if p.exists():
        return p

    eid = int(row["episode_id"])
    fallback = processed_dir / "episodes_npz" / f"episode_{eid:05d}.npz"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"npz not found: {p} or {fallback}")

def load_scaler(assets_dir: Path):
    scaler_path = assets_dir / "scaler_stats.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)

    scaler = np.load(scaler_path, allow_pickle=True)
    mean = scaler["mean"].astype(np.float64)
    std = scaler["std"].astype(np.float64)
    return mean, std

def compute_naive_for_split(split_df, processed_dir: Path, input_len: int, horizon_steps: int, std):
    max_start = 3000 - input_len - horizon_steps
    if max_start < 0:
        raise ValueError("input_len + horizon_steps exceeds episode length")

    starts = np.arange(max_start + 1)
    last_idx = starts + input_len - 1
    target_idx = last_idx + horizon_steps

    abs_sum_raw = np.zeros(6, dtype=np.float64)
    abs_sum_scaled = np.zeros(6, dtype=np.float64)
    n_total = 0

    for _, row in split_df.iterrows():
        npz_path = resolve_npz_path(row, processed_dir)
        data = np.load(npz_path, allow_pickle=True)
        values = data["values"].astype(np.float64)

        y_last = values[last_idx, :]
        y_true = values[target_idx, :]

        delta = y_true - y_last

        abs_sum_raw += np.abs(delta).sum(axis=0)
        abs_sum_scaled += np.abs(delta / std.reshape(1, -1)).sum(axis=0)
        n_total += delta.shape[0]

    mae_raw = abs_sum_raw / max(n_total, 1)
    mae_scaled = abs_sum_scaled / max(n_total, 1)
    return mae_raw, mae_scaled, n_total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./processed")
    parser.add_argument("--assets_dir", default="./training_assets")
    parser.add_argument("--input_len", type=int, default=100)
    parser.add_argument("--horizon_name", default="9s")
    parser.add_argument("--horizon_steps", type=int, default=90)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    assets_dir = Path(args.assets_dir).resolve()

    split_manifest_path = processed_dir / "split_manifest.csv"
    if not split_manifest_path.exists():
        raise FileNotFoundError(split_manifest_path)

    _, std = load_scaler(assets_dir)

    split_manifest = pd.read_csv(split_manifest_path)

    rows = []
    for split_name, split_df in split_manifest.groupby("split"):
        mae_raw, mae_scaled, n_windows = compute_naive_for_split(
            split_df=split_df,
            processed_dir=processed_dir,
            input_len=args.input_len,
            horizon_steps=args.horizon_steps,
            std=std,
        )

        for i, sensor in enumerate(SENSORS):
            rows.append({
                "horizon_name": args.horizon_name,
                "horizon_steps": args.horizon_steps,
                "horizon_seconds": args.horizon_steps * 0.1,
                "split": split_name,
                "sensor": sensor,
                "n_windows": int(n_windows),
                "mae_raw": float(mae_raw[i]),
                "mae_scaled": float(mae_scaled[i]),
            })

        rows.append({
            "horizon_name": args.horizon_name,
            "horizon_steps": args.horizon_steps,
            "horizon_seconds": args.horizon_steps * 0.1,
            "split": split_name,
            "sensor": "ALL_MEAN",
            "n_windows": int(n_windows),
            "mae_raw": float(np.mean(mae_raw)),
            "mae_scaled": float(np.mean(mae_scaled)),
        })

    new_df = pd.DataFrame(rows)

    out_path = assets_dir / "naive_baseline_mae.csv"
    if out_path.exists():
        old_df = pd.read_csv(out_path)
        old_df = old_df[old_df["horizon_name"].astype(str) != args.horizon_name].copy()
        final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"saved: {out_path}")
    print(new_df[new_df["sensor"] == "ALL_MEAN"].sort_values(["split"]).to_string(index=False))

if __name__ == "__main__":
    main()
