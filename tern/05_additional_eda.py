#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

SENSORS = ["Accelerometer", "GasLeak", "Pressure_1", "Pressure_2", "Temperature_1", "Temperature_2"]
HORIZONS = {"3s": 30, "6s": 60, "30s": 300, "60s": 600, "120s": 1200}

def resolve_npz_path(row, processed_dir):
    p = Path(str(row["npz_path"]))
    if p.exists():
        return p
    eid = int(row["episode_id"])
    p2 = Path(processed_dir) / "episodes_npz" / f"episode_{eid:05d}.npz"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"npz not found: {p} or {p2}")

def load_values(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    values = data["values"].astype(np.float64)
    if values.shape != (3000, 6):
        raise ValueError(f"unexpected shape: {npz_path}, {values.shape}")
    if np.isnan(values).any():
        raise ValueError(f"NaN found: {npz_path}")
    return values

def load_scaler(assets_dir):
    p = Path(assets_dir) / "scaler_stats.npz"
    if not p.exists():
        print(f"warning: scaler not found: {p}")
        return None, None
    data = np.load(p, allow_pickle=True)
    return data["mean"].astype(np.float64), data["std"].astype(np.float64)

def make_episode_level_summary(split_manifest, processed_dir):
    rows = []
    for i, (_, row) in enumerate(split_manifest.iterrows(), start=1):
        values = load_values(resolve_npz_path(row, processed_dir))
        out = {"episode_id": int(row["episode_id"]), "split": row["split"]}
        for j, sensor in enumerate(SENSORS):
            v = values[:, j]
            out[f"{sensor}_mean"] = float(np.mean(v))
            out[f"{sensor}_std"] = float(np.std(v))
            out[f"{sensor}_min"] = float(np.min(v))
            out[f"{sensor}_max"] = float(np.max(v))
            out[f"{sensor}_first"] = float(v[0])
            out[f"{sensor}_last"] = float(v[-1])
            out[f"{sensor}_range"] = float(np.max(v) - np.min(v))
        rows.append(out)
        if i % 100 == 0:
            print(f"episode summary: {i}/{len(split_manifest)}")
    return pd.DataFrame(rows)

def make_split_sensor_stats(split_manifest, processed_dir):
    rows = []
    for split_name, g in split_manifest.groupby("split"):
        n_total = 0
        sum_x = np.zeros(6)
        sum_x2 = np.zeros(6)
        min_x = np.full(6, np.inf)
        max_x = np.full(6, -np.inf)
        for _, row in g.iterrows():
            values = load_values(resolve_npz_path(row, processed_dir))
            n_total += values.shape[0]
            sum_x += values.sum(axis=0)
            sum_x2 += (values ** 2).sum(axis=0)
            min_x = np.minimum(min_x, values.min(axis=0))
            max_x = np.maximum(max_x, values.max(axis=0))
        mean = sum_x / n_total
        var = np.maximum((sum_x2 / n_total) - (mean ** 2), 0.0)
        std = np.sqrt(var)
        for j, sensor in enumerate(SENSORS):
            rows.append({"split": split_name, "sensor": sensor, "n_samples": int(n_total),
                         "mean": float(mean[j]), "std": float(std[j]),
                         "min": float(min_x[j]), "max": float(max_x[j]),
                         "range": float(max_x[j] - min_x[j])})
    return pd.DataFrame(rows)

def compute_correlation_for_rows(rows, processed_dir):
    n_total = 0
    sum_x = np.zeros(6)
    sum_xx = np.zeros((6, 6))
    for _, row in rows.iterrows():
        values = load_values(resolve_npz_path(row, processed_dir))
        n_total += values.shape[0]
        sum_x += values.sum(axis=0)
        sum_xx += values.T @ values
    mean = sum_x / n_total
    cov = (sum_xx / n_total) - np.outer(mean, mean)
    cov = (cov + cov.T) / 2
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    return pd.DataFrame(corr, index=SENSORS, columns=SENSORS)

def plot_corr(corr_df, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_df.values, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(SENSORS)))
    ax.set_yticks(np.arange(len(SENSORS)))
    ax.set_xticklabels(SENSORS, rotation=45, ha="right")
    ax.set_yticklabels(SENSORS)
    for i in range(len(SENSORS)):
        for j in range(len(SENSORS)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_pairplot(episode_df, stat_name, out_path):
    cols = [f"{s}_{stat_name}" for s in SENSORS]
    df = episode_df[cols].copy()
    df.columns = SENSORS
    axes = scatter_matrix(df, figsize=(12, 12), diagonal="hist", alpha=0.45, marker=".", range_padding=0.05)
    for ax in axes.flatten():
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig = axes[0, 0].figure
    fig.suptitle(f"Episode-level sensor {stat_name} pairplot", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_split_compare(split_stats, value_col, title, out_path):
    pivot = split_stats.pivot(index="sensor", columns="split", values=value_col).reindex(SENSORS)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Sensor")
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def make_horizon_delta_summary(split_manifest, processed_dir, input_len, mean=None, std=None):
    rows = []
    for horizon_name, horizon_steps in HORIZONS.items():
        w = 3000 - input_len - horizon_steps + 1
        last_idx = np.arange(w) + input_len - 1
        target_idx = last_idx + horizon_steps
        for split_name, g in split_manifest.groupby("split"):
            abs_sum_raw = np.zeros(6)
            sq_sum_raw = np.zeros(6)
            abs_sum_scaled = np.zeros(6)
            n_total = 0
            for _, row in g.iterrows():
                values = load_values(resolve_npz_path(row, processed_dir))
                delta = values[target_idx, :] - values[last_idx, :]
                abs_delta = np.abs(delta)
                abs_sum_raw += abs_delta.sum(axis=0)
                sq_sum_raw += (delta ** 2).sum(axis=0)
                if mean is not None and std is not None:
                    abs_sum_scaled += np.abs(delta / std.reshape(1, -1)).sum(axis=0)
                n_total += w
            mae_raw = abs_sum_raw / n_total
            rmse_raw = np.sqrt(sq_sum_raw / n_total)
            mae_scaled = abs_sum_scaled / n_total if mean is not None else np.full(6, np.nan)
            for j, sensor in enumerate(SENSORS):
                rows.append({"horizon_name": horizon_name, "horizon_steps": horizon_steps,
                             "horizon_seconds": horizon_steps * 0.1, "split": split_name,
                             "sensor": sensor, "n_windows": int(n_total),
                             "mean_abs_delta_raw": float(mae_raw[j]),
                             "rmse_delta_raw": float(rmse_raw[j]),
                             "mean_abs_delta_scaled": float(mae_scaled[j])})
            rows.append({"horizon_name": horizon_name, "horizon_steps": horizon_steps,
                         "horizon_seconds": horizon_steps * 0.1, "split": split_name,
                         "sensor": "ALL_MEAN", "n_windows": int(n_total),
                         "mean_abs_delta_raw": float(np.mean(mae_raw)),
                         "rmse_delta_raw": float(np.mean(rmse_raw)),
                         "mean_abs_delta_scaled": float(np.mean(mae_scaled))})
        print(f"horizon delta done: {horizon_name}")
    return pd.DataFrame(rows)

def plot_horizon_delta(delta_df, value_col, title, out_path):
    df = delta_df[(delta_df["sensor"] == "ALL_MEAN") & (delta_df["split"] == "test")].sort_values("horizon_steps")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["horizon_seconds"], df[value_col], marker="o")
    ax.set_xlabel("Horizon seconds")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./processed")
    parser.add_argument("--assets_dir", default="./training_assets")
    parser.add_argument("--out_dir", default="./eda_report_assets")
    parser.add_argument("--input_len", type=int, default=100)
    parser.add_argument("--max_episodes", type=int, default=0)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    assets_dir = Path(args.assets_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    csv_dir = out_dir / "csv"
    fig_dir = out_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = pd.read_csv(processed_dir / "split_manifest.csv")
    if args.max_episodes and args.max_episodes > 0:
        split_manifest = split_manifest.sort_values(["split", "episode_id"]).groupby("split").head(args.max_episodes).reset_index(drop=True)
        print(f"max_episodes mode: {args.max_episodes} per split, rows={len(split_manifest)}")

    mean, std = load_scaler(assets_dir)

    print("[1/5] episode-level summary")
    episode_df = make_episode_level_summary(split_manifest, processed_dir)
    episode_df.to_csv(csv_dir / "episode_level_summary.csv", index=False, encoding="utf-8-sig")

    print("[2/5] split sensor stats")
    split_stats = make_split_sensor_stats(split_manifest, processed_dir)
    split_stats.to_csv(csv_dir / "split_sensor_stats.csv", index=False, encoding="utf-8-sig")
    split_stats[["split", "sensor", "mean", "std", "min", "max", "range"]].to_csv(
        csv_dir / "split_sensor_mean_std_compare.csv", index=False, encoding="utf-8-sig"
    )

    print("[3/5] correlation heatmaps")
    groups = {"overall": split_manifest}
    for split_name, g in split_manifest.groupby("split"):
        groups[split_name] = g
    for name, g in groups.items():
        corr = compute_correlation_for_rows(g, processed_dir)
        corr.to_csv(csv_dir / f"sensor_correlation_{name}.csv", encoding="utf-8-sig")
        plot_corr(corr, f"Sensor correlation: {name}", fig_dir / f"correlation_heatmap_{name}.png")

    print("[4/5] pairplots and split comparison")
    plot_pairplot(episode_df, "mean", fig_dir / "pairplot_episode_mean.png")
    plot_pairplot(episode_df, "std", fig_dir / "pairplot_episode_std.png")
    plot_split_compare(split_stats, "mean", "Train/Val/Test sensor mean comparison", fig_dir / "split_sensor_mean_compare.png")
    plot_split_compare(split_stats, "std", "Train/Val/Test sensor std comparison", fig_dir / "split_sensor_std_compare.png")

    print("[5/5] horizon delta analysis")
    delta_df = make_horizon_delta_summary(split_manifest, processed_dir, args.input_len, mean, std)
    delta_df.to_csv(csv_dir / "horizon_delta_summary.csv", index=False, encoding="utf-8-sig")
    plot_horizon_delta(delta_df, "mean_abs_delta_scaled", "Test horizon difficulty by scaled mean absolute delta", fig_dir / "horizon_delta_mae_scaled.png")
    plot_horizon_delta(delta_df, "mean_abs_delta_raw", "Test horizon difficulty by raw mean absolute delta", fig_dir / "horizon_delta_mae_raw.png")

    print("done")
    print("CSV files:")
    for p in sorted(csv_dir.glob("*.csv")):
        print(p)
    print("Figure files:")
    for p in sorted(fig_dir.glob("*.png")):
        print(p)

if __name__ == "__main__":
    main()
