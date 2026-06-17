#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_eda_pairplot.py

목적:
- processed/episodes_npz/*.npz 데이터를 샘플링하여 EDA용 DataFrame을 만든다.
- raw 센서값과 StandardScaler 적용값을 비교한다.
- seaborn pairplot, correlation heatmap을 생성한다.

입력:
    processed/split_manifest.csv
    processed/episodes_npz/*.npz
    training_assets/scaler_stats.npz

출력:
    eda_pairplot/
        eda_sample_raw.csv
        eda_sample_scaled.csv
        corr_raw.csv
        corr_scaled.csv
        pairplot_raw.png
        pairplot_scaled.png
        corr_raw_heatmap.png
        corr_scaled_heatmap.png

실행:
    /usr/bin/python 05_eda_pairplot.py \
      --processed_dir ./processed \
      --assets_dir ./training_assets \
      --out_dir ./eda_pairplot \
      --max_rows 5000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


SENSORS = [
    "Accelerometer",
    "GasLeak",
    "Pressure_1",
    "Pressure_2",
    "Temperature_1",
    "Temperature_2",
]


def resolve_npz_path(row, processed_dir):
    p = Path(str(row["npz_path"]))

    if p.exists():
        return p

    eid = int(row["episode_id"])
    fallback = processed_dir / "episodes_npz" / f"episode_{eid:05d}.npz"

    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"npz not found: {p} or {fallback}")


def make_sample_dataframe(split_manifest, processed_dir, max_rows, seed):
    rng = np.random.default_rng(seed)

    rows = []
    rows_per_split = max_rows // 3

    for split_name in ["train", "val", "test"]:
        split_df = split_manifest[split_manifest["split"] == split_name].copy()

        if split_df.empty:
            continue

        # split별로 episode를 랜덤 선택
        episode_sample_count = min(len(split_df), 30)
        sampled_episodes = split_df.sample(
            n=episode_sample_count,
            random_state=seed
        )

        # episode당 추출할 row 수
        rows_per_episode = max(1, rows_per_split // episode_sample_count)

        for _, row in sampled_episodes.iterrows():
            npz_path = resolve_npz_path(row, processed_dir)
            data = np.load(npz_path, allow_pickle=True)

            values = data["values"].astype(np.float32)
            time_sec = data["time_sec"].astype(np.float32)
            episode_id = int(row["episode_id"])

            idx = rng.choice(
                np.arange(values.shape[0]),
                size=min(rows_per_episode, values.shape[0]),
                replace=False
            )

            for i in idx:
                item = {
                    "split": split_name,
                    "episode_id": episode_id,
                    "time_sec": float(time_sec[i]),
                }

                for j, sensor in enumerate(SENSORS):
                    item[sensor] = float(values[i, j])

                rows.append(item)

    df = pd.DataFrame(rows)

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    return df


def scale_dataframe(df_raw, mean, std):
    df_scaled = df_raw.copy()

    for i, sensor in enumerate(SENSORS):
        df_scaled[sensor] = (df_scaled[sensor] - mean[i]) / std[i]

    return df_scaled


def save_corr_and_heatmap(df, out_csv, out_png, title):
    corr = df[SENSORS].corr()
    corr.to_csv(out_csv, encoding="utf-8-sig")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_pairplot(df, out_png, title):
    plot_df = df[SENSORS + ["split"]].copy()

    g = sns.pairplot(
        plot_df,
        vars=SENSORS,
        hue="split",
        corner=True,
        plot_kws={
            "s": 8,
            "alpha": 0.35
        },
        diag_kws={
            "alpha": 0.5
        }
    )

    g.fig.suptitle(title, y=1.02)
    g.fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./processed")
    parser.add_argument("--assets_dir", default="./training_assets")
    parser.add_argument("--out_dir", default="./eda_pairplot")
    parser.add_argument("--max_rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_manifest_path = processed_dir / "split_manifest.csv"
    scaler_path = assets_dir / "scaler_stats.npz"

    if not split_manifest_path.exists():
        raise FileNotFoundError(split_manifest_path)

    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)

    split_manifest = pd.read_csv(split_manifest_path)

    scaler = np.load(scaler_path, allow_pickle=True)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    print("[1/5] make sampled raw dataframe")
    df_raw = make_sample_dataframe(
        split_manifest=split_manifest,
        processed_dir=processed_dir,
        max_rows=args.max_rows,
        seed=args.seed
    )

    raw_csv = out_dir / "eda_sample_raw.csv"
    df_raw.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    print(f"saved: {raw_csv}")
    print(df_raw.head())
    print(df_raw[SENSORS].describe())

    print("[2/5] make scaled dataframe")
    df_scaled = scale_dataframe(df_raw, mean, std)

    scaled_csv = out_dir / "eda_sample_scaled.csv"
    df_scaled.to_csv(scaled_csv, index=False, encoding="utf-8-sig")
    print(f"saved: {scaled_csv}")
    print(df_scaled[SENSORS].describe())

    print("[3/5] correlation heatmaps")
    save_corr_and_heatmap(
        df_raw,
        out_dir / "corr_raw.csv",
        out_dir / "corr_raw_heatmap.png",
        "Raw sensor correlation"
    )

    save_corr_and_heatmap(
        df_scaled,
        out_dir / "corr_scaled.csv",
        out_dir / "corr_scaled_heatmap.png",
        "Scaled sensor correlation"
    )

    print("[4/5] pairplot raw")
    save_pairplot(
        df_raw,
        out_dir / "pairplot_raw.png",
        "Pairplot - raw sensor values"
    )

    print("[5/5] pairplot scaled")
    save_pairplot(
        df_scaled,
        out_dir / "pairplot_scaled.png",
        "Pairplot - StandardScaler applied"
    )

    print()
    print("done.")
    print("review files:")
    for p in sorted(out_dir.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
