#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_data_understanding_report.py

목적:
- 학습을 더 돌리기 전에 데이터 자체를 이해하기 위한 보고서 자료를 만든다.
- 한 episode의 6개 센서 시계열을 수치적으로 분석한다.
- 센서 간 상관관계, lag별 자기상관, lag별 변화량을 계산한다.
- 교수님 피드백에 맞춰 window size × horizon 조합별 window 수를 계산한다.
- 현재 Dense 결과와 naive baseline을 비교한다.

사용 예:
    /usr/bin/python 05_data_understanding_report.py \
      --processed_dir ./processed \
      --assets_dir ./training_assets \
      --model_eval ./models_dense_mid/dense_eval_summary.csv \
      --model_history ./models_dense_mid/dense_training_history.csv \
      --episode_id 0 \
      --out_dir ./analysis_data_understanding
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_SENSORS = [
    "Accelerometer",
    "GasLeak",
    "Pressure_1",
    "Pressure_2",
    "Temperature_1",
    "Temperature_2",
]

DEFAULT_HORIZONS = {
    "3s": 30,
    "6s": 60,
    "30s": 300,
    "60s": 600,
    "120s": 1200,
}


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_config(assets_dir):
    config_path = assets_dir / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "sensors": DEFAULT_SENSORS,
        "horizons": DEFAULT_HORIZONS,
        "input_len": 100,
    }


def load_episode(processed_dir, episode_id):
    npz_path = processed_dir / "episodes_npz" / f"episode_{episode_id:05d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"episode npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    values = data["values"].astype(np.float64)
    time_sec = data["time_sec"].astype(np.float64)
    sensors = [str(x) for x in data["sensors"].tolist()]

    return npz_path, values, time_sec, sensors


def make_episode_stats(values, sensors):
    rows = []
    for i, sensor in enumerate(sensors):
        x = values[:, i]
        dx = np.diff(x)

        rows.append({
            "sensor": sensor,
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "first": float(x[0]),
            "last": float(x[-1]),
            "total_change": float(x[-1] - x[0]),
            "mean_step_diff": float(np.mean(dx)),
            "std_step_diff": float(np.std(dx)),
            "positive_diff_ratio": float(np.mean(dx > 0)),
            "negative_diff_ratio": float(np.mean(dx < 0)),
            "zero_diff_ratio": float(np.mean(dx == 0)),
            "max_abs_step_change": float(np.max(np.abs(dx))),
            "large_jump_count_3std": int(np.sum(np.abs(dx - np.mean(dx)) > 3 * np.std(dx))) if np.std(dx) > 0 else 0,
        })

    return pd.DataFrame(rows)


def make_lag_analysis(values, sensors, lags):
    rows = []

    for i, sensor in enumerate(sensors):
        x = values[:, i]

        for lag in lags:
            if lag <= 0 or lag >= len(x):
                continue

            a = x[:-lag]
            b = x[lag:]

            if np.std(a) == 0 or np.std(b) == 0:
                autocorr = np.nan
            else:
                autocorr = float(np.corrcoef(a, b)[0, 1])

            rows.append({
                "sensor": sensor,
                "lag_steps": int(lag),
                "lag_seconds": float(lag * 0.1),
                "autocorr": autocorr,
                "mae_raw": float(np.mean(np.abs(b - a))),
                "rmse_raw": float(np.sqrt(np.mean((b - a) ** 2))),
            })

    return pd.DataFrame(rows)


def make_window_plan(split_manifest, horizons, window_steps):
    split_counts = split_manifest["split"].value_counts().to_dict()

    rows = []
    for w in window_steps:
        for horizon_name, h in horizons.items():
            windows_per_episode = max(0, 3000 - w - int(h) + 1)

            row = {
                "window_steps": int(w),
                "window_seconds": float(w * 0.1),
                "horizon_name": horizon_name,
                "horizon_steps": int(h),
                "horizon_seconds": float(int(h) * 0.1),
                "windows_per_episode": int(windows_per_episode),
                "first_prediction_wait_seconds": float(w * 0.1),
                "total_windows": int(windows_per_episode * len(split_manifest)),
            }

            for split_name in ["train", "val", "test"]:
                row[f"{split_name}_episodes"] = int(split_counts.get(split_name, 0))
                row[f"{split_name}_windows"] = int(windows_per_episode * split_counts.get(split_name, 0))

            rows.append(row)

    return pd.DataFrame(rows)


def compare_dense_with_naive(assets_dir, model_eval_path):
    naive_path = assets_dir / "naive_baseline_mae.csv"

    if not naive_path.exists() or model_eval_path is None or not model_eval_path.exists():
        return pd.DataFrame()

    naive = pd.read_csv(naive_path)
    model = pd.read_csv(model_eval_path)

    naive_all = naive[naive["sensor"] == "ALL_MEAN"].copy()
    model_all = model[model["sensor"] == "ALL_MEAN"].copy()

    rows = []
    for _, m in model_all.iterrows():
        matched = naive_all[
            (naive_all["horizon_name"] == m["horizon_name"])
            & (naive_all["split"] == m["split"])
        ]

        if matched.empty:
            continue

        n = matched.iloc[0]

        rows.append({
            "horizon_name": m["horizon_name"],
            "split": m["split"],
            "model": m["model"],
            "model_mae_scaled": float(m["mae_scaled"]),
            "naive_mae_scaled": float(n["mae_scaled"]),
            "model_minus_naive_scaled": float(m["mae_scaled"] - n["mae_scaled"]),
            "model_mae_raw": float(m["mae_raw"]),
            "naive_mae_raw": float(n["mae_raw"]),
            "model_minus_naive_raw": float(m["mae_raw"] - n["mae_raw"]),
            "model_better_than_naive": bool(m["mae_scaled"] < n["mae_scaled"]),
        })

    return pd.DataFrame(rows)


def plot_all_sensors_zscore(values, time_sec, sensors, out_path):
    plt.figure(figsize=(12, 7))
    for i, sensor in enumerate(sensors):
        x = values[:, i]
        z = (x - np.mean(x)) / (np.std(x) + 1e-12)
        plt.plot(time_sec, z, label=sensor, linewidth=1)

    plt.title("Episode z-score sensor trends")
    plt.xlabel("Time (sec)")
    plt.ylabel("Z-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_corr_heatmap(corr_df, out_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(corr_df.values)
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.colorbar(label="Correlation")
    plt.title("Sensor correlation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_window_counts(window_plan, out_path):
    plt.figure(figsize=(10, 6))
    for horizon_name, g in window_plan.groupby("horizon_name"):
        g = g.sort_values("window_seconds")
        plt.plot(g["window_seconds"], g["train_windows"], marker="o", label=horizon_name)

    plt.title("Train window count by input window size")
    plt.xlabel("Input window size (sec)")
    plt.ylabel("Train windows")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_history(model_history_path, out_path):
    if model_history_path is None or not model_history_path.exists():
        return

    hist = pd.read_csv(model_history_path)

    if not {"epoch", "mae", "val_mae"}.issubset(hist.columns):
        return

    plt.figure(figsize=(9, 5))
    plt.plot(hist["epoch"], hist["mae"], marker="o", label="train_mae")
    plt.plot(hist["epoch"], hist["val_mae"], marker="o", label="val_mae")
    plt.title("Dense training history")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_markdown_summary(
    out_path,
    episode_id,
    episode_npz_path,
    values,
    time_sec,
    sensors,
    stats_df,
    corr_df,
    lag_df,
    window_plan,
    model_compare_df,
):
    lines = []
    lines.append(f"# 데이터 이해 보고서: episode {episode_id:05d}")
    lines.append("")
    lines.append("## 1. Episode 기본 구조")
    lines.append("")
    lines.append(f"- npz file: `{episode_npz_path}`")
    lines.append(f"- values shape: `{values.shape}`")
    lines.append(f"- time range: `{time_sec[0]:.1f}` sec ~ `{time_sec[-1]:.1f}` sec")
    lines.append(f"- sensor order: `{sensors}`")
    lines.append("")
    lines.append("## 2. 핵심 해석")
    lines.append("")
    lines.append("- 이 데이터는 분류가 아니라 미래 센서값을 예측하는 **다변량 시계열 회귀 문제**다.")
    lines.append("- 한 episode 안에서는 6개 센서가 동시에 상승하는 경향이 강하다.")
    lines.append("- 짧은 lag에서는 자기상관이 매우 높기 때문에 3초 예측에서는 naive baseline이 강하다.")
    lines.append("- Temperature_2는 후반부 급상승 구간이 뚜렷하여 긴 horizon 예측 난도가 상대적으로 높다.")
    lines.append("- window size가 커질수록 최초 예측 대기 시간이 길어지고, 학습 window 수가 줄어든다.")
    lines.append("")
    lines.append("## 3. 센서별 통계")
    lines.append("")
    lines.append(stats_df.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## 4. 센서 간 상관관계")
    lines.append("")
    lines.append(corr_df.round(4).to_markdown())
    lines.append("")
    lines.append("## 5. Lag별 자기상관")
    lines.append("")
    pivot_auto = lag_df.pivot(index="sensor", columns="lag_seconds", values="autocorr")
    lines.append(pivot_auto.round(4).to_markdown())
    lines.append("")
    lines.append("## 6. Lag별 평균 변화량 raw MAE")
    lines.append("")
    pivot_mae = lag_df.pivot(index="sensor", columns="lag_seconds", values="mae_raw")
    lines.append(pivot_mae.round(4).to_markdown())
    lines.append("")
    lines.append("## 7. Window size × horizon window 수")
    lines.append("")
    pivot_win = window_plan.pivot(index=["window_steps", "window_seconds"], columns="horizon_name", values="train_windows")
    lines.append(pivot_win.to_markdown())
    lines.append("")

    if not model_compare_df.empty:
        lines.append("## 8. Dense vs naive baseline")
        lines.append("")
        lines.append(model_compare_df.round(6).to_markdown(index=False))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./processed")
    parser.add_argument("--assets_dir", default="./training_assets")
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--window_steps", default="5,10,20,30,50,100,300,600,1200")
    parser.add_argument("--lags", default="5,10,20,30,50,100,300,600,1200")
    parser.add_argument("--model_eval", default="")
    parser.add_argument("--model_history", default="")
    parser.add_argument("--out_dir", default="./analysis_data_understanding")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    assets_dir = Path(args.assets_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(assets_dir)
    horizons = config.get("horizons", DEFAULT_HORIZONS)

    split_manifest_path = processed_dir / "split_manifest.csv"
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"split_manifest not found: {split_manifest_path}")
    split_manifest = pd.read_csv(split_manifest_path)

    episode_npz_path, values, time_sec, sensors = load_episode(processed_dir, args.episode_id)

    stats_df = make_episode_stats(values, sensors)
    corr_df = pd.DataFrame(values, columns=sensors).corr()
    lag_df = make_lag_analysis(values, sensors, parse_int_list(args.lags))
    window_plan = make_window_plan(split_manifest, horizons, parse_int_list(args.window_steps))

    model_eval_path = Path(args.model_eval).resolve() if args.model_eval else None
    model_history_path = Path(args.model_history).resolve() if args.model_history else None
    model_compare_df = compare_dense_with_naive(assets_dir, model_eval_path)

    stats_df.to_csv(out_dir / f"episode_{args.episode_id:05d}_sensor_stats.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(out_dir / f"episode_{args.episode_id:05d}_sensor_correlation.csv", encoding="utf-8-sig")
    lag_df.to_csv(out_dir / f"episode_{args.episode_id:05d}_lag_analysis.csv", index=False, encoding="utf-8-sig")
    window_plan.to_csv(out_dir / "window_horizon_plan.csv", index=False, encoding="utf-8-sig")

    if not model_compare_df.empty:
        model_compare_df.to_csv(out_dir / "dense_vs_naive_comparison.csv", index=False, encoding="utf-8-sig")

    plot_all_sensors_zscore(values, time_sec, sensors, out_dir / f"episode_{args.episode_id:05d}_all_sensors_zscore.png")
    plot_corr_heatmap(corr_df, out_dir / f"episode_{args.episode_id:05d}_sensor_correlation.png")
    plot_window_counts(window_plan, out_dir / "window_train_counts.png")
    plot_training_history(model_history_path, out_dir / "dense_training_curve.png")

    write_markdown_summary(
        out_path=out_dir / "data_understanding_summary.md",
        episode_id=args.episode_id,
        episode_npz_path=episode_npz_path,
        values=values,
        time_sec=time_sec,
        sensors=sensors,
        stats_df=stats_df,
        corr_df=corr_df,
        lag_df=lag_df,
        window_plan=window_plan,
        model_compare_df=model_compare_df,
    )

    print("saved files:")
    for p in sorted(out_dir.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
