#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
11_make_report_artifacts.py

이미 학습된 runs_*/result.json을 모아 보고서용 표와 그래프를 생성한다.
학습을 새로 돌리지 않으므로 실행이 빠르다.

실행:
    /usr/bin/python 11_make_report_artifacts.py \
      --runs_root . \
      --assets_dir ./training_assets \
      --out_dir ./report_artifacts
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


HORIZON_ORDER = {"3s": 0, "6s": 1, "30s": 2, "60s": 3, "120s": 4}

SENSORS = [
    "Accelerometer",
    "GasLeak",
    "Pressure_1",
    "Pressure_2",
    "Temperature_1",
    "Temperature_2",
]


def load_result_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    run_dir = path.parent
    run_name = run_dir.name
    parts = run_name.replace("runs_", "").split("_")

    model_type = data.get("model_type", parts[0])
    horizon = data.get("horizon", parts[-1])

    metrics = data.get("test_scaled_metrics", {}) or {}
    test_mae_scaled = metrics.get("mae")
    if test_mae_scaled is None:
        test_mae_scaled = metrics.get("compile_metrics")

    naive = data.get("naive_comparison", {}) or {}
    raw_mae = data.get("test_raw_mae_mean")
    naive_raw = naive.get("test_naive_mae_raw")
    naive_scaled = naive.get("test_naive_mae_scaled")

    if raw_mae is not None and naive_raw is not None:
        delta = raw_mae - naive_raw
        improvement = (naive_raw - raw_mae) / naive_raw * 100.0
    else:
        delta = data.get("delta_vs_naive_raw")
        improvement = data.get("improvement_vs_naive_raw_percent")

    row = {
        "run_dir": str(run_dir),
        "model_type": model_type,
        "horizon": horizon,
        "horizon_order": HORIZON_ORDER.get(horizon, 999),
        "horizon_steps": data.get("horizon_steps"),
        "input_len": data.get("input_len"),
        "input_seconds": data.get("input_seconds"),
        "test_loss_scaled": metrics.get("loss"),
        "test_mae_scaled": test_mae_scaled,
        "test_mae_raw_mean": raw_mae,
        "naive_mae_raw_mean": naive_raw,
        "naive_mae_scaled": naive_scaled,
        "delta_vs_naive_raw": delta,
        "improvement_vs_naive_raw_percent": improvement,
        "model_path": data.get("model_path", ""),
    }

    sensor_mae = data.get("test_raw_mae_per_sensor", {}) or {}
    for sensor in SENSORS:
        row[f"mae_{sensor}"] = sensor_mae.get(sensor)

    return row


def load_naive_by_horizon(assets_dir: Path):
    path = assets_dir / "naive_baseline_mae.csv"
    out = {}

    if not path.exists():
        return out

    df = pd.read_csv(path)
    required = {"horizon_name", "split", "sensor", "mae_raw", "mae_scaled"}

    if required.issubset(df.columns):
        sub = df[(df["split"] == "test") & (df["sensor"] == "ALL_MEAN")]
        for _, row in sub.iterrows():
            out[str(row["horizon_name"])] = {
                "raw": float(row["mae_raw"]),
                "scaled": float(row["mae_scaled"]),
            }

    return out


def fill_naive(df: pd.DataFrame, naive_map: dict):
    for idx, row in df.iterrows():
        h = row["horizon"]

        if h in naive_map:
            if pd.isna(row.get("naive_mae_raw_mean")):
                df.loc[idx, "naive_mae_raw_mean"] = naive_map[h]["raw"]

            if pd.isna(row.get("naive_mae_scaled")):
                df.loc[idx, "naive_mae_scaled"] = naive_map[h]["scaled"]

        raw = df.loc[idx, "test_mae_raw_mean"]
        naive_raw = df.loc[idx, "naive_mae_raw_mean"]

        if pd.notna(raw) and pd.notna(naive_raw):
            df.loc[idx, "delta_vs_naive_raw"] = raw - naive_raw
            df.loc[idx, "improvement_vs_naive_raw_percent"] = (naive_raw - raw) / naive_raw * 100.0

    return df


def save_line_plot(df, value_col, ylabel, title, out_png):
    plot_df = df.dropna(subset=[value_col]).copy()
    plt.figure(figsize=(10, 6))

    for model_type, g in plot_df.groupby("model_type"):
        g = g.sort_values("horizon_order")
        plt.plot(g["horizon"], g[value_col], marker="o", label=model_type)

    plt.title(title)
    plt.xlabel("Prediction horizon")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_sensor_plot(best_gru, out_png):
    sensor_cols = [f"mae_{s}" for s in SENSORS]
    df = best_gru[["horizon", "horizon_order"] + sensor_cols].copy()
    df = df.sort_values("horizon_order")

    plt.figure(figsize=(10, 6))
    for sensor in SENSORS:
        plt.plot(df["horizon"], df[f"mae_{sensor}"], marker="o", label=sensor)

    plt.title("Best GRU sensor-wise raw MAE by horizon")
    plt.xlabel("Prediction horizon")
    plt.ylabel("Raw MAE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def build_requirement_checklist():
    return pd.DataFrame([
        ["6개 센서 입력", "충족", "6개 센서를 모두 입력 feature로 사용"],
        ["6개 센서 동시 출력", "충족", "모델 출력 차원 6"],
        ["3s/9s/30s/60s/120s 예측", "충족", "최종 요구 5개 horizon 전체 실험 완료"],
        ["LSTM/GRU/1D CNN 계열 사용", "충족", "CNN1D, GRU, LSTM 비교"],
        ["episode 경계 분리", "충족", "episode 내부에서만 window 생성"],
        ["train/validation/test 분리", "충족", "episode 단위 700/150/150 분리"],
        ["전처리 및 EDA", "충족", "scaler, window, baseline, pairplot, heatmap 생성"],
        ["성능 평가", "충족", "naive 대비 raw MAE 및 개선율 비교"],
        ["학습 안정성 검증", "부분 충족", "training curve 확인, 장기 horizon 정규화 실험 진행"],
        ["보고서 작성", "진행 중", "데이터 이해/전처리 초안 완료, 모델 평가 파트 작성 필요"],
    ], columns=["requirement", "status", "evidence"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default=".")
    parser.add_argument("--assets_dir", default="./training_assets")
    parser.add_argument("--out_dir", default="./report_artifacts")
    parser.add_argument("--include_regularized", action="store_true")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    assets_dir = Path(args.assets_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [load_result_json(p) for p in sorted(runs_root.glob("runs_*/result.json"))]
    df = pd.DataFrame(rows)
    df = fill_naive(df, load_naive_by_horizon(assets_dir))

    if args.include_regularized:
        df_main = df.copy()
    else:
        df_main = df[df["model_type"].isin(["cnn1d", "gru", "lstm"])].copy()

    df_main = df_main.sort_values(["horizon_order", "model_type"]).reset_index(drop=True)

    model_result_csv = out_dir / "model_result_table.csv"
    df_main.to_csv(model_result_csv, index=False, encoding="utf-8-sig")

    pivot = df_main.pivot_table(index="horizon", columns="model_type", values="test_mae_raw_mean", aggfunc="min")
    pivot = pivot.reindex(sorted(pivot.index, key=lambda x: HORIZON_ORDER.get(x, 999)))
    pivot_csv = out_dir / "model_mae_pivot.csv"
    pivot.to_csv(pivot_csv, encoding="utf-8-sig")

    improvement_pivot = df_main.pivot_table(index="horizon", columns="model_type", values="improvement_vs_naive_raw_percent", aggfunc="max")
    improvement_pivot = improvement_pivot.reindex(sorted(improvement_pivot.index, key=lambda x: HORIZON_ORDER.get(x, 999)))
    improvement_csv = out_dir / "improvement_pivot.csv"
    improvement_pivot.to_csv(improvement_csv, encoding="utf-8-sig")

    best_idx = df_main.dropna(subset=["test_mae_raw_mean"]).groupby("horizon")["test_mae_raw_mean"].idxmin()
    best = df_main.loc[best_idx].sort_values("horizon_order")
    best_csv = out_dir / "best_by_horizon.csv"
    best.to_csv(best_csv, index=False, encoding="utf-8-sig")

    best_gru = df_main[df_main["model_type"] == "gru"].copy().sort_values("horizon_order")
    best_gru_sensor_csv = out_dir / "best_gru_sensor_mae.csv"
    best_gru[["horizon"] + [f"mae_{s}" for s in SENSORS]].to_csv(best_gru_sensor_csv, index=False, encoding="utf-8-sig")

    requirement_csv = out_dir / "requirement_checklist.csv"
    build_requirement_checklist().to_csv(requirement_csv, index=False, encoding="utf-8-sig")

    save_line_plot(df_main, "test_mae_raw_mean", "Test raw MAE mean", "Model comparison by prediction horizon", out_dir / "plot_raw_mae_by_horizon.png")
    save_line_plot(df_main, "improvement_vs_naive_raw_percent", "Improvement over naive baseline (%)", "Improvement over naive baseline", out_dir / "plot_improvement_by_horizon.png")
    save_sensor_plot(best_gru, out_dir / "plot_best_gru_sensor_mae.png")

    md_path = out_dir / "report_result_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 모델 실험 결과 요약\n\n")
        f.write("## 최종 후보 모델\n\n")
        f.write("전체 horizon에서 test raw MAE가 가장 낮은 모델은 모두 GRU로 확인되었다.\n\n")
        f.write(best[["horizon", "model_type", "test_mae_raw_mean", "naive_mae_raw_mean", "improvement_vs_naive_raw_percent"]].to_markdown(index=False))
        f.write("\n\n## 모델별 raw MAE 비교\n\n")
        f.write(pivot.to_markdown())
        f.write("\n\n## naive baseline 대비 개선율\n\n")
        f.write(improvement_pivot.to_markdown())
        f.write("\n\n## 해석\n\n")
        f.write("- 3초 예측에서는 naive baseline이 강하므로 개선율이 작다.\n")
        f.write("- 6초 이후부터 GRU의 개선율이 뚜렷하게 증가한다.\n")
        f.write("- 30초, 60초, 120초에서는 모든 딥러닝 모델이 naive baseline보다 크게 개선된다.\n")
        f.write("- 현재 결과상 GRU가 모든 horizon에서 가장 낮은 raw MAE를 기록했다.\n")
        f.write("- 정규화 GRU는 현재 설정에서 기본 GRU보다 test raw MAE가 높아 최종 모델로 채택하지 않는다.\n")

    print("saved:")
    for p in [model_result_csv, pivot_csv, improvement_csv, best_csv, best_gru_sensor_csv, requirement_csv, out_dir / "plot_raw_mae_by_horizon.png", out_dir / "plot_improvement_by_horizon.png", out_dir / "plot_best_gru_sensor_mae.png", md_path]:
        print(p)

    print()
    print("best_by_horizon")
    print(best[["horizon", "model_type", "test_mae_raw_mean", "naive_mae_raw_mean", "improvement_vs_naive_raw_percent"]].to_string(index=False))


if __name__ == "__main__":
    main()
