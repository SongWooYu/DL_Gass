#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_train_dense_baseline.py

Step 4: Dense baseline 모델 학습

목표:
- Step 3에서 만든 training_assets를 사용한다.
- horizon별로 5개 모델을 학습한다.
    3s, 6s, 30s, 60s, 120s
- 입력: 과거 10초 = (100, 6)
- 출력: 미래 특정 시점의 6개 센서값 = (6,)
- X/y 모두 train scaler로 표준화한다.
- 전체 window를 메모리에 만들지 않고, 배치 단위로 npz에서 잘라온다.

사용 예:
    python 04_train_dense_baseline.py \
        --processed_dir ./processed \
        --assets_dir ./training_assets \
        --out_dir ./models_dense \
        --epochs 20 \
        --batch_size 256

빠른 테스트:
    python 04_train_dense_baseline.py \
        --processed_dir ./processed \
        --assets_dir ./training_assets \
        --out_dir ./models_dense_test \
        --epochs 2 \
        --train_batches 50 \
        --val_batches 20

산출물:
    models_dense/
        dense_3s.keras
        dense_6s.keras
        ...
        dense_training_history.csv
        dense_eval_summary.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SENSORS = [
    "Accelerometer",
    "GasLeak",
    "Pressure_1",
    "Pressure_2",
    "Temperature_1",
    "Temperature_2",
]


class RandomWindowSequence(keras.utils.Sequence):
    """
    랜덤 window 배치 생성기.
    전체 window를 메모리에 만들지 않는다.

    train:
        매 batch마다 episode와 start를 랜덤 샘플링한다.

    val/test:
        seed를 다르게 주고 고정 난수 기반으로 샘플링한다.
        정확한 전체 평가가 아니라 빠른 검증용이다.
    """

    def __init__(
        self,
        manifest_df,
        processed_dir,
        mean,
        std,
        input_len,
        horizon_steps,
        batch_size,
        batches_per_epoch,
        seed=42,
    ):
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.input_len = int(input_len)
        self.horizon_steps = int(horizon_steps)
        self.batch_size = int(batch_size)
        self.batches_per_epoch = int(batches_per_epoch)
        self.rng = np.random.default_rng(seed)

        self.max_start = 3000 - self.input_len - self.horizon_steps
        if self.max_start < 0:
            raise ValueError("max_start < 0. input_len/horizon_steps 설정 오류")

        if len(self.manifest_df) == 0:
            raise ValueError("manifest_df is empty")

    def __len__(self):
        return self.batches_per_epoch

    def _resolve_npz_path(self, row):
        p = Path(str(row["npz_path"]))
        if p.exists():
            return p

        eid = int(row["episode_id"])
        fallback = self.processed_dir / "episodes_npz" / f"episode_{eid:05d}.npz"
        if fallback.exists():
            return fallback

        raise FileNotFoundError(f"npz not found: {p} or {fallback}")

    def __getitem__(self, idx):
        x = np.zeros((self.batch_size, self.input_len, 6), dtype=np.float32)
        y = np.zeros((self.batch_size, 6), dtype=np.float32)

        episode_indices = self.rng.integers(0, len(self.manifest_df), size=self.batch_size)
        start_indices = self.rng.integers(0, self.max_start + 1, size=self.batch_size)

        # 같은 batch 안에서 같은 episode가 여러 번 나올 수 있다.
        # 구현을 단순하게 유지한다.
        for i in range(self.batch_size):
            row = self.manifest_df.iloc[int(episode_indices[i])]
            start = int(start_indices[i])

            npz_path = self._resolve_npz_path(row)
            data = np.load(npz_path, allow_pickle=True)
            values = data["values"].astype(np.float32)

            end = start + self.input_len
            target_idx = end - 1 + self.horizon_steps

            x_raw = values[start:end, :]
            y_raw = values[target_idx, :]

            x[i] = (x_raw - self.mean) / self.std
            y[i] = (y_raw - self.mean) / self.std

        return x, y


def build_dense_model(input_len, n_features, learning_rate):
    """
    가벼운 Dense baseline.
    시계열 순서를 직접 모델링하지는 않고 flatten해서 회귀한다.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_len, n_features)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(6),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model


def inverse_scaled_mae(y_true_scaled, y_pred_scaled, std):
    """
    scaled 예측을 raw 단위 MAE로 환산한다.
    """
    err_raw = np.abs((y_true_scaled - y_pred_scaled) * std.reshape(1, -1))
    return err_raw.mean(axis=0)


def evaluate_generator_raw_mae(model, seq, std, max_batches):
    """
    검증 generator 일부 batch에 대해 센서별 raw MAE 계산.
    """
    abs_sum = np.zeros(6, dtype=np.float64)
    n = 0

    for i in range(min(len(seq), max_batches)):
        x, y_true = seq[i]
        y_pred = model.predict(x, verbose=0)
        err_raw = np.abs((y_true - y_pred) * std.reshape(1, -1))
        abs_sum += err_raw.sum(axis=0)
        n += y_true.shape[0]

    mae_raw = abs_sum / max(n, 1)
    return mae_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./processed")
    parser.add_argument("--assets_dir", default="./training_assets")
    parser.add_argument("--out_dir", default="./models_dense")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_batches", type=int, default=300)
    parser.add_argument("--val_batches", type=int, default=80)
    parser.add_argument("--test_batches", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only_horizon", default="", help="예: 3s 또는 120s. 비우면 전체 horizon 학습")

    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = assets_dir / "training_config.json"
    scaler_path = assets_dir / "scaler_stats.npz"
    split_manifest_path = processed_dir / "split_manifest.csv"

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)
    if not split_manifest_path.exists():
        raise FileNotFoundError(split_manifest_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    scaler = np.load(scaler_path, allow_pickle=True)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    split_manifest = pd.read_csv(split_manifest_path)

    train_df = split_manifest[split_manifest["split"] == "train"].copy()
    val_df = split_manifest[split_manifest["split"] == "val"].copy()
    test_df = split_manifest[split_manifest["split"] == "test"].copy()

    input_len = int(config["input_len"])
    horizons = config["horizons"]

    if args.only_horizon:
        if args.only_horizon not in horizons:
            raise ValueError(f"unknown horizon: {args.only_horizon}")
        horizons = {args.only_horizon: horizons[args.only_horizon]}

    print("TensorFlow:", tf.__version__)
    print("train/val/test episodes:", len(train_df), len(val_df), len(test_df))
    print("input_len:", input_len)
    print("horizons:", horizons)

    history_rows = []
    eval_rows = []

    for horizon_name, horizon_steps in horizons.items():
        print()
        print("=" * 70)
        print(f"Train Dense baseline: horizon={horizon_name}, steps={horizon_steps}")
        print("=" * 70)

        train_seq = RandomWindowSequence(
            manifest_df=train_df,
            processed_dir=processed_dir,
            mean=mean,
            std=std,
            input_len=input_len,
            horizon_steps=horizon_steps,
            batch_size=args.batch_size,
            batches_per_epoch=args.train_batches,
            seed=args.seed + int(horizon_steps),
        )

        val_seq = RandomWindowSequence(
            manifest_df=val_df,
            processed_dir=processed_dir,
            mean=mean,
            std=std,
            input_len=input_len,
            horizon_steps=horizon_steps,
            batch_size=args.batch_size,
            batches_per_epoch=args.val_batches,
            seed=args.seed + int(horizon_steps) + 10000,
        )

        test_seq = RandomWindowSequence(
            manifest_df=test_df,
            processed_dir=processed_dir,
            mean=mean,
            std=std,
            input_len=input_len,
            horizon_steps=horizon_steps,
            batch_size=args.batch_size,
            batches_per_epoch=args.test_batches,
            seed=args.seed + int(horizon_steps) + 20000,
        )

        model = build_dense_model(
            input_len=input_len,
            n_features=6,
            learning_rate=args.learning_rate,
        )

        model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
            ),
        ]

        hist = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        for epoch_idx in range(len(hist.history["loss"])):
            row = {
                "model": "dense",
                "horizon_name": horizon_name,
                "horizon_steps": horizon_steps,
                "epoch": epoch_idx + 1,
            }
            for k, v in hist.history.items():
                row[k] = v[epoch_idx]
            history_rows.append(row)

        model_path = out_dir / f"dense_{horizon_name}.keras"
        model.save(model_path)
        print(f"saved model: {model_path}")

        # scaled metric
        val_metrics = model.evaluate(val_seq, verbose=0, return_dict=True)
        test_metrics = model.evaluate(test_seq, verbose=0, return_dict=True)

        # raw MAE
        val_raw_mae = evaluate_generator_raw_mae(model, val_seq, std, args.val_batches)
        test_raw_mae = evaluate_generator_raw_mae(model, test_seq, std, args.test_batches)

        for split_name, metrics, raw_mae in [
            ("val", val_metrics, val_raw_mae),
            ("test", test_metrics, test_raw_mae),
        ]:
            eval_rows.append({
                "model": "dense",
                "horizon_name": horizon_name,
                "horizon_steps": horizon_steps,
                "split": split_name,
                "sensor": "ALL_MEAN",
                "loss_mse_scaled": metrics["loss"],
                "mae_scaled": metrics["mae"],
                "rmse_scaled": metrics["rmse"],
                "mae_raw": float(np.mean(raw_mae)),
            })

            for i, sensor in enumerate(SENSORS):
                eval_rows.append({
                    "model": "dense",
                    "horizon_name": horizon_name,
                    "horizon_steps": horizon_steps,
                    "split": split_name,
                    "sensor": sensor,
                    "loss_mse_scaled": metrics["loss"],
                    "mae_scaled": np.nan,
                    "rmse_scaled": np.nan,
                    "mae_raw": float(raw_mae[i]),
                })

        history_path = out_dir / "dense_training_history.csv"
        eval_path = out_dir / "dense_eval_summary.csv"

        pd.DataFrame(history_rows).to_csv(history_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(eval_rows).to_csv(eval_path, index=False, encoding="utf-8-sig")

        print(f"saved history: {history_path}")
        print(f"saved eval: {eval_path}")

    print()
    print("done.")
    print("review files:")
    print(out_dir / "dense_training_history.csv")
    print(out_dir / "dense_eval_summary.csv")


if __name__ == "__main__":
    main()
