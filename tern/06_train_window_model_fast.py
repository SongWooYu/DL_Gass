#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_train_window_model_fast.py

목적:
- 기존 04_train_dense_baseline.py보다 빠르게 학습한다.
- processed/episodes_npz 데이터를 메모리에 미리 올려서 npz 반복 읽기 병목을 제거한다.
- input window size를 실험 인자로 바꿀 수 있다.
- Dense, 1D CNN, GRU, LSTM 모델을 같은 입출력 정의로 학습할 수 있다.

핵심 입출력:
    X = values[start : start + input_len]
    y = values[start + input_len - 1 + horizon_steps]

출력 shape:
    X: (batch, input_len, 6)
    y: (batch, 6)

사용 예:
    /usr/bin/python 06_train_window_model_fast.py \
      --processed_dir ./processed \
      --assets_dir ./training_assets \
      --out_dir ./experiments_fast \
      --model_type dense \
      --input_len 30 \
      --only_horizon 3s \
      --epochs 5 \
      --train_batches 150 \
      --val_batches 50 \
      --test_batches 50 \
      --batch_size 256

모델 종류:
    --model_type dense
    --model_type cnn1d
    --model_type gru
    --model_type lstm
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


class PreloadedWindowSequence(keras.utils.Sequence):
    """
    episode npz를 매 batch마다 읽지 않는다.
    split에 해당하는 모든 episode values를 메모리에 올려두고,
    미리 만든 (episode_index, start_index) 쌍으로 batch를 생성한다.

    이 데이터는 1000 episode 전체를 올려도 대략 72MB 수준이므로 preload가 맞다.
    """

    def __init__(
        self,
        episode_values,
        mean,
        std,
        input_len,
        horizon_steps,
        batch_size,
        batches_per_epoch,
        seed=42,
        shuffle=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.episode_values = episode_values
        self.mean = mean.astype(np.float32).reshape(1, 6)
        self.std = std.astype(np.float32).reshape(1, 6)
        self.input_len = int(input_len)
        self.horizon_steps = int(horizon_steps)
        self.batch_size = int(batch_size)
        self.batches_per_epoch = int(batches_per_epoch)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)

        if len(self.episode_values) == 0:
            raise ValueError("episode_values is empty")

        self.rows = self.episode_values[0].shape[0]
        self.max_start = self.rows - self.input_len - self.horizon_steps

        if self.max_start < 0:
            raise ValueError(
                f"invalid input_len/horizon_steps: rows={self.rows}, "
                f"input_len={self.input_len}, horizon_steps={self.horizon_steps}"
            )

        self.n_samples = self.batch_size * self.batches_per_epoch
        self.on_epoch_end()

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        self.episode_indices = self.rng.integers(
            0,
            len(self.episode_values),
            size=self.n_samples,
            dtype=np.int32,
        )
        self.start_indices = self.rng.integers(
            0,
            self.max_start + 1,
            size=self.n_samples,
            dtype=np.int32,
        )

        if self.shuffle:
            order = self.rng.permutation(self.n_samples)
            self.episode_indices = self.episode_indices[order]
            self.start_indices = self.start_indices[order]

    def __getitem__(self, batch_idx):
        lo = batch_idx * self.batch_size
        hi = lo + self.batch_size

        ep_idx_batch = self.episode_indices[lo:hi]
        start_batch = self.start_indices[lo:hi]

        x = np.empty((len(ep_idx_batch), self.input_len, 6), dtype=np.float32)
        y = np.empty((len(ep_idx_batch), 6), dtype=np.float32)

        for i, (ep_idx, start) in enumerate(zip(ep_idx_batch, start_batch)):
            values = self.episode_values[int(ep_idx)]
            end = int(start) + self.input_len
            target_idx = end - 1 + self.horizon_steps

            x_raw = values[int(start):end, :]
            y_raw = values[target_idx, :]

            x[i] = (x_raw - self.mean) / self.std
            y[i] = (y_raw - self.mean.reshape(6,)) / self.std.reshape(6,)

        return x, y


def resolve_npz_path(row, processed_dir):
    p = Path(str(row["npz_path"]))
    if p.exists():
        return p

    eid = int(row["episode_id"])
    fallback = processed_dir / "episodes_npz" / f"episode_{eid:05d}.npz"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"npz not found: {p} or {fallback}")


def preload_split_values(split_df, processed_dir):
    values_list = []

    for i, (_, row) in enumerate(split_df.iterrows(), start=1):
        p = resolve_npz_path(row, processed_dir)
        data = np.load(p, allow_pickle=True)
        values = data["values"].astype(np.float32)

        if values.shape != (3000, 6):
            raise ValueError(f"unexpected shape: {p}, {values.shape}")

        values_list.append(values)

        if i % 100 == 0:
            print(f"preloaded {i}/{len(split_df)} episodes")

    return values_list


def build_model(model_type, input_len, learning_rate):
    model_type = model_type.lower()

    if model_type == "dense":
        model = keras.Sequential([
            layers.Input(shape=(input_len, 6)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(6),
        ])

    elif model_type == "cnn1d":
        model = keras.Sequential([
            layers.Input(shape=(input_len, 6)),
            layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
            layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(6),
        ])

    elif model_type == "gru":
        model = keras.Sequential([
            layers.Input(shape=(input_len, 6)),
            layers.GRU(64),
            layers.Dense(32, activation="relu"),
            layers.Dense(6),
        ])

    elif model_type == "lstm":
        model = keras.Sequential([
            layers.Input(shape=(input_len, 6)),
            layers.LSTM(64),
            layers.Dense(32, activation="relu"),
            layers.Dense(6),
        ])

    else:
        raise ValueError(f"unknown model_type: {model_type}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model


def evaluate_raw_mae(model, seq, std, max_batches):
    abs_sum = np.zeros(6, dtype=np.float64)
    n = 0

    for i in range(min(len(seq), max_batches)):
        x, y_true = seq[i]
        y_pred = model.predict(x, verbose=0)

        err_raw = np.abs((y_true - y_pred) * std.reshape(1, 6))
        abs_sum += err_raw.sum(axis=0)
        n += y_true.shape[0]

    return abs_sum / max(n, 1)


def make_prefix(model_type, input_len, horizon_name):
    return f"{model_type}_w{input_len:04d}_{horizon_name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./processed")
    parser.add_argument("--assets_dir", default="./training_assets")
    parser.add_argument("--out_dir", default="./experiments_fast")

    parser.add_argument("--model_type", default="dense", choices=["dense", "cnn1d", "gru", "lstm"])
    parser.add_argument("--input_len", type=int, default=100)
    parser.add_argument("--only_horizon", default="3s")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_batches", type=int, default=150)
    parser.add_argument("--val_batches", type=int, default=50)
    parser.add_argument("--test_batches", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    assets_dir = Path(args.assets_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_manifest_path = processed_dir / "split_manifest.csv"
    config_path = assets_dir / "training_config.json"
    scaler_path = assets_dir / "scaler_stats.npz"

    if not split_manifest_path.exists():
        raise FileNotFoundError(split_manifest_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    horizons = config["horizons"]
    if args.only_horizon not in horizons:
        raise ValueError(f"unknown horizon: {args.only_horizon}")

    horizon_name = args.only_horizon
    horizon_steps = int(horizons[horizon_name])

    scaler = np.load(scaler_path, allow_pickle=True)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    split_manifest = pd.read_csv(split_manifest_path)

    train_df = split_manifest[split_manifest["split"] == "train"].copy()
    val_df = split_manifest[split_manifest["split"] == "val"].copy()
    test_df = split_manifest[split_manifest["split"] == "test"].copy()

    print("TensorFlow:", tf.__version__)
    print("model_type:", args.model_type)
    print("input_len:", args.input_len, "steps =", args.input_len * 0.1, "sec")
    print("horizon:", horizon_name, horizon_steps, "steps =", horizon_steps * 0.1, "sec")
    print("train/val/test episodes:", len(train_df), len(val_df), len(test_df))
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    print("[1/4] preload train episodes")
    train_values = preload_split_values(train_df, processed_dir)
    print("[2/4] preload val episodes")
    val_values = preload_split_values(val_df, processed_dir)
    print("[3/4] preload test episodes")
    test_values = preload_split_values(test_df, processed_dir)

    train_seq = PreloadedWindowSequence(
        episode_values=train_values,
        mean=mean,
        std=std,
        input_len=args.input_len,
        horizon_steps=horizon_steps,
        batch_size=args.batch_size,
        batches_per_epoch=args.train_batches,
        seed=args.seed + horizon_steps + args.input_len,
        shuffle=True,
    )

    val_seq = PreloadedWindowSequence(
        episode_values=val_values,
        mean=mean,
        std=std,
        input_len=args.input_len,
        horizon_steps=horizon_steps,
        batch_size=args.batch_size,
        batches_per_epoch=args.val_batches,
        seed=args.seed + horizon_steps + args.input_len + 10000,
        shuffle=False,
    )

    test_seq = PreloadedWindowSequence(
        episode_values=test_values,
        mean=mean,
        std=std,
        input_len=args.input_len,
        horizon_steps=horizon_steps,
        batch_size=args.batch_size,
        batches_per_epoch=args.test_batches,
        seed=args.seed + horizon_steps + args.input_len + 20000,
        shuffle=False,
    )

    print("[4/4] build and train model")
    model = build_model(args.model_type, args.input_len, args.learning_rate)
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

    prefix = make_prefix(args.model_type, args.input_len, horizon_name)

    model_path = out_dir / f"{prefix}.keras"
    history_path = out_dir / f"{prefix}_history.csv"
    eval_path = out_dir / f"{prefix}_eval.csv"

    model.save(model_path)

    history_rows = []
    for epoch_idx in range(len(hist.history["loss"])):
        row = {
            "model": args.model_type,
            "input_len": args.input_len,
            "input_seconds": args.input_len * 0.1,
            "horizon_name": horizon_name,
            "horizon_steps": horizon_steps,
            "horizon_seconds": horizon_steps * 0.1,
            "epoch": epoch_idx + 1,
        }

        for k, v in hist.history.items():
            row[k] = v[epoch_idx]

        history_rows.append(row)

    pd.DataFrame(history_rows).to_csv(history_path, index=False, encoding="utf-8-sig")

    val_metrics = model.evaluate(val_seq, verbose=0, return_dict=True)
    test_metrics = model.evaluate(test_seq, verbose=0, return_dict=True)

    val_raw_mae = evaluate_raw_mae(model, val_seq, std, args.val_batches)
    test_raw_mae = evaluate_raw_mae(model, test_seq, std, args.test_batches)

    eval_rows = []

    for split_name, metrics, raw_mae in [
        ("val", val_metrics, val_raw_mae),
        ("test", test_metrics, test_raw_mae),
    ]:
        eval_rows.append({
            "model": args.model_type,
            "input_len": args.input_len,
            "input_seconds": args.input_len * 0.1,
            "horizon_name": horizon_name,
            "horizon_steps": horizon_steps,
            "horizon_seconds": horizon_steps * 0.1,
            "split": split_name,
            "sensor": "ALL_MEAN",
            "loss_mse_scaled": metrics["loss"],
            "mae_scaled": metrics["mae"],
            "rmse_scaled": metrics["rmse"],
            "mae_raw": float(np.mean(raw_mae)),
        })

        for i, sensor in enumerate(SENSORS):
            eval_rows.append({
                "model": args.model_type,
                "input_len": args.input_len,
                "input_seconds": args.input_len * 0.1,
                "horizon_name": horizon_name,
                "horizon_steps": horizon_steps,
                "horizon_seconds": horizon_steps * 0.1,
                "split": split_name,
                "sensor": sensor,
                "loss_mse_scaled": metrics["loss"],
                "mae_scaled": np.nan,
                "rmse_scaled": np.nan,
                "mae_raw": float(raw_mae[i]),
            })

    pd.DataFrame(eval_rows).to_csv(eval_path, index=False, encoding="utf-8-sig")

    print("saved:")
    print(model_path)
    print(history_path)
    print(eval_path)


if __name__ == "__main__":
    main()
