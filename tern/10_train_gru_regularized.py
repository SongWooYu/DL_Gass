#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_train_gru_regularized.py

Purpose:
- Train a regularized GRU model for multivariate time-series forecasting.
- This is intended as a follow-up experiment after basic CNN1D / GRU / LSTM comparison.
- Recommended first targets: 60s and 120s, where train/validation gap is larger.

Example:
    /usr/bin/python 10_train_gru_regularized.py \
      --processed_dir ./processed \
      --assets_dir ./training_assets \
      --out_dir ./runs_gru_reg_60s \
      --horizon 60s \
      --epochs 40 \
      --batch_size 128 \
      --steps_per_epoch 300 \
      --val_steps 80 \
      --gru_units 64 \
      --dropout 0.15 \
      --dense_dropout 0.20 \
      --l2 0.0001
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


SENSORS = [
    "Accelerometer",
    "GasLeak",
    "Pressure_1",
    "Pressure_2",
    "Temperature_1",
    "Temperature_2",
]

HORIZONS = {
    "3s": 30,
    "6s": 60,
    "30s": 300,
    "60s": 600,
    "120s": 1200,
}


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_npz_path(row, processed_dir: Path) -> Path:
    p = Path(str(row["npz_path"]))
    if p.exists():
        return p

    eid = int(row["episode_id"])
    fallback = processed_dir / "episodes_npz" / f"episode_{eid:05d}.npz"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"npz not found: {p} or {fallback}")


def load_episode_index(processed_dir: Path, split_name: str):
    manifest_path = processed_dir / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    df = pd.read_csv(manifest_path)
    df = df[df["split"] == split_name].copy()
    if df.empty:
        raise ValueError(f"empty split: {split_name}")

    items = []
    for _, row in df.iterrows():
        npz_path = resolve_npz_path(row, processed_dir)
        episode_id = int(row["episode_id"])

        data = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        values = data["values"]

        items.append({
            "episode_id": episode_id,
            "npz_path": str(npz_path),
            "length": int(values.shape[0]),
        })

    return items


class RandomWindowSequence(keras.utils.Sequence):
    def __init__(
        self,
        episodes,
        mean,
        std,
        input_len,
        horizon_steps,
        batch_size,
        steps_per_epoch,
        seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.episodes = episodes
        self.mean = mean.reshape(1, -1).astype(np.float32)
        self.std = std.reshape(1, -1).astype(np.float32)
        self.input_len = int(input_len)
        self.horizon_steps = int(horizon_steps)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.rng = np.random.default_rng(seed)

        self.valid_episodes = []
        for ep in episodes:
            max_start = ep["length"] - self.input_len - self.horizon_steps
            if max_start > 0:
                item = dict(ep)
                item["max_start"] = int(max_start)
                self.valid_episodes.append(item)

        if not self.valid_episodes:
            raise ValueError("no valid episodes for this horizon")

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.input_len, len(SENSORS)), dtype=np.float32)
        y = np.zeros((self.batch_size, len(SENSORS)), dtype=np.float32)

        for b in range(self.batch_size):
            ep = self.valid_episodes[self.rng.integers(0, len(self.valid_episodes))]
            data = np.load(ep["npz_path"], allow_pickle=True, mmap_mode="r")
            values = data["values"].astype(np.float32)

            start = int(self.rng.integers(0, ep["max_start"] + 1))
            end = start + self.input_len
            target_idx = end - 1 + self.horizon_steps

            x_raw = values[start:end]
            y_raw = values[target_idx]

            X[b] = (x_raw - self.mean) / self.std
            y[b] = (y_raw - self.mean.reshape(-1)) / self.std.reshape(-1)

        return X, y


class FixedWindowSequence(keras.utils.Sequence):
    def __init__(
        self,
        episodes,
        mean,
        std,
        input_len,
        horizon_steps,
        batch_size,
        max_windows,
        seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mean = mean.reshape(1, -1).astype(np.float32)
        self.std = std.reshape(1, -1).astype(np.float32)
        self.input_len = int(input_len)
        self.horizon_steps = int(horizon_steps)
        self.batch_size = int(batch_size)
        self.max_windows = int(max_windows)
        self.rng = np.random.default_rng(seed)

        pairs = []
        for ep in episodes:
            max_start = ep["length"] - self.input_len - self.horizon_steps
            if max_start <= 0:
                continue

            per_episode = min(50, max_start + 1)
            if per_episode <= 1:
                starts = [0]
            else:
                starts = np.linspace(0, max_start, per_episode).astype(int).tolist()

            for s in starts:
                pairs.append((ep, int(s)))

        if len(pairs) > self.max_windows:
            idx = self.rng.choice(np.arange(len(pairs)), size=self.max_windows, replace=False)
            pairs = [pairs[i] for i in idx]

        self.pairs = pairs

        if not self.pairs:
            raise ValueError("no fixed windows")

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        batch_pairs = self.pairs[idx * self.batch_size : (idx + 1) * self.batch_size]
        n = len(batch_pairs)

        X = np.zeros((n, self.input_len, len(SENSORS)), dtype=np.float32)
        y = np.zeros((n, len(SENSORS)), dtype=np.float32)

        for b, (ep, start) in enumerate(batch_pairs):
            data = np.load(ep["npz_path"], allow_pickle=True, mmap_mode="r")
            values = data["values"].astype(np.float32)

            end = start + self.input_len
            target_idx = end - 1 + self.horizon_steps

            x_raw = values[start:end]
            y_raw = values[target_idx]

            X[b] = (x_raw - self.mean) / self.std
            y[b] = (y_raw - self.mean.reshape(-1)) / self.std.reshape(-1)

        return X, y


def build_model(input_len, n_sensors, gru_units, dropout, dense_dropout, l2_value):
    reg = regularizers.l2(l2_value) if l2_value and l2_value > 0 else None

    inputs = keras.Input(shape=(input_len, n_sensors))
    x = layers.GRU(
        gru_units,
        dropout=dropout,
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        bias_regularizer=None,
        name="gru_regularized",
    )(inputs)

    if dense_dropout > 0:
        x = layers.Dropout(dense_dropout, name="dropout_after_gru")(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_regularized",
    )(x)

    if dense_dropout > 0:
        x = layers.Dropout(dense_dropout, name="dropout_after_dense")(x)

    outputs = layers.Dense(n_sensors, name="output")(x)

    model = keras.Model(inputs, outputs, name="gru_regularized_forecaster")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    return model


def load_naive_baseline(assets_dir: Path, horizon: str):
    candidates = [
        assets_dir / "naive_baseline_mae.csv",
        assets_dir / "naive_baseline_summary.csv",
    ]

    for p in candidates:
        if not p.exists():
            continue

        df = pd.read_csv(p)
        horizon_cols = [c for c in df.columns if c.lower() in ["horizon", "horizon_name"]]
        if not horizon_cols:
            continue

        hcol = horizon_cols[0]
        row = df[df[hcol].astype(str) == horizon]
        if row.empty:
            continue

        raw_cols = [c for c in row.columns if "ALL_MEAN" in c and "raw" in c.lower()]
        scaled_cols = [c for c in row.columns if "ALL_MEAN" in c and "scaled" in c.lower()]

        raw = float(row.iloc[0][raw_cols[0]]) if raw_cols else None
        scaled = float(row.iloc[0][scaled_cols[0]]) if scaled_cols else None
        return raw, scaled

    return None, None


def evaluate_raw_mae(model, seq, mean, std):
    y_true_all = []
    y_pred_all = []

    for i in range(len(seq)):
        X, y_scaled = seq[i]
        pred_scaled = model.predict(X, verbose=0)

        y_true = y_scaled * std.reshape(1, -1) + mean.reshape(1, -1)
        y_pred = pred_scaled * std.reshape(1, -1) + mean.reshape(1, -1)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    mae_per_sensor = np.mean(np.abs(y_pred_all - y_true_all), axis=0)
    mae_mean = float(np.mean(mae_per_sensor))
    return mae_mean, mae_per_sensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./processed")
    parser.add_argument("--assets_dir", default="./training_assets")
    parser.add_argument("--out_dir", default="./runs_gru_reg_60s")
    parser.add_argument("--horizon", default="60s", choices=list(HORIZONS.keys()))
    parser.add_argument("--input_len", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=300)
    parser.add_argument("--val_steps", type=int, default=80)
    parser.add_argument("--test_windows", type=int, default=5000)
    parser.add_argument("--gru_units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--dense_dropout", type=float, default=0.20)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    processed_dir = Path(args.processed_dir).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = assets_dir / "scaler_stats.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)

    scaler = np.load(scaler_path, allow_pickle=True)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    horizon_steps = HORIZONS[args.horizon]

    print("config")
    print(f"model_type      : gru_reg")
    print(f"horizon         : {args.horizon} ({horizon_steps} steps)")
    print(f"input_len       : {args.input_len}")
    print(f"epochs          : {args.epochs}")
    print(f"batch_size      : {args.batch_size}")
    print(f"steps_per_epoch : {args.steps_per_epoch}")
    print(f"val_steps       : {args.val_steps}")
    print(f"gru_units       : {args.gru_units}")
    print(f"dropout         : {args.dropout}")
    print(f"dense_dropout   : {args.dense_dropout}")
    print(f"l2              : {args.l2}")

    print()
    print("[1/6] load episode index")
    train_eps = load_episode_index(processed_dir, "train")
    val_eps = load_episode_index(processed_dir, "val")
    test_eps = load_episode_index(processed_dir, "test")
    print(f"train/val/test episodes: {len(train_eps)} / {len(val_eps)} / {len(test_eps)}")

    print()
    print("[2/6] build sequences")
    train_seq = RandomWindowSequence(
        train_eps, mean, std, args.input_len, horizon_steps,
        args.batch_size, args.steps_per_epoch, seed=args.seed,
    )
    val_seq = RandomWindowSequence(
        val_eps, mean, std, args.input_len, horizon_steps,
        args.batch_size, args.val_steps, seed=args.seed + 1,
    )
    test_seq = FixedWindowSequence(
        test_eps, mean, std, args.input_len, horizon_steps,
        args.batch_size, args.test_windows, seed=args.seed + 2,
    )

    print()
    print("[3/6] build model")
    model = build_model(
        input_len=args.input_len,
        n_sensors=len(SENSORS),
        gru_units=args.gru_units,
        dropout=args.dropout,
        dense_dropout=args.dense_dropout,
        l2_value=args.l2,
    )
    model.summary()

    model_path = out_dir / f"best_gru_reg_{args.horizon}.keras"
    history_csv = out_dir / "history.csv"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_mae",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=8,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(history_csv)),
    ]

    print()
    print("[4/6] train")
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print()
    print("[5/6] evaluate scaled")
    scaled_eval = model.evaluate(test_seq, verbose=1)
    scaled_metrics = dict(zip(model.metrics_names, [float(x) for x in scaled_eval]))

    print()
    print("[6/6] evaluate raw MAE")
    raw_mae_mean, raw_mae_per_sensor = evaluate_raw_mae(model, test_seq, mean, std)

    sensor_mae = pd.DataFrame({
        "sensor": SENSORS,
        "mae_raw": raw_mae_per_sensor,
    })
    sensor_mae_csv = out_dir / "test_sensor_mae.csv"
    sensor_mae.to_csv(sensor_mae_csv, index=False, encoding="utf-8-sig")

    naive_raw, naive_scaled = load_naive_baseline(assets_dir, args.horizon)
    if naive_raw is not None:
        delta = raw_mae_mean - naive_raw
        improvement = (naive_raw - raw_mae_mean) / naive_raw * 100.0
    else:
        delta = None
        improvement = None

    result = {
        "model_type": "gru_reg",
        "horizon": args.horizon,
        "horizon_steps": horizon_steps,
        "input_len": args.input_len,
        "input_seconds": args.input_len * 0.1,
        "regularization": {
            "gru_units": args.gru_units,
            "dropout": args.dropout,
            "dense_dropout": args.dense_dropout,
            "l2": args.l2,
        },
        "test_scaled_metrics": scaled_metrics,
        "test_raw_mae_mean": raw_mae_mean,
        "test_raw_mae_per_sensor": {
            sensor: float(v) for sensor, v in zip(SENSORS, raw_mae_per_sensor)
        },
        "naive_comparison": {
            "test_naive_mae_raw": naive_raw,
            "test_naive_mae_scaled": naive_scaled,
        },
        "delta_vs_naive_raw": delta,
        "improvement_vs_naive_raw_percent": improvement,
        "model_path": str(model_path),
        "history_csv": str(history_csv),
        "sensor_mae_csv": str(sensor_mae_csv),
    }

    result_path = out_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print("result")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    print("saved files")
    print(model_path)
    print(history_csv)
    print(sensor_mae_csv)
    print(result_path)


if __name__ == "__main__":
    main()
