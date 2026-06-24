#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_train_sequence_model.py

목적
- 동일한 데이터 분할, 동일한 scaler, 동일한 평가 방식으로 여러 딥러닝 예측 모델을 학습한다.
- 지원 모델: dense, cnn1d, gru, lstm
- 지원 horizon: 3s, 6s, 9s, 30s, 60s, 120s

입력
- processed/split_manifest.csv
- processed/episodes_npz/episode_XXXXX.npz
- training_assets/scaler_stats.npz
- training_assets/naive_baseline_mae.csv (있으면 자동 비교)

출력
- runs_<model>_<horizon>/best_<model>_<horizon>.keras
- runs_<model>_<horizon>/history.csv
- runs_<model>_<horizon>/result.json
- runs_<model>_<horizon>/test_sensor_mae.csv

실행 예
/usr/bin/python 07_train_sequence_model.py \
  --processed_dir ./processed \
  --assets_dir ./training_assets \
  --out_dir ./runs_gru_3s \
  --model_type gru \
  --horizon 3s \
  --epochs 20 \
  --batch_size 128 \
  --steps_per_epoch 300 \
  --val_steps 80
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
from tensorflow.keras import layers


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
    "9s": 90,
    "30s": 300,
    "60s": 600,
    "120s": 1200,
}


class EpisodeCache:
    def __init__(self, processed_dir: Path, split_name: str):
        self.processed_dir = processed_dir
        self.split_name = split_name
        self.items = []
        self.arrays = []
        self._load()

    def _resolve_npz_path(self, row):
        p = Path(str(row["npz_path"]))
        if p.exists():
            return p
        eid = int(row["episode_id"])
        fallback = self.processed_dir / "episodes_npz" / f"episode_{eid:05d}.npz"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"npz not found: {p} or {fallback}")

    def _load(self):
        manifest_path = self.processed_dir / "split_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = pd.read_csv(manifest_path)
        split_df = manifest[manifest["split"] == self.split_name].copy()
        if split_df.empty:
            raise ValueError(f"empty split: {self.split_name}")

        for _, row in split_df.iterrows():
            path = self._resolve_npz_path(row)
            data = np.load(path, allow_pickle=True)
            values = data["values"].astype(np.float32)
            if values.shape != (3000, 6):
                raise ValueError(f"unexpected shape: {path}, {values.shape}")
            self.items.append({
                "episode_id": int(row["episode_id"]),
                "npz_path": str(path),
                "length": int(values.shape[0]),
            })
            self.arrays.append(values)

    def __len__(self):
        return len(self.arrays)


class RandomWindowSequence(keras.utils.Sequence):
    def __init__(self, cache: EpisodeCache, mean, std, input_len, horizon_steps,
                 batch_size, steps_per_epoch, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.cache = cache
        self.mean = mean.reshape(1, -1).astype(np.float32)
        self.std = std.reshape(1, -1).astype(np.float32)
        self.input_len = int(input_len)
        self.horizon_steps = int(horizon_steps)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.rng = np.random.default_rng(seed)
        self.max_start = 3000 - self.input_len - self.horizon_steps
        if self.max_start < 0:
            raise ValueError("input_len and horizon_steps exceed episode length")

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.input_len, 6), dtype=np.float32)
        y = np.zeros((self.batch_size, 6), dtype=np.float32)
        episode_indices = self.rng.integers(0, len(self.cache), size=self.batch_size)
        start_indices = self.rng.integers(0, self.max_start + 1, size=self.batch_size)

        for b in range(self.batch_size):
            values = self.cache.arrays[int(episode_indices[b])]
            start = int(start_indices[b])
            end = start + self.input_len
            target_idx = end - 1 + self.horizon_steps
            X[b] = (values[start:end, :] - self.mean) / self.std
            y[b] = (values[target_idx, :] - self.mean.reshape(-1)) / self.std.reshape(-1)
        return X, y


class FixedWindowSequence(keras.utils.Sequence):
    def __init__(self, cache: EpisodeCache, mean, std, input_len, horizon_steps,
                 batch_size, max_windows, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.cache = cache
        self.mean = mean.reshape(1, -1).astype(np.float32)
        self.std = std.reshape(1, -1).astype(np.float32)
        self.input_len = int(input_len)
        self.horizon_steps = int(horizon_steps)
        self.batch_size = int(batch_size)
        self.max_windows = int(max_windows)
        self.rng = np.random.default_rng(seed)
        self.pairs = self._make_pairs()

    def _make_pairs(self):
        pairs = []
        max_start = 3000 - self.input_len - self.horizon_steps
        if max_start < 0:
            raise ValueError("input_len and horizon_steps exceed episode length")
        for episode_idx in range(len(self.cache)):
            per_episode = min(50, max_start + 1)
            if per_episode <= 1:
                starts = [0]
            else:
                starts = np.linspace(0, max_start, per_episode).astype(int).tolist()
            for start in starts:
                pairs.append((episode_idx, int(start)))
        if len(pairs) > self.max_windows:
            sampled = self.rng.choice(np.arange(len(pairs)), size=self.max_windows, replace=False)
            pairs = [pairs[int(i)] for i in sampled]
        return pairs

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        batch_pairs = self.pairs[idx * self.batch_size:(idx + 1) * self.batch_size]
        n = len(batch_pairs)
        X = np.zeros((n, self.input_len, 6), dtype=np.float32)
        y = np.zeros((n, 6), dtype=np.float32)
        for b, (episode_idx, start) in enumerate(batch_pairs):
            values = self.cache.arrays[int(episode_idx)]
            end = start + self.input_len
            target_idx = end - 1 + self.horizon_steps
            X[b] = (values[start:end, :] - self.mean) / self.std
            y[b] = (values[target_idx, :] - self.mean.reshape(-1)) / self.std.reshape(-1)
        return X, y


def build_model(model_type: str, input_len: int, n_sensors: int, lr: float):
    model_type = model_type.lower()
    if model_type == "dense":
        model = keras.Sequential([
            layers.Input(shape=(input_len, n_sensors)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_sensors),
        ])
    elif model_type == "cnn1d":
        model = keras.Sequential([
            layers.Input(shape=(input_len, n_sensors)),
            layers.Conv1D(32, kernel_size=5, padding="causal", activation="relu"),
            layers.Conv1D(64, kernel_size=5, padding="causal", activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_sensors),
        ])
    elif model_type == "gru":
        model = keras.Sequential([
            layers.Input(shape=(input_len, n_sensors)),
            layers.GRU(64, return_sequences=False),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_sensors),
        ])
    elif model_type == "lstm":
        model = keras.Sequential([
            layers.Input(shape=(input_len, n_sensors)),
            layers.LSTM(64, return_sequences=False),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_sensors),
        ])
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def evaluate_raw_mae(model, seq, mean, std):
    abs_sum = np.zeros(6, dtype=np.float64)
    n_total = 0
    for i in range(len(seq)):
        X, y_scaled = seq[i]
        pred_scaled = model.predict(X, verbose=0)
        y_true = y_scaled * std.reshape(1, -1) + mean.reshape(1, -1)
        y_pred = pred_scaled * std.reshape(1, -1) + mean.reshape(1, -1)
        abs_sum += np.abs(y_pred - y_true).sum(axis=0)
        n_total += y_true.shape[0]
    mae_per_sensor = abs_sum / max(n_total, 1)
    return float(mae_per_sensor.mean()), mae_per_sensor


def load_naive_comparison(assets_dir: Path, horizon: str):
    p = assets_dir / "naive_baseline_mae.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    row = df[(df["horizon_name"] == horizon) & (df["split"] == "test") & (df["sensor"] == "ALL_MEAN")]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"test_naive_mae_raw": float(r["mae_raw"]), "test_naive_mae_scaled": float(r["mae_scaled"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./processed")
    parser.add_argument("--assets_dir", default="./training_assets")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_type", choices=["dense", "cnn1d", "gru", "lstm"], required=True)
    parser.add_argument("--horizon", choices=list(HORIZONS.keys()), required=True)
    parser.add_argument("--input_len", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=300)
    parser.add_argument("--val_steps", type=int, default=80)
    parser.add_argument("--test_windows", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

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
    print(f"model_type      : {args.model_type}")
    print(f"horizon         : {args.horizon} ({horizon_steps} steps)")
    print(f"input_len       : {args.input_len}")
    print(f"batch_size      : {args.batch_size}")
    print(f"epochs          : {args.epochs}")
    print(f"steps_per_epoch : {args.steps_per_epoch}")
    print(f"val_steps       : {args.val_steps}")
    print(f"processed_dir   : {processed_dir}")
    print(f"assets_dir      : {assets_dir}")
    print(f"out_dir         : {out_dir}")

    print("\n[1/6] load cached episode arrays")
    train_cache = EpisodeCache(processed_dir, "train")
    val_cache = EpisodeCache(processed_dir, "val")
    test_cache = EpisodeCache(processed_dir, "test")
    print(f"train/val/test episodes: {len(train_cache)} / {len(val_cache)} / {len(test_cache)}")

    print("\n[2/6] build sequences")
    train_seq = RandomWindowSequence(train_cache, mean, std, args.input_len, horizon_steps,
                                     args.batch_size, args.steps_per_epoch, seed=args.seed)
    val_seq = RandomWindowSequence(val_cache, mean, std, args.input_len, horizon_steps,
                                   args.batch_size, args.val_steps, seed=args.seed + 1)
    test_seq = FixedWindowSequence(test_cache, mean, std, args.input_len, horizon_steps,
                                   args.batch_size, args.test_windows, seed=args.seed + 2)

    print("\n[3/6] build model")
    model = build_model(args.model_type, args.input_len, 6, args.learning_rate)
    model.summary()

    model_path = out_dir / f"best_{args.model_type}_{args.horizon}.keras"
    history_path = out_dir / "history.csv"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_mae", mode="min", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(str(history_path)),
        keras.callbacks.ReduceLROnPlateau(monitor="val_mae", mode="min", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    print("\n[4/6] train")
    model.fit(train_seq, validation_data=val_seq, epochs=args.epochs, callbacks=callbacks, verbose=1)

    print("\n[5/6] evaluate scaled")
    scaled_eval = model.evaluate(test_seq, verbose=1, return_dict=True)

    print("\n[6/6] evaluate raw MAE")
    raw_mae_mean, raw_mae_per_sensor = evaluate_raw_mae(model, test_seq, mean, std)

    sensor_rows = []
    for sensor, v in zip(SENSORS, raw_mae_per_sensor):
        sensor_rows.append({"model_type": args.model_type, "horizon": args.horizon, "sensor": sensor, "test_mae_raw": float(v)})
    sensor_rows.append({"model_type": args.model_type, "horizon": args.horizon, "sensor": "ALL_MEAN", "test_mae_raw": float(raw_mae_mean)})
    sensor_csv = out_dir / "test_sensor_mae.csv"
    pd.DataFrame(sensor_rows).to_csv(sensor_csv, index=False, encoding="utf-8-sig")

    naive = load_naive_comparison(assets_dir, args.horizon)
    result = {
        "model_type": args.model_type,
        "horizon": args.horizon,
        "horizon_steps": horizon_steps,
        "input_len": args.input_len,
        "input_seconds": args.input_len * 0.1,
        "test_scaled_metrics": {k: float(v) for k, v in scaled_eval.items()},
        "test_raw_mae_mean": float(raw_mae_mean),
        "test_raw_mae_per_sensor": {s: float(v) for s, v in zip(SENSORS, raw_mae_per_sensor)},
        "naive_comparison": naive,
        "model_path": str(model_path),
        "history_csv": str(history_path),
        "sensor_mae_csv": str(sensor_csv),
    }
    if naive is not None:
        result["delta_vs_naive_raw"] = float(raw_mae_mean - naive["test_naive_mae_raw"])
        result["improvement_vs_naive_raw_percent"] = float((naive["test_naive_mae_raw"] - raw_mae_mean) / naive["test_naive_mae_raw"] * 100.0)

    result_path = out_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nresult")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nsaved files")
    print(model_path)
    print(history_path)
    print(sensor_csv)
    print(result_path)


if __name__ == "__main__":
    main()
