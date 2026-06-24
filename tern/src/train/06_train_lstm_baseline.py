#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_train_lstm_baseline.py

목적:
- processed/episodes_npz/*.npz를 사용해 다변량 시계열 예측 LSTM baseline을 학습한다.
- 입력 X: 과거 100 step, 즉 10초 구간의 6개 센서값
- 출력 y: horizon 뒤의 6개 센서값
- scaler는 training_assets/scaler_stats.npz를 사용한다.
- train/val/test split은 processed/split_manifest.csv를 사용한다.

실행 예:
    /usr/bin/python 06_train_lstm_baseline.py \
      --processed_dir ./processed \
      --assets_dir ./training_assets \
      --out_dir ./runs_lstm_3s \
      --horizon 3s \
      --epochs 20 \
      --batch_size 128 \
      --steps_per_epoch 300 \
      --val_steps 80

주의:
- 전체 window를 전부 메모리에 올리지 않고, batch마다 window를 샘플링한다.
- 학습 속도와 안정성을 위해 우선 baseline 구조로 시작한다.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# TensorFlow 로그를 줄인다.
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

        # mmap_mode='r'로 전체를 한 번에 메모리에 올리지 않는다.
        data = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        values = data["values"]

        items.append({
            "episode_id": episode_id,
            "npz_path": str(npz_path),
            "length": int(values.shape[0]),
        })

    return items


class RandomWindowSequence(keras.utils.Sequence):
    """
    split에 속한 episode들에서 랜덤 window를 뽑아 batch를 만든다.
    """

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
        shuffle=True,
    ):
        self.episodes = episodes
        self.mean = mean.reshape(1, -1).astype(np.float32)
        self.std = std.reshape(1, -1).astype(np.float32)
        self.input_len = int(input_len)
        self.horizon_steps = int(horizon_steps)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.shuffle = shuffle
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
    """
    val/test 평가용.
    episode별 일정 간격으로 window를 고정 추출한다.
    """

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
    ):
        self.episodes = episodes
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

            # 한 episode에서 너무 많이 뽑지 않도록 제한
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


def build_lstm_model(input_len: int, n_sensors: int):
    model = keras.Sequential([
        layers.Input(shape=(input_len, n_sensors)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_sensors),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    return model


def evaluate_raw_mae(model, seq, mean, std):
    """
    scaled 예측값을 원래 단위로 되돌려 센서별 MAE를 계산한다.
    """
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
    parser.add_argument("--out_dir", default="./runs_lstm_3s")
    parser.add_argument("--horizon", default="3s", choices=list(HORIZONS.keys()))
    parser.add_argument("--input_len", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=300)
    parser.add_argument("--val_steps", type=int, default=80)
    parser.add_argument("--test_windows", type=int, default=5000)
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
    print(f"processed_dir     : {processed_dir}")
    print(f"assets_dir        : {assets_dir}")
    print(f"out_dir           : {out_dir}")
    print(f"horizon           : {args.horizon} ({horizon_steps} steps)")
    print(f"input_len         : {args.input_len}")
    print(f"batch_size        : {args.batch_size}")
    print(f"steps_per_epoch   : {args.steps_per_epoch}")
    print(f"epochs            : {args.epochs}")

    print()
    print("[1/6] load split episode index")
    train_eps = load_episode_index(processed_dir, "train")
    val_eps = load_episode_index(processed_dir, "val")
    test_eps = load_episode_index(processed_dir, "test")

    print(f"train episodes: {len(train_eps)}")
    print(f"val episodes  : {len(val_eps)}")
    print(f"test episodes : {len(test_eps)}")

    print()
    print("[2/6] build data sequences")
    train_seq = RandomWindowSequence(
        episodes=train_eps,
        mean=mean,
        std=std,
        input_len=args.input_len,
        horizon_steps=horizon_steps,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )

    val_seq = RandomWindowSequence(
        episodes=val_eps,
        mean=mean,
        std=std,
        input_len=args.input_len,
        horizon_steps=horizon_steps,
        batch_size=args.batch_size,
        steps_per_epoch=args.val_steps,
        seed=args.seed + 1,
    )

    test_seq = FixedWindowSequence(
        episodes=test_eps,
        mean=mean,
        std=std,
        input_len=args.input_len,
        horizon_steps=horizon_steps,
        batch_size=args.batch_size,
        max_windows=args.test_windows,
        seed=args.seed + 2,
    )

    print()
    print("[3/6] build model")
    model = build_lstm_model(args.input_len, len(SENSORS))
    model.summary()

    model_path = out_dir / f"best_lstm_{args.horizon}.keras"
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
            patience=5,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(history_csv)),
    ]

    print()
    print("[4/6] train")
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print()
    print("[5/6] evaluate scaled")
    scaled_eval = model.evaluate(test_seq, verbose=1)
    print(dict(zip(model.metrics_names, scaled_eval)))

    print()
    print("[6/6] evaluate raw unit MAE")
    raw_mae_mean, raw_mae_per_sensor = evaluate_raw_mae(model, test_seq, mean, std)

    result = {
        "horizon": args.horizon,
        "horizon_steps": horizon_steps,
        "input_len": args.input_len,
        "input_seconds": args.input_len * 0.1,
        "test_scaled_metrics": dict(zip(model.metrics_names, [float(x) for x in scaled_eval])),
        "test_raw_mae_mean": raw_mae_mean,
        "test_raw_mae_per_sensor": {
            sensor: float(v) for sensor, v in zip(SENSORS, raw_mae_per_sensor)
        },
        "sensors": SENSORS,
        "model_path": str(model_path),
    }

    result_path = out_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print("result")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    print(f"saved model : {model_path}")
    print(f"saved history: {history_csv}")
    print(f"saved result : {result_path}")


if __name__ == "__main__":
    main()
