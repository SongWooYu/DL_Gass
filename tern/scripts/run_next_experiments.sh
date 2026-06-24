#!/usr/bin/env bash
set -e

PY=/usr/bin/python

# 1) 빠른 검증: 3초 예측에서 CNN/GRU/LSTM을 같은 조건으로 비교
$PY 07_train_sequence_model.py --processed_dir ./processed --assets_dir ./training_assets --out_dir ./runs_cnn1d_3s --model_type cnn1d --horizon 3s --epochs 20 --batch_size 128 --steps_per_epoch 300 --val_steps 80
$PY 07_train_sequence_model.py --processed_dir ./processed --assets_dir ./training_assets --out_dir ./runs_gru_3s   --model_type gru   --horizon 3s --epochs 20 --batch_size 128 --steps_per_epoch 300 --val_steps 80

# 2) 결과 요약
$PY 08_summarize_experiments.py --runs_root . --assets_dir ./training_assets --out_dir ./summary_results
