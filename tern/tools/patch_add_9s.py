#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

files = [
    "src/train/07_train_sequence_model.py",
    "src/train/10_train_gru_regularized.py",
    "src/eda/05_additional_eda.py",
    "src/evaluate/08_summarize_experiments.py",
    "src/evaluate/08_summarize_experiments_v2.py",
    "src/evaluate/09_plot_training_curves.py",
    "src/report/11_make_report_artifacts.py",
]

replacements = [
    # HORIZONS dict: 6s를 유지하면서 9s 추가
    (
        '"3s": 30,\n    "6s": 60,\n    "30s": 300,',
        '"3s": 30,\n    "6s": 60,\n    "9s": 90,\n    "30s": 300,'
    ),

    # HORIZON_ORDER: 최종 보고서 기준은 3s, 9s, 30s, 60s, 120s
    (
        '"3s": 0,\n    "6s": 1,\n    "30s": 2,\n    "60s": 3,\n    "120s": 4,',
        '"3s": 0,\n    "9s": 1,\n    "30s": 2,\n    "60s": 3,\n    "120s": 4,\n    "6s": 99,'
    ),

    # training curve parser
    (
        'for horizon in ["120s", "60s", "30s", "6s", "3s"]:',
        'for horizon in ["120s", "60s", "30s", "9s", "6s", "3s"]:'
    ),

    # 보고서 체크리스트 문구
    (
        '3s/6s/30s/60s/120s 예측',
        '3s/9s/30s/60s/120s 예측'
    ),
    (
        '5개 horizon 전체 실험 완료',
        '최종 요구 5개 horizon 전체 실험 완료'
    ),

    # 주석/문서 문자열
    (
        '지원 horizon: 3s, 6s, 30s, 60s, 120s',
        '지원 horizon: 3s, 6s, 9s, 30s, 60s, 120s'
    ),
]

for rel in files:
    path = Path(rel)
    if not path.exists():
        print(f"[SKIP] not found: {path}")
        continue

    text = path.read_text(encoding="utf-8")
    old_text = text

    for old, new in replacements:
        text = text.replace(old, new)

    if text != old_text:
        path.write_text(text, encoding="utf-8")
        print(f"[PATCHED] {path}")
    else:
        print(f"[NO CHANGE] {path}")

print("done")
