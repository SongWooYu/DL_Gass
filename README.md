# Gas Sensor Multivariate Time-Series Forecasting

가스 환경 센서 데이터를 이용해 미래 3초, 9초, 30초, 60초, 120초 후의 센서값을 예측하는 머신러닝 기초 텀프로젝트입니다. 6개 센서의 과거 10초 시계열을 입력으로 사용하고, 지정된 예측 horizon의 6개 센서값을 동시에 출력하는 다변량 시계열 회귀 문제로 구성했습니다.

## 1. 프로젝트 개요

- 과목: 머신러닝기초
- 과제 목표: 가스 환경 예측을 위한 6개 센서 데이터 기반 미래 센서값 예측 모델 학습
- 문제 유형: 지도학습 기반 다변량 시계열 회귀
- 입력: 과거 10초, 100 step, 6개 센서값
- 출력: 미래 특정 horizon의 6개 센서값
- 예측 horizon: 3s, 9s, 30s, 60s, 120s
- 최종 후보 모델: GRU

## 2. 데이터 구성

원본 데이터는 1000개의 독립 episode로 구성됩니다. 각 episode는 300초 길이이며, 100ms 간격으로 기록되어 episode당 3000 step을 가집니다.

| 항목 | 내용 |
|---|---|
| 센서 수 | 6개 |
| 센서 목록 | Accelerometer, GasLeak, Pressure_1, Pressure_2, Temperature_1, Temperature_2 |
| episode 수 | 1000개 |
| episode 길이 | 300초, 3000 step |
| 샘플링 주기 | 0.1초, 100ms |
| 입력 window | 100 step, 10초 |
| 출력 | 미래 horizon 시점의 6개 센서값 |

PNG 파일은 사람이 확인하기 위한 시각화 자료로만 사용했고, 학습에는 CSV에서 읽은 수치 데이터만 사용했습니다. CSV는 `header=None`으로 읽어 실제 3000행 구조를 유지했습니다.

## 3. 전처리 파이프라인

1. 같은 episode 번호를 가진 6개 센서 CSV를 읽습니다.
2. 시간축 기준으로 6개 센서를 병합하여 `(3000, 6)` 형태의 NumPy 배열을 만듭니다.
3. episode별 배열을 `episode_XXXXX.npz` 형식으로 저장합니다.
4. train, validation, test를 row 단위가 아니라 episode 단위로 분리합니다.
5. StandardScaler의 평균과 표준편차는 train split에서만 계산합니다.
6. 학습 window와 target은 같은 episode 내부에서만 생성합니다.

분할 구조는 다음과 같습니다.

| Split | Episode 수 | 용도 |
|---|---:|---|
| Train | 700 | 모델 학습 및 scaler 통계 계산 |
| Validation | 150 | checkpoint 선택, early stopping 판단 |
| Test | 150 | 최종 성능 평가 |

## 4. 모델 구성

동일한 입력 shape `(100, 6)`과 출력 shape `(6,)` 조건에서 여러 모델을 비교했습니다.

| 모델 | 구조 요약 | 파라미터 수 | 실험 목적 |
|---|---|---:|---|
| Dense baseline | Flatten + Dense(128) + Dense(64) + Dense(6) | 85,574 | 파이프라인 검증용 기준 모델 |
| CNN1D | Conv1D(32) + Conv1D(64) + GlobalAveragePooling + Dense | 15,846 | 시간축 지역 패턴 학습 |
| GRU | GRU(64) + Dense(64) + Dense(6) | 18,374 | 최종 후보 모델 |
| LSTM | LSTM(64) + Dense(64) + Dense(6) | 22,726 | 장기 의존성 비교 |
| GRU_REG | GRU + Dropout + L2 | 18,374 | 정규화 추가 실험 |

## 5. 학습 조건

| 항목 | 값 |
|---|---|
| Input shape | `(100, 6)` |
| Output shape | `(6,)` |
| Loss | MSE |
| Metric | MAE |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 128 |
| Epoch | 최대 20 |
| Steps per epoch | 300 |
| Validation steps | 80 |
| Test windows | 5000 |
| Callbacks | ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger |

모델은 scaled 공간에서 학습하고, 최종 평가는 원 센서 단위로 복원한 raw MAE와 scaled MAE를 함께 사용했습니다.

## 6. 주요 결과

GRU가 모든 horizon에서 가장 낮은 test raw MAE를 기록했습니다.

| Horizon | CNN1D | GRU | LSTM | Naive baseline |
|---|---:|---:|---:|---:|
| 3s | 4.3599 | 3.6925 | 3.9287 | 3.9790 |
| 9s | 5.4674 | 4.8655 | 4.9996 | 6.8639 |
| 30s | 8.0875 | 7.2831 | 7.4360 | 18.6660 |
| 60s | 10.4342 | 9.6277 | 9.7289 | 35.9669 |
| 120s | 13.8000 | 13.0141 | 13.8980 | 70.0574 |

GRU의 naive baseline 대비 개선율은 다음과 같습니다.

| Horizon | GRU 개선율 |
|---|---:|
| 3s | 7.20% |
| 9s | 29.11% |
| 30s | 60.98% |
| 60s | 73.23% |
| 120s | 81.42% |

예측 horizon이 길어질수록 naive baseline의 오차는 급격히 증가했지만, GRU는 상대적으로 완만하게 증가했습니다. 따라서 최종 모델 후보는 성능과 파라미터 수의 균형이 가장 우수한 GRU로 선정했습니다.

## 7. 실행 예시

아래 명령은 GRU 모델을 특정 horizon에 대해 학습하는 예시입니다.

```bash
python tern/src/train/07_train_sequence_model.py \
  --processed_dir ./processed \
  --assets_dir ./training_assets \
  --out_dir ./runs_gru_9s \
  --model_type gru \
  --horizon 9s \
  --epochs 20 \
  --batch_size 128 \
  --steps_per_epoch 300 \
  --val_steps 80
```

실험 결과 요약은 다음과 같이 생성할 수 있습니다.

```bash
python tern/src/evaluate/08_summarize_experiments_v2.py \
  --runs_root . \
  --out_dir ./summary_results_v2
```

## 8. 산출물

| 산출물 | 설명 |
|---|---|
| `best_<model>_<horizon>.keras` | validation MAE 기준 best checkpoint |
| `history.csv` | epoch별 학습 및 검증 지표 |
| `result.json` | 모델별 최종 평가 결과 |
| `test_sensor_mae.csv` | 센서별 raw MAE |
| `model_comparison_summary_all.csv` | 전체 실험 요약표 |
| `best_model_by_horizon.csv` | horizon별 최적 모델 |

## 9. 결론

본 프로젝트는 CSV/PNG 구조 확인, episode 단위 병합, train/validation/test 분리, StandardScaler 적용, sliding window 생성, CNN1D/GRU/LSTM 비교, naive baseline 평가까지 포함한 시계열 예측 파이프라인을 구현했습니다. 최종적으로 GRU가 모든 예측 horizon에서 가장 낮은 test raw MAE를 기록했으며, 120초 예측에서는 naive baseline 대비 81.42% 개선을 보였습니다.
