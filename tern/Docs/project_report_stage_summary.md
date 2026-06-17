# 도시가스/보일러 센서 시계열 예측 프로젝트 정리

## 1. 보고서 항목 도출

과제 보고서의 큰 항목은 다음 5개로 정리한다.

1. 데이터셋 구성
2. 전처리 방법
3. 모델 구성
4. 학습 방법
5. 성능 평가

현재까지의 작업은 이 5개 항목 중에서 **데이터셋 구성**, **전처리 방법**, **기준 성능 평가**, **Dense baseline 학습 파이프라인 검증**까지 진행한 상태이다.

---

## 2. 프로젝트 문제 정의

이 프로젝트는 6개 센서의 과거 시계열 데이터를 입력으로 사용하여 미래 시점의 센서값을 예측하는 문제이다.

- 입력: 과거 10초 동안의 6개 센서값
- 출력: 특정 미래 시점의 6개 센서값
- 예측 시점: 3초, 6초, 30초, 60초, 120초 후
- 문제 유형: 지도학습 회귀 문제
- 데이터 유형: 다변량 시계열 데이터
- 모델 유형: 딥러닝 기반 시계열 예측 모델

즉, 이 프로젝트는 분류 문제가 아니라 **연속적인 센서값을 예측하는 회귀 문제**이다.

---

## 3. 이론적 배경과 수업 내용 연결

### 3.1 인공지능플랫폼 실습의 학습 목표와 연결

수업 자료에서는 인공지능 모델 학습을 위해 다음 역량을 다룬다.

- 목적에 맞는 데이터 선별
- 데이터 가공 및 정제
- 인공지능 모델 학습
- 성능 평가

본 프로젝트의 전체 흐름도 이 구조를 따른다.

| 수업 내용 | 프로젝트 적용 |
|---|---|
| 데이터 선별 | 원본 센서 CSV와 PNG 구조 확인 |
| 데이터 가공/정제 | 6개 센서 CSV를 episode 단위로 병합 |
| 모델 학습 | Dense baseline, 이후 CNN/GRU/LSTM 학습 |
| 성능 평가 | MAE, RMSE, naive baseline 비교 |

---

### 3.2 머신러닝/딥러닝 분류

본 프로젝트는 다음과 같이 분류된다.

| 기준 | 해당 내용 |
|---|---|
| 학습 방식 | 지도학습 |
| 예측 대상 | 연속값 |
| 문제 유형 | 회귀 |
| 데이터 구조 | 다변량 시계열 |
| 모델 계열 | 딥러닝 |
| 현재 모델 | Dense baseline |
| 이후 모델 | 1D CNN, GRU, LSTM |

K-NN, SVM, PCA, 군집화 등도 수업에서 배운 내용이지만, 현재 프로젝트의 핵심 문제는 시계열 센서값 예측이므로 직접적인 중심 모델로 쓰지는 않는다.

---

### 3.3 시계열 예측과의 연결

수업 자료의 시계열 딥러닝 내용에서는 일정 시간 간격으로 관측된 데이터를 사용해 미래값을 예측하는 forecasting 문제를 다룬다.

본 프로젝트의 센서 데이터도 0.1초 간격으로 측정된 시계열 데이터이다.

따라서 이 프로젝트는 다음 구조의 시계열 예측 문제로 정의할 수 있다.

```text
과거 일정 구간의 센서값 -> 미래 특정 시점의 센서값
```

---

### 3.4 회귀 문제와의 연결

회귀는 입력 데이터에 대응되는 연속적인 출력값을 예측하는 지도학습 문제이다.

본 프로젝트에서는 다음과 같이 입력과 출력이 정의된다.

```text
X = 과거 10초 동안의 6개 센서값
y = 미래 특정 시점의 6개 센서값
```

정답이 범주가 아니라 숫자이므로 classification이 아니라 regression이다.

---

### 3.5 피처 스케일링과 StandardScaler

센서마다 값의 범위가 다르기 때문에 스케일링이 필요하다.

예를 들어 다음처럼 센서별 범위가 크게 다르다.

| 센서 | 값 범위 특성 |
|---|---|
| Accelerometer | 약 0 ~ 0.1 |
| GasLeak | 약 0 ~ 1007 |
| Pressure_1 | 약 0 ~ 3.25 |
| Pressure_2 | 약 0 ~ 3.25 |
| Temperature_1 | 약 0 ~ 110 |
| Temperature_2 | 약 0 ~ 110 |

값의 범위가 다른 상태에서 모델을 학습하면 GasLeak처럼 큰 값을 가진 센서가 손실 함수에 더 큰 영향을 줄 수 있다. 따라서 센서별 평균과 표준편차를 이용해 표준화하였다.

중요한 점은 scaler를 전체 데이터가 아니라 **train 데이터로만 fit**했다는 점이다. validation/test 데이터까지 포함해 평균과 표준편차를 계산하면 평가 데이터 정보가 학습 과정에 들어가는 data leakage가 발생할 수 있다.

---

## 4. 데이터셋 구성

### 4.1 원본 데이터 구조

원본 데이터는 1000개의 episode로 구성되어 있다.

```text
episode_id: 00000 ~ 00999
총 episode 수: 1000
```

각 episode는 6개 센서의 CSV 파일로 구성된다.

센서 목록은 다음과 같다.

| 센서명 | 의미 |
|---|---|
| Accelerometer | 가속도 또는 진동 관련 센서 |
| GasLeak | 가스 누출 관련 센서 |
| Pressure_1 | 압력 센서 1 |
| Pressure_2 | 압력 센서 2 |
| Temperature_1 | 온도 센서 1 |
| Temperature_2 | 온도 센서 2 |

각 센서 CSV는 다음 구조를 갖는다.

```text
행 수: 3000
열 수: 2
1열: step index
2열: sensor value
```

시간 구조는 다음과 같다.

```text
1 step = 0.1초
3000 step = 약 300초
index 0 = 0.0초
index 2999 = 299.9초
```

---

### 4.2 episode의 의미

본 프로젝트에서 가장 중요한 데이터 해석은 다음이다.

```text
같은 episode_id를 가진 6개 센서 파일은 하나의 독립적인 상황이다.
서로 다른 episode_id는 시간적으로 이어진 데이터가 아니다.
```

따라서 `episode_00000`과 `episode_00001`을 이어 붙여 하나의 긴 시계열처럼 처리하면 안 된다.

이 원칙 때문에 sliding window도 반드시 episode 내부에서만 생성한다.

---

### 4.3 데이터 split

데이터는 episode 단위로 train/validation/test로 나누었다.

| split | episode 수 | 비율 |
|---|---:|---:|
| train | 700 | 70% |
| validation | 150 | 15% |
| test | 150 | 15% |

episode 단위로 나눈 이유는 같은 episode에서 나온 window가 train과 test에 동시에 들어가면 데이터 누수가 발생할 수 있기 때문이다.

---

## 5. 탐색적 데이터 분석 결과

EDA를 통해 다음 항목을 확인하였다.

### 5.1 파일 구조 검증

- 센서별 CSV 파일 수: 1000개
- 센서별 PNG 파일 수: 1000개
- 전체 CSV 수: 6000개
- 전체 PNG 수: 6000개
- 총 파일 수: 12000개

### 5.2 CSV 구조 검증

- 모든 CSV는 3000행
- 모든 CSV는 2열
- step index는 0부터 2999까지 존재
- step 간격은 1
- 결측치 없음
- CSV 파싱 오류 없음

### 5.3 episode 구조 검증

- 총 episode 수: 1000
- 각 episode는 6개 센서 CSV로 구성
- 모든 episode가 정상적으로 구성됨
- 누락 센서 없음

### 5.4 센서별 값 범위

| 센서 | file_count | rows_min | rows_max | global min | global max | 평균값 대략 |
|---|---:|---:|---:|---:|---:|---:|
| Accelerometer | 1000 | 3000 | 3000 | 0.0 | 0.10 | 0.0476 |
| GasLeak | 1000 | 3000 | 3000 | 0.0 | 1007.00 | 498.02 |
| Pressure_1 | 1000 | 3000 | 3000 | 0.0 | 3.25 | 1.59 |
| Pressure_2 | 1000 | 3000 | 3000 | 0.0 | 3.25 | 1.56 |
| Temperature_1 | 1000 | 3000 | 3000 | 0.0 | 110.00 | 52.67 |
| Temperature_2 | 1000 | 3000 | 3000 | 0.0 | 110.00 | 52.43 |

이 결과를 통해 StandardScaler가 필요하다는 결론을 얻었다.

---

## 6. 전처리 방법

### 6.1 episode 병합

원본 데이터는 센서별 CSV 파일로 분리되어 있었기 때문에, 같은 episode_id를 가진 6개 센서 CSV를 하나의 배열로 병합하였다.

변환 전:

```text
Accelerometer_00000.csv
GasLeak_00000.csv
Pressure_1_00000.csv
Pressure_2_00000.csv
Temperature_1_00000.csv
Temperature_2_00000.csv
```

변환 후:

```text
episode_00000.npz
values.shape = (3000, 6)
```

센서 순서는 다음과 같이 고정하였다.

```text
0: Accelerometer
1: GasLeak
2: Pressure_1
3: Pressure_2
4: Temperature_1
5: Temperature_2
```

---

### 6.2 train 기준 StandardScaler 생성

train episode 700개만 사용해 센서별 평균과 표준편차를 계산하였다.

```text
train samples = 700 episode × 3000 step = 2,100,000 samples
```

이 평균과 표준편차는 train/validation/test 모두에 동일하게 적용한다.

스케일링 공식:

```text
x_scaled = (x - train_mean) / train_std
```

---

### 6.3 window 생성

입력 window는 과거 10초로 설정하였다.

```text
input_len = 100 step
1 step = 0.1초
100 step = 10초
```

입력 X:

```text
X = values[start : start + 100]
X shape = (100, 6)
```

정답 y:

```text
y = values[start + 100 - 1 + horizon_steps]
y shape = (6,)
```

예측 horizon은 다음과 같다.

| horizon | step | 초 |
|---|---:|---:|
| 3s | 30 | 3초 |
| 6s | 60 | 6초 |
| 30s | 300 | 30초 |
| 60s | 600 | 60초 |
| 120s | 1200 | 120초 |

---

### 6.4 horizon별 window 수

| 예측 시점 | episode당 window 수 | train window 수 | val window 수 | test window 수 |
|---|---:|---:|---:|---:|
| 3초 | 2871 | 2,009,700 | 430,650 | 430,650 |
| 6초 | 2841 | 1,988,700 | 426,150 | 426,150 |
| 30초 | 2601 | 1,820,700 | 390,150 | 390,150 |
| 60초 | 2301 | 1,610,700 | 345,150 | 345,150 |
| 120초 | 1701 | 1,190,700 | 255,150 | 255,150 |

window 수가 매우 많기 때문에 전체 window를 메모리에 한 번에 만들지 않고, 학습 시 batch 단위로 npz에서 잘라서 사용하는 방식으로 구현하였다.

---

## 7. 모델 구성

### 7.1 현재 모델: Dense baseline

현재까지 학습 검증에 사용한 모델은 Dense baseline이다.

모델 구조:

```text
Input: (100, 6)
Flatten: 600
Dense(128, relu)
Dense(64, relu)
Dense(6)
```

모델 파라미터 수:

```text
85,574
```

이 모델은 시계열 순서를 적극적으로 모델링하지는 않는다. 과거 10초 데이터를 단순히 flatten한 뒤 Dense 층으로 미래값을 예측한다.

따라서 최종 모델이라기보다는 기준 모델이다.

---

### 7.2 앞으로 사용할 모델

앞으로 사용할 모델은 다음과 같다.

| 모델 | 목적 |
|---|---|
| Dense baseline | 가장 단순한 딥러닝 기준 모델 |
| 1D CNN | 짧은 시간 구간의 지역적 패턴 학습 |
| GRU | 순환 신경망 기반 시계열 패턴 학습 |
| LSTM | 장기 의존성 학습 |

---

## 8. 학습 방법

### 8.1 현재 학습 방식

현재는 Dense baseline을 다음 방식으로 테스트 학습하였다.

- horizon: 3초
- epochs: 2
- train_batches: 20
- val_batches: 10
- test_batches: 10
- batch_size: 128
- optimizer: Adam
- loss: MSE
- metrics: MAE, RMSE

이 학습은 성능 확보가 아니라 학습 파이프라인 검증이 목적이었다.

---

### 8.2 학습 파이프라인 검증 결과

검증된 항목:

- `/usr/bin/python`에서 TensorFlow 실행 가능
- GPU RTX 3080 사용 가능
- `processed` 데이터 읽기 가능
- `training_assets` 설정 읽기 가능
- npz episode에서 batch 생성 가능
- StandardScaler 적용 가능
- Keras 모델 학습 가능
- `.keras` 모델 저장 가능
- history/eval CSV 저장 가능

---

## 9. 성능 평가

### 9.1 평가 지표

평가 지표는 다음을 사용한다.

| 지표 | 의미 |
|---|---|
| MSE | 예측 오차 제곱 평균 |
| MAE | 예측 오차 절대값 평균 |
| RMSE | MSE의 제곱근 |
| raw MAE | 실제 센서 단위에서의 평균 절대 오차 |
| scaled MAE | 표준화된 값 기준 평균 절대 오차 |

---

### 9.2 naive baseline

naive baseline은 다음 방식이다.

```text
미래값 예측 = 입력 window의 마지막 값
```

즉, 센서값이 앞으로도 그대로 유지된다고 가정하는 방식이다.

3초, 6초처럼 짧은 horizon에서는 naive baseline이 매우 강할 수 있다. 센서값이 짧은 시간에는 크게 변하지 않을 가능성이 높기 때문이다.

---

### 9.3 Dense baseline 테스트 결과

2 epoch 테스트 결과:

| split | scaled MAE | raw MAE 평균 |
|---|---:|---:|
| validation | 약 0.143 | 약 9.71 |
| test | 약 0.140 | 약 9.69 |

기존 3초 naive baseline은 test scaled MAE가 약 0.059였다. 따라서 현재 Dense 2 epoch 테스트 모델은 naive baseline보다 좋지 않다.

그러나 이 결과는 실패가 아니다. 이 실험은 파이프라인이 정상 작동하는지 확인하기 위한 짧은 테스트였다.

---

## 10. 현재 단계

현재 위치는 다음과 같다.

```text
Step 1. 원본 데이터 구조 검증 완료
Step 2. episode 단위 병합 완료
Step 3. scaler/window/baseline 생성 완료
Step 4. Dense baseline 3초 테스트 학습 완료
```

현재는 모델 성능 개선 단계로 넘어가기 전, 보고서의 데이터 이해 부분을 정리하는 단계이다.

---

## 11. 앞으로 남은 단계

### Step 5. 데이터셋 및 EDA 보고서 정리

해야 할 일:

- 데이터 구조 표 작성
- 센서 목록과 의미 작성
- episode 개념 설명
- 결측치/행 수/파일 수 검증 결과 작성
- 센서별 값 범위 작성
- StandardScaler 필요성 작성
- train/val/test 분할 방식 작성
- window/target 정의 작성
- naive baseline 의미 작성

---

### Step 6. Dense baseline 중간 규모 학습

3초 horizon에 대해 조금 더 긴 학습을 수행한다.

목적:

- Dense baseline이 어느 정도까지 개선되는지 확인
- naive baseline과 비교
- 이후 CNN/GRU/LSTM의 기준으로 사용

---

### Step 7. 전체 horizon Dense baseline 학습

3s, 6s, 30s, 60s, 120s 전체 horizon에 대해 Dense baseline을 학습한다.

---

### Step 8. 1D CNN 모델 학습

짧은 구간의 센서 변화 패턴을 학습하는 모델을 구성한다.

---

### Step 9. GRU 또는 LSTM 모델 학습

순환 신경망 기반 시계열 예측 모델을 구성한다.

---

### Step 10. 최종 성능 비교 및 보고서 작성

최종 보고서에는 다음 비교표가 들어가야 한다.

| 모델 | 3s | 6s | 30s | 60s | 120s |
|---|---:|---:|---:|---:|---:|
| Naive baseline |  |  |  |  |  |
| Dense baseline |  |  |  |  |  |
| 1D CNN |  |  |  |  |  |
| GRU/LSTM |  |  |  |  |  |

또한 센서별 MAE도 정리한다.

| 모델 | Accelerometer | GasLeak | Pressure_1 | Pressure_2 | Temperature_1 | Temperature_2 |
|---|---:|---:|---:|---:|---:|---:|
| Naive baseline |  |  |  |  |  |  |
| Dense baseline |  |  |  |  |  |  |
| 1D CNN |  |  |  |  |  |  |
| GRU/LSTM |  |  |  |  |  |  |

---

## 12. 보고서 작성 시 핵심 문장

다음 문장은 보고서에 그대로 사용할 수 있다.

> 본 프로젝트는 도시가스/보일러 관련 6종 센서의 시계열 데이터를 이용하여 미래 센서값을 예측하는 지도학습 기반 다변량 시계열 회귀 문제이다. 원본 데이터는 1000개의 독립 episode로 구성되며, 각 episode는 0.1초 간격으로 측정된 3000 step의 센서값을 포함한다. 각 episode마다 Accelerometer, GasLeak, Pressure_1, Pressure_2, Temperature_1, Temperature_2의 6개 센서가 존재한다. 서로 다른 episode는 시간적으로 연속된 데이터가 아니므로, sliding window는 episode 내부에서만 생성하였다. 또한 센서별 값 범위 차이가 크기 때문에 train split에 대해서만 평균과 표준편차를 계산하여 StandardScaler 방식의 표준화를 수행하였다. 이후 과거 10초의 센서값을 입력으로 사용하여 3초, 6초, 30초, 60초, 120초 후의 6개 센서값을 예측하도록 학습 데이터를 구성하였다.

---

## 13. 오늘 기준 결론

오늘의 핵심 목표는 모델 성능 개선이 아니라 데이터 이해이다.

현재까지의 결론은 다음과 같다.

1. 데이터 구조는 정상이다.
2. episode 단위 병합은 정상이다.
3. train/validation/test 분할은 episode 단위로 적절히 수행되었다.
4. 센서별 스케일 차이가 커서 표준화가 반드시 필요하다.
5. train 데이터 기준 scaler를 사용해 데이터 누수를 방지하였다.
6. 문제는 분류가 아니라 시계열 회귀 예측이다.
7. Dense baseline은 학습 파이프라인 검증용으로 정상 작동하였다.
8. 본격적인 성능 비교는 이후 Dense 중간 학습, 1D CNN, GRU/LSTM으로 진행한다.
