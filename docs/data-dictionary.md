# 데이터 사전

## `batches.jsonl`

한 줄이 실제 학습 microbatch 한 번입니다. `optimizer_step`은 이 batch가 기여하는 update 번호입니다. 누적8이면 여덟 줄이 같은 optimizer step을 가집니다.

| 필드 | 의미 |
|---|---|
| `microbatch` | 이 실행 내 forward 기록 순번, 1부터 |
| `optimizer_step` | 현재 누적 gradient가 적용될 optimizer update |
| `examples_seen` | 중복·oversampling을 포함한 누적 샘플 노출 |
| `batch_size` | 이 forward의 실제 샘플 수 |
| `loss` | 같은 forward의 평균 학습 loss |
| `samples` | 모든 샘플의 입력·예측·정답·CE |
| `time` | Unix seconds; UI에서는 KST 표시 |
| `forward_ms` | 계측 구간 CPU wall time; 총 step latency 아님 |
| `gpu_allocated_mib` | 이 기록 시점 PyTorch allocated MiB |
| `gpu_peak_allocated_mib` | 해당 계측 구간의 누적 peak |

### sample

`source`는 공식 MNIST train/test 배열, `image_id`는 그 배열의 index입니다. 숫자 label이 아닙니다. `current`는 현재 숫자이며, `training_row`는 균형화된 row pool의 인덱스입니다. 같은 이미지·현재상태가 oversampling 때문에 여러 row/배치에 나타날 수 있습니다.

`model_input`에는 이미지 참조, 실제 영어 state, questions가 들어갑니다. `gold`와 `target_digit_metadata`는 학습 감독/해석용입니다. `prediction`은 확률이 가장 큰 행동입니다. `probabilities`는 key로 행동을 식별해야 합니다. JSON key 순서가 -1,0,+1 순서라고 가정하면 안 됩니다. `cross_entropy`는 -log p(gold)입니다.

`input_provenance`가 실제 encoded source 캡처인지, 과거 schema 보완인지를 구분합니다. 모든 raw sample을 바꾸지 않고 보존했으며 파생 export에만 누락 model_input 설명을 보완할 수 있습니다.

## `training_history.json`

Trainer가 logging interval마다 남긴 loss·gradient norm·LR·epoch 등의 기록입니다. loss는 그 interval의 평균이므로 마지막 microbatch loss와 같지 않습니다. epoch는 row pool 대비 데이터 노출량이며 원본 이미지 고유 개수의 비율이 아닙니다.

## `full_test/images.jsonl`

한 줄이 이미지 한 장입니다. `predictions[s]`와 `probabilities[s]`는 current=s의 결과, `episodes[s]`는 start=s의 rollout입니다. 각 episode는 성공 여부, 종료 이유, `trace`의 before/action/after, 성공 시 최단 이동 대비 추가 이동 수를 포함합니다. 캐시 정책을 replay한 결과이며 모든 trajectory에서 VLM을 새로 호출한 것이 아닙니다.

`summary.cells[target][current]`는 해당 조건의 이미지 수, 행동 오답 수, 그 출발점에서의 실패 수입니다. `errors`와 `failed_episodes`는 다른 지표입니다.

## 정적 사이트 파일

`site/data/catalog.js`: 실행 목록. `site/data/runs/<run>.js`: config·curve·Trainer/validation·파일 목록. `site/data/batches/<run>/<index>.js`: 해당 배치 전체. `site/data/test.js`: 1,000개 이미지 결과, summary, analysis, latency. JavaScript assignment로 포장하여 file://에서 추가 서버 없이 읽도록 했습니다. `site/images/`는 실제 표시용 PNG이며 숫자를 새로 그린 이미지가 아닙니다.

사이트의 rule 및 temperature 위젯만 설명용 계산입니다. 실험 예측과 구분해서 표시합니다.
