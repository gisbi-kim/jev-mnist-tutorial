# 재현 안내

> GitHub 공개판 안내: 최종 모델만 제공하고 중간 checkpoint는 제외하였다. 공개 범위는 [publication.md](publication.md), 현재 다운로드·실행 방법은 [model-download.md](model-download.md)를 따른다. 아래는 원래 실험 snapshot에 대한 기록이다.

## 결과 열람

`site/index.html`을 더블클릭하거나 `start-preview.ps1`을 실행합니다. site의 데이터 파일은 script로 로딩하므로 file://에서도 작동합니다. 저장된 결과의 탐색이며 새 추론은 수행하지 않습니다.

## 학습 환경

원래 환경: Windows, Python 3.13, RTX 5070 Ti Laptop GPU 12GB. 기록된 주요 버전: Unsloth 2026.10.10, torch 2.14.1+cu130, transformers 5.17.0, peft 0.21.2. 정확한 dependency lock은 `raw/runs/20261030-010019/requirements.lock.txt`입니다. OS/CUDA 별 wheel과 driver 호환성이 필요하며 다른 환경의 속도·수치 동일성을 보장하지 않습니다.

패키징 시 실제 Python은 3.13.12였습니다. `environment-snapshot.json`에 현재 설치 환경과 관찰된 기반 모델 캐시 revision `23c69c53358a07516b5827588b3fdb12ae78fd65`를 기록했습니다. 원래 CLI는 revision을 명시적으로 pin하지 않았으므로 관찰된 cache 식별자라는 점을 구분합니다. 기반 모델 config만 `raw/backbone-metadata/`에 보관하고 대용량 기반 가중치는 제외했습니다.

`resolved-training-args/`에는 남아 있는 checkpoint의 `training_args.bin`을 읽어 풀어 쓴 설정이 있습니다. 본학습 optimizer는 `adamw_torch_fused`, β1=0.9, β2=0.999, ε=1e-8, max gradient norm=1.0입니다. `DecisionTrainer`는 encoder 여부 및 decay 대상 여부로 parameter group을 분리하고 head LR을 별도로 설정합니다. bias/norm 등 decay 제외 파라미터는 weight decay 0입니다.

원자료는 직접 수정하지 말고 별도 작업 디렉터리를 만드세요. `raw/code/*`를 새 작업 디렉터리에 복사하고 `raw/data/`를 그 안의 `data/`로 복사하면 원래 상대경로 레이아웃이 됩니다. 저장 모델을 사용할 때는 필요한 run을 그 안의 `runs/`에 복사하세요. 원래 `latest.json`은 절대경로를 담으므로 새 위치에 맞게 다시 만들거나 명시적 adapter 경로를 쓰세요. `.venv`는 복사하지 않았습니다.

## 본학습 명령

환경 구성 후 새 작업 디렉터리에서 다음 명령을 사용합니다. 원래 코드의 기본 head LR은 1e-4이므로 **1e-3을 명시**해야 이번 성공 설정과 같습니다.

```powershell
$env:UNSLOTH_COMPILE_DISABLE='1'
$env:UNSLOTH_DECISION_COMPILE='0'
.\.venv\Scripts\python.exe train.py --model unsloth/Qwen3.5-0.8B --train-images 2000 --val-images 100 --test-images 100 --steps 600 --validation-steps 100 --batch-size 32 --grad-accum 1 --learning-rate 0.0002 --head-lr 0.001 --trace-batches
```

자동 생성되는 새 run ID를 사용하세요. 기존 결과를 덮어쓰지 마세요. 처음 600-step 실패 실행의 코드는 개별 버전으로 저장되지 않았으므로, 현재 소스만으로 그 실행을 완전히 재현한다고 주장할 수 없습니다.

## 저장 모델 재로딩 및 1,000장 평가

```powershell
.\.venv\Scripts\python.exe full_test.py --run NEW_RUN_ID --batch-size 32 --limit 1000
```

`full_test.py`는 저장된 adapter를 새로 불러옵니다. 기본 `--limit`은 최종 수정 후 1000이며, 원래 평가 실행은 10000으로 시작했다가 1000장에서 중단한 기록입니다. 원자료의 `summary.json`과 `config.json`에 변경 범위가 남아 있습니다. 같은 run의 결과가 있으면 resume 동작을 확인하세요. 새 속도 측정에는 `profile_inference.py`의 run 경로를 새 run ID로 수정해야 합니다. 그 스크립트는 당시 실행 ID를 고정한 실험용 코드입니다.

## 순서와 확인 사항

1. `core.py`의 oracle 및 rollout 규칙을 확인합니다. label은 입력이 아니라 감독 신호입니다.
2. `data.py`의 split/seed/이미지 변환을 확인합니다.
3. `model.py`가 multimodal processor와 CUDA를 확인하고 4bit backbone + LoRA를 로딩합니다.
4. `train.py`가 2,000×10 row를 생성하고 행동별 개수를 맞추기 위해 oversampling합니다.
5. Unsloth `build_dataset`에서 누락·truncation이 없는지 검사합니다.
6. 실제 forward 기록과 target metadata가 일치하는지 telemetry가 확인합니다.
7. 600 step 후 별도 calibration 이미지로 온도를 맞추고 adapter/head를 저장합니다.
8. reload 후 action mapping·확률 합·단건/배치 parity를 점검하고 test와 latency를 측정합니다.

정확한 bitwise 재현이나 모든 커널의 결정성까지 보장하는 설정은 아닙니다. 최종 모델은 best-validation checkpoint 자동 선택 결과가 아니라 step 600 모델입니다.
