# 출처와 설명의 근거

> GitHub 공개판 안내: 최종 모델만 제공하고 중간 checkpoint는 제외하였다. 공개 범위는 [publication.md](publication.md), 현재 다운로드·실행 방법은 [model-download.md](model-download.md)를 따른다. 아래는 원래 실험 snapshot에 대한 기록이다.

웹 자료 확인일: 2026-10-30. 설명은 요약이며 외부 페이지 전체를 복제하지 않았습니다. 본 실험 수치는 외부 벤치마크가 아니라 `raw/runs/`에서 가져왔습니다.

| 주제 | 1차 자료 |
|---|---|
| Jev / System One / typed decisions | https://typesafe.ai/ 및 https://docs.typesafe.ai/ |
| Clef 공개 모델과 Jev API 호환성 | https://blog.cloudflare.com/clef-decision-models/ 및 https://huggingface.co/Cloudflare/clef |
| Unsloth decision 학습 API | https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth |
| 실제 사용한 구현 | `raw/upstream/unsloth/models/decision.py`, `clef.py`, `_vendor/clef/joint_schema_model.py` |
| Qwen3.5-0.8B | https://huggingface.co/Qwen/Qwen3.5-0.8B |
| Qwen3.5-2B | https://huggingface.co/Qwen/Qwen3.5-2B |
| 실제 로딩 식별자 | https://huggingface.co/unsloth/Qwen3.5-0.8B |
| LoRA 논문 | https://arxiv.org/abs/2106.09685 |
| 실제 설치 버전 | 각 run `requirements.lock.txt`, config의 `versions` |
| MNIST 다운로드 출처·체크섬·split | `raw/code/data.py`, `raw/data/manifest.json`, `splits.json` |

Jev의 내부 구조나 독자 학습 알고리즘을 이 실험에서 재현했다고 주장하지 않습니다. RLCD를 사용했다는 Jev 설명과 여기서 수행한 supervised cross-entropy + post-hoc temperature calibration을 구분합니다. 공식 제품의 속도·가격 비교 수치를 이 로컬 GPU 결과와 직접 비교하지 않습니다.

## 핵심 주장 → 증거

- 설정: `raw/runs/20261030-010019/config.json`, `raw/code/train.py`, adapter configs.
- 학습 과정: 같은 실행의 `batches.jsonl`, `training_history.json`, `validation_history.json`.
- 작은 데이터 암기: `raw/runs/20261030-005547-memorize/diagnosis.json`, `memorization_history.json`, `input_audit.json` 등 실제 파일 목록.
- 단건·배치 속도: 본학습 `full_test/latency.json`, `raw/code/profile_inference.py`.
- 실패 사례: 본학습 `full_test/images.jsonl`, `summary.json`, `analysis.json`.
- 보정: 본학습 `calibration.json`, `adapter/unsloth_decision_config.json`; 보정 전후 동일 test의 paired calibration 개선 비교는 별도로 저장하지 않았으므로 개선량을 주장하지 않습니다.
- 전체 로그 파일·보존된 checkpoint: 사이트 실행별 파일 목록과 패키지 SHA256SUMS.

## 라이선스

이 묶음은 원자료 보존용입니다. 외부 모델·라이브러리의 라이선스를 변경하지 않습니다. 포함된 upstream 파일의 헤더와 `raw/upstream/unsloth/licenses/`, vendored Clef 라이선스를 확인하세요. 기반 Qwen 모델은 공식 model card의 라이선스를 따릅니다. 패키지 전체에 새 라이선스를 일괄 적용하지 않았습니다.
