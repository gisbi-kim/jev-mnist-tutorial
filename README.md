# MNIST로 decision model 학습해 보기

**[튜토리얼 웹사이트](https://gisbi-kim.github.io/jev-mnist-tutorial/)** · [최종 모델 다운로드·실행 안내](docs/model-download.md) · [실험 재현](docs/reproduction.md)

Jev와 decision model의 개념을 소개하고, Unsloth로 Qwen3.5-0.8B에 LoRA와 Clef 방식 decision head를 붙여 학습한 과정을 정리하였다. 이미지의 숫자를 목표로 주고, 영어로 주어진 현재 숫자에서 −1, 0(STOP), +1 중 어떤 행동을 해야 하는지 고르는 MNIST 실험이다. Jev 자체를 학습한 결과나 공식 튜토리얼은 아니다.

본문에는 입력부터 행동까지의 그림, LoRA와 head의 계산, 초기 학습 실패와 작은 데이터 암기, 최종 설정, 확률 보정, 학습·추론 시간, 실패 사례 분석을 넣었다. 저장된 **770개 microbatch의 20,892개 샘플**과 **1,000장 테스트의 10,000개 판단**을 페이지에서 조회할 수 있다. 각 배치의 loss와 이동평균, 실제 입력·정답·예측, 숫자와 현재 상태별 실패 필터가 동작한다.

## 웹사이트 열기

위 웹사이트는 GPU 없이 저장된 결과를 읽는다. clone한 뒤에는 저장소 루트에서 다음 명령으로 같은 사이트를 로컬에서 열 수 있다.

```bash
python -m http.server 7865 --bind 127.0.0.1
```

브라우저에서 `http://127.0.0.1:7865/`를 연다. Windows에서는 `Open tutorial.cmd`도 사용할 수 있다. 사이트의 `../raw/`, `../docs/` 링크가 있으므로 `site/` 폴더만 따로 떼어 서버에 올리면 안 된다. GitHub Actions가 전체 공개 파일 구조를 Pages에 배포한다. `.nojekyll`과 사용자 지정 배포를 사용하여 `_vendor` 코드 경로도 보존한다.

## 최종 가중치 포함

**Google Drive나 Git LFS 없이 일반 clone/ZIP에 최종 모델이 포함된다.**

| 구성 | 크기 (십진 MB) |
|---|---:|
| LoRA (`adapter_model.safetensors`) | 43.35 MB |
| Decision head (`joint_head.safetensors`) | 54.66 MB |
| tokenizer·processor·모델 설정을 포함한 전체 adapter 폴더 | 118.02 MB |

```bash
git clone https://github.com/gisbi-kim/jev-mnist-tutorial.git
cd jev-mnist-tutorial
python tools/prepare_workspace.py --destination ../mnist-decision-workspace
```

[모델 실행 안내](docs/model-download.md)에 환경 구성, 단건 예측, 새 1,000장 평가 명령을 적었다. 첫 모델 로딩에는 Hugging Face의 `unsloth/Qwen3.5-0.8B` 기반 가중치 다운로드가 별도로 필요하다. 이 저장소의 118MB는 학습한 adapter/head와 재로딩 자료이며 완전한 기반 VLM 파일까지 포함한 크기가 아니다.

최종 모델은 `20261030-010019`, step 600이다. 과거 모델과 optimizer checkpoint는 제외했다. 9개 실행의 로그·설정·평가 결과는 유지하여 실패부터 최종 결과까지 비교할 수 있다. 공개 범위와 변경 내역은 [publication.md](docs/publication.md)를 참고한다.

## 실험 결과와 한계

- 학습 이미지 2,000장, 균형화 row 27,336개, batch 32 × 누적 1, 600 update, 실제 19,200개 샘플 노출.
- 공식 MNIST test ID 0–999에서 행동 정확도 **97.70%**, 에피소드 성공률 **93.62%**.
- 최종 모델은 best-validation checkpoint를 자동 선택한 결과가 아니라 step 600 모델이다. 초기 실패와 본학습 사이에는 batch, head 학습률, 노출량, 영어 입력 등 여러 조건이 함께 바뀌었다.
- 1,000장 테스트는 공식 test의 앞부분이다. 반복적으로 관찰한 결과이므로 완전히 봉인된 독립 테스트라고 주장하지 않는다.
- 기록되지 않은 과거 배치 예측을 사후에 생성하지 않았다. Qwen 2B는 설명·후속 비교 대상이며 실행하지 않았다.

## 자료 구성

| 위치 | 내용 |
|---|---|
| `site/` | 정적 튜토리얼, SVG 그림, loss plot, 배치·실패 사례 탐색기 |
| `raw/code/` | 실험 종료 시점 학습·추론·평가 코드와 기존 대시보드 |
| `raw/runs/` | 9개 실행 로그·설정·결과, 최종 실행의 adapter/head |
| `raw/data/` | MNIST 배열·원본 압축 파일·split·압축 JSONL |
| `raw/session_logs/`, `raw/dashboard_api/` | 당시 실행 로그와 API 응답 snapshot |
| `raw/upstream/`, `raw/backbone-metadata/` | 사용한 구현과 라이선스, 관찰한 기반 모델 설정 |
| `raw/prior_reports/` | 당시 보고서와 측정 자료 |
| `docs/`, `prompts/` | 재현·검증·출처·보존 범위, 정리한 실험 요청 |
| `tools/` | 데이터 검증, 무손실 복원, workspace 구성, Pages 배포 도구 |

세 데이터 JSONL은 346.46MB에서 약 3.90MB로 무손실 압축하였다. 원래 JSONL이 필요하면 `python tools/restore_data.py`로 풀 수 있다. 학습 코드는 배열과 split에서 row를 만들므로 일반 실행에는 JSONL 복원이 필요하지 않다.

## 검증과 무결성

```bash
python tools/verify_tutorial.py
python tools/verify_publication.py
```

Python 표준 라이브러리만으로 770개 배치와 원본 로그의 일치, 1,000장 rollout 재계산, 정적 링크, 실행별 파일 링크, 최종 모델 파일과 전체 SHA-256, 압축 데이터의 원본 해시를 검사한다. GitHub Actions도 같은 검사를 통과한 뒤 Pages를 배포한다. 현재 파일 목록과 바이트 수는 [manifest.json](manifest.json), 해시는 [SHA256SUMS.txt](SHA256SUMS.txt)에 있다.

원자료 묶음은 약 3.29GB였으며, 공개판은 과거 checkpoint를 제외하고 데이터 JSONL을 압축하였다. 공개 전 일반적인 credential 패턴을 검사했고, 기록 속 로컬 사용자 경로는 `USER`로 일반화하였다. 측정값은 유지했다. 모델 가중치와 tokenizer는 원본과 바이트 단위로 동일하다.

## 출처와 버전

[출처·라이선스](docs/sources.md), [기록 범위와 한계](docs/provenance.md), [공개판 변경 내역](docs/publication.md)에 정리하였다. 외부 코드와 모델의 기존 라이선스를 유지한다. 원래 `raw/code/README.md`는 초기 버전의 보존 자료이며 현재 실행 안내는 이 README와 `docs/model-download.md`를 기준으로 한다.
