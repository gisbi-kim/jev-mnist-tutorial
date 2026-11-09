# 최종 모델 다운로드와 실행

최종 실행은 `20261030-010019`이다. 저장소의 `raw/runs/20261030-010019/adapter/`에 LoRA, decision head, processor와 tokenizer 설정을 함께 제공한다. 과거 실행의 모델과 optimizer state는 제공하지 않는다.

| 파일 | 바이트 | 역할 |
|---|---:|---|
| `adapter_model.safetensors` | 43,346,432 | 학습한 LoRA |
| `joint_head.safetensors` | 54,661,902 | 학습한 decision head |
| `tokenizer.json` | 19,989,325 | 저장된 tokenizer |
| 나머지 설정·template | 24,389 | 모델 및 processor 재로딩 |

전체 adapter 디렉터리는 118,022,048 bytes(약 118MB)이다. Git LFS나 Google Drive를 사용하지 않으므로 일반 `git clone` 또는 GitHub의 **Code → Download ZIP**으로 받을 수 있다.

```bash
git clone https://github.com/gisbi-kim/jev-mnist-tutorial.git
cd jev-mnist-tutorial
python tools/prepare_workspace.py --destination ../mnist-decision-workspace
```

이 명령은 새로운 폴더에 코드, MNIST 배열, 최종 adapter를 복사한다. 과거 평가 결과는 복사하지 않으므로 새 테스트가 기존 예측을 재사용하지 않는다. 원래의 행 단위 JSONL도 필요하면 저장소에서 `python tools/restore_data.py`를 실행한다. 학습 코드는 이미지 배열과 split으로 row를 만들기 때문에 JSONL을 풀지 않아도 된다.

## 추론 환경

GPU 학습·추론에는 CUDA 호환 PyTorch, Unsloth, transformers 등이 필요하다. 실험 당시 환경은 [재현 안내](reproduction.md), 정확한 버전은 [requirements.lock.txt](../raw/runs/20261030-010019/requirements.lock.txt)에 기록하였다. 모든 OS에 동일한 wheel이 제공된다는 뜻은 아니며, 대상 장치에 맞는 CUDA 환경을 먼저 구성해야 한다.

저장된 LoRA와 head만으로 전체 VLM이 되는 것은 아니다. 첫 로딩 시 `unsloth/Qwen3.5-0.8B` 기반 모델을 Hugging Face에서 추가로 다운로드한다. 기존 HF 캐시가 있으면 재사용한다. 기반 모델은 이 저장소에 중복 포함하지 않았다.

환경을 준비한 뒤 새 workspace에서 실행한다.

```bash
python predict_saved.py --image-id 0 --current 3
python full_test.py --run 20261030-010019 --batch-size 32 --limit 1000
```

첫 명령은 공식 test 0번 이미지와 영어 현재 상태를 입력하고 선택한 행동 및 확률을 출력한다. 두 번째 명령은 저장된 모델로 1,000장 평가를 새로 수행한다. 이는 GPU 작업이며 정적 웹사이트를 보는 것과 별개이다. 모델 파일의 SHA-256은 저장소 루트 `manifest.json`에서 확인할 수 있다.

공개 준비 과정에서는 파일 무결성, workspace 복원, 정적 결과를 검사하였다. 새로운 환경에서 GPU 추론을 다시 실행한 것은 아니다. 기존 저장·재로딩 결과는 실험 기록에 보존되어 있다.

## Apple Silicon Mac에서 추론 및 웹 실습

Apple Silicon에서는 CUDA 대신 PyTorch MPS와 float16으로 저장된 LoRA와 Clef head를 로드한다. 아래는 M4 Pro에서 단일 예측과 웹 앱의 한 단계 예측을 확인한 경로다. 전체 테스트 정확도와 CUDA와의 수치적 동등성은 아직 확인하지 않았다. 일부 연산은 CPU fallback 또는 최적화되지 않은 PyTorch 구현을 사용한다. 학습 및 전체 평가의 CUDA 경로는 변경하지 않았다.

```bash
python tools/prepare_workspace.py --destination ../mnist-decision-workspace
cd ../mnist-decision-workspace
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mac.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
python predict_saved.py --image-id 0 --current 3
python server.py --port 7864
```

브라우저에서 http://127.0.0.1:7864 를 연다. 첫 실행 시 Hugging Face에서 기본 모델을 다운로드한다. `clef_reference.py`는 Cloudflare Clef의 Apache-2.0 코드이며 출처와 라이선스는 `clef_manifest.json`, `CLEF_LICENSE.txt`에 있다.
