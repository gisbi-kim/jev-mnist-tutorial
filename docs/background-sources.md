# 배경과 API 가이드: 출처와 확인 범위

확인일: 2026-10-30 (한국 시간). 대상: `site/background.html`.

이 글은 공식 문서·발표, 개발자의 직접 보고, 논문 초록, 저장소에 보존된 실험 코드를 바탕으로 작성했다. 제조사 측정, 제품 기능 공개, 독립 평가, 설명용 예제를 구분한다. 연표는 확인한 주요 공개 이정표이며 분야 전체의 최초 발명이나 영향력을 입증하는 순위가 아니다.

## 출처별 사용 범위

| 출처 | 본문에 사용한 내용 |
|---|---|
| [TypeSafe 최초 발표](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 2026-09-15 공개, Diogo Almeida, System One·RLCD라는 제품 접근 |
| [TypeSafe 소개](https://docs.typesafe.ai/introduction) | state·questions·typed answers, 분류와 선택지 설명 |
| [Vercel 개념 설명](https://vercel.com/i/what-is-jev) | System One의 명칭과 빠른 판단이라는 비유 |
| [LangSmith 발표](https://www.langchain.com/blog/jev-is-now-available-in-langsmith-evals) | 2026-09-21 평가자 통합, 에이전트 기록 평가 |
| [2389 Research 개발자 보고](https://2389.ai/research/writing/jev-plays-pokemon/) | 2026-09-18, Michael Sugimura, 정책 선택과 계획·경로 계산의 역할 분담 |
| [Cloudflare 공개 발표](https://blog.cloudflare.com/clef-decision-models/) | 2026-10-01 Clef·Clef-flash, Matt Mastracci 관련 선행 실험, 도메인 분류의 전체 처리 시간 |
| [Clef 모델 카드](https://huggingface.co/Cloudflare/clef) | Qwen 기반 모델, joint schema head, 공개 가중치와 라이선스 |
| [Liquid d1 발표](https://www.liquid.ai/blog/d1-decision-model) | 2026-10-05 텍스트·이미지 모델과 여섯 응용 데모, 단회 실행이라는 측정 조건 |
| [Open d1](https://www.liquid.ai/blog/d1-open) | 2026-10-07 공개 가중치와 엣지 방향, omni의 연구용 상태 |
| [Deußer 등, arXiv:2609.37647](https://arxiv.org/abs/2609.37647) | 37개 데이터셋, 346,009개 요청, 과제별 차이와 이진 임계값의 중요성. 초록 범위 요약 |
| [Li 등, arXiv:2610.17820](https://arxiv.org/abs/2610.17820) | 공격에 대한 과제별 차이. 초록 범위 요약 |
| [Choice](https://docs.typesafe.ai/primitives/choice), [Score](https://docs.typesafe.ai/primitives/score), [Noul](https://docs.typesafe.ai/primitives/noul) | 요청 criteria와 응답 필드, score의 확률 가중 평균 |
| [Images](https://docs.typesafe.ai/images), [Box](https://docs.typesafe.ai/primitives/box) | 이미지와 Box의 존재, 실험 기능 상태와 별도 헤더 |
| [TypeSafe Quickstart](https://docs.typesafe.ai/introduction/quickstart) | REST endpoint, Bearer 인증, answers 구조 |
| [동기 Python 클라이언트](https://docs.typesafe.ai/sdk/python/api/clients/sync) | TypeSafeClient·system_one, 환경변수와 객체 반환 |
| [오류 명세](https://docs.typesafe.ai/api/errors) | 401·422·429·529와 재시도 |
| [Workers AI Clef](https://developers.cloudflare.com/workers-ai/models/clef/) | 모델별 endpoint와 필수 인자 |
| [Workers AI REST](https://developers.cloudflare.com/workers-ai/get-started/rest-api/) | 계정·토큰과 REST 응답 래퍼 |
| [Unsloth 소개](https://unsloth.ai/docs), [Desktop](https://unsloth.ai/docs/desktop) | 프레임워크와 앱의 구분 |
| [Unsloth README](https://github.com/unslothai/unsloth/blob/main/README.md) | LoRA·QLoRA·전체 미세조정 등 전체 기능 범위 |
| [RL 안내](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) | 보상 기반 학습 기능 |
| [GGUF 안내](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf) | 지원 내보내기 경로 |
| [Decision 학습](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) | 일반 기반 모델을 decision model로 추가 학습하는 기능 |
| [로컬 Decision API](https://unsloth.ai/docs/models/decision-laya) | 서빙 설정, localhost:8888, 모델 별칭, confidence 차이 |
| [보존된 decision.py](../raw/upstream/unsloth/models/decision.py) | Python 함수·인자·반환값·양자화 제약 |
| [model.py](../raw/code/model.py), [train.py](../raw/code/train.py) | MNIST 실험의 실제 로딩·학습·보정·저장·예측 호출 |

## 예제와 검증의 범위

- 문의 분류 예제와 확률 평균 예시는 설명을 위해 새로 작성했다. 실제 API 응답이나 측정 결과가 아니다.
- Python 예제는 문법을 검사하고, 함수와 반환 구조를 문서 및 보존 소스와 대조했다.
- 원격 API 키 생성·전송, 유료 호출, 모델 설치, GPU 학습은 수행하지 않았다.
- 학습 코드는 흐름을 보여주는 골격이다. 전체 재현에는 기존 `docs/reproduction.md`, `raw/code/train.py`, 저장된 버전·학습 설정을 사용한다.
- 현재 웹 문서와 보존된 Python 구현은 같은 버전이라고 가정하지 않는다. 페이지에서 각각의 기준을 명시했다.
- 역사·사례·API 설명의 외부 링크는 본문 가까이에 연결했다.


## 추가: 모델·API 지도 (2026-10-30)

호스팅 API와 가중치 공개를 별도 기준으로 분류. 최신 서버 모델과 공개 체크포인트의 동일성을 가정하지 않음.

### A. 서비스 API로 사용하는 모델
- [Jev · TypeSafe](https://docs.typesafe.ai/introduction): API. 선택·참거짓·등급 판단. 이 튜토리얼에서 소개하는 System One 형식의 기준점.
- [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions): API. predicate·choice·score를 지원합니다. 일반 채팅 API와 별도이며 Jev의 요청 JSON을 그대로 보내는 방식은 아닙니다.
- [Microsoft-Decision-1](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/): Foundry · OpenRouter. Qwen3.5-9B를 바탕으로 한 판단 모델. 분류·우선순위·검증 등에 쓰며 Foundry에서는 배포 설정이 필요합니다.
- [GLiDE · Fastino](https://fastino.ai/blog/introducing-glide-the-first-thinking-decision-model): API. 불확실한 질문에 추가 추론을 쓰는 접근. 아래의 공개 모델 GLiNER2.5-Decide와는 다른 제품입니다.
- [Mercury Decide · Inception](https://www.inceptionlabs.ai/): API. 상황과 타입이 정해진 질문을 받아 판단과 확률을 반환하는 제품입니다.
- [Nimble · Bespoke](https://console.bespokelabs.ai/): API · 가입 대기. 호스팅 판단 모델. 확인 당시 신규 가입이 일시 중단되어 대기 목록으로 안내되었습니다.
- [Parallel Decisions](https://parallel.ai/blog/decisions-api): API · 웹 근거 포함. 웹에서 자료를 찾고 출처와 함께 판단합니다. 제공한 입력만 채점하는 모델과는 검색 시간·비용까지 비교해야 합니다.
- [Span-01 · Respan](https://www.respan.ai/docs/documentation/features/gateway/guardrails): API · 기록 평가. 에이전트 실행 기록에 특정 행동이 있었는지 평가합니다. 범용 선택 모델보다 관측·평가 업무에 초점을 둡니다.

### B. API와 공개 모델을 함께 살펴볼 계열
- [Clef · Cloudflare](https://huggingface.co/Cloudflare/clef): Workers AI + 공개 가중치. 호스팅으로 호출하거나 공개 모델을 직접 실행할 수 있습니다. 모델별 실행 코드와 라이선스를 확인합니다.
- [d1 · Liquid AI](https://www.liquid.ai/blog/d1-open): 호스팅 + 일부 공개 모델. 호스팅 d1 계열과 공개 d1-3B·d1-omni-600M을 구별합니다. omni는 초기 연구 모델이며 공개 모델은 Liquid의 라이선스를 따릅니다.
- [pplx-Decider · Perplexity](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b): API + 공개 체크포인트. API는 v1.2와 Flash v1.2를 안내합니다. 공개 v1.1-27b는 Apache-2.0이며 전용 판단부·attention 설정을 보존해야 합니다. 최신 API 모델의 가중치 공개까지 의미하지 않습니다.
- [Span-02 · Respan](https://www.respan.ai/articles/best-ai-decision-models): API + 공개 모델 발표. 타입이 정해진 판단과 추론 예산을 다룹니다. 제조사는 공개 가중치도 안내합니다. 실제 다운로드 버전의 모델 카드와 라이선스를 다시 확인하세요.

### C. 직접 실행하거나 학습할 공개 모델·구현
- [Laya · Convai Innovations](https://huggingface.co/convaiinnovations/laya): Apache-2.0. 작은 encoder 계열. 영어 ModernBERT와 다국어 mmBERT 변형을 구별합니다. 기본 모델과 typed-decisions 특화 모델의 성능도 다릅니다.
- [Kev · Jared Palmer](https://github.com/jaredpalmer/kev): 공개 가중치 · 코드. Qwen 계열에 판단부를 붙인 구현. 크기별 모델을 로컬에서 비교하고 실행 코드를 살펴보기 좋습니다.
- [GLiNER2.5-Decide · Fastino](https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model): Apache-2.0 · 340M. CPU 실행을 지원하는 소형 모델. 판단뿐 아니라 span·관계 추출과 출력 사이 제약, full·LoRA 학습을 다룹니다.
- [H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b): Apache-2.0 · 4B. Qwen3.5-4B 기반의 텍스트·이미지 판단 모델입니다. 일반 채팅 모델용 예제보다 모델 카드의 판단 실행 방식을 우선 확인합니다.
- [vLLM Semantic Router · Decision](https://vllm-sr.ai/blog/decision-models/): 공개 모델 + 실행 프로젝트. 라우팅 등에 쓸 판단 모델 계열과 실행 환경입니다. vLLM이라는 추론 엔진 자체와 개별 Decision 모델은 구별합니다.
- [Strands Decider](https://github.com/strands-labs/strands-decider): 공개 모델 + 학습·서빙 코드. 판단부를 붙이는 학습 파이프라인부터 평가·로컬 서버까지 제공하는 프로젝트입니다. Perplexity의 Decider와는 별개입니다.


## 가격·성능·한국어 비교와 MNIST 추천 (2026-10-30)

모델별 값과 출처: [model-comparison.json](model-comparison.json). 속도와 점수는 서로 다른 조건이며 순위표로 사용하지 않는다. 미확인은 미지원이 아니다. 새 API 호출·벤치마크를 실행하지 않았다.

MNIST 기준값은 raw/runs/20261030-010019/full_test/summary.json 및 latency.json에서 읽었다. 행동 정확도 97.70%, episode 성공률 93.62%, 단건 p50 93.327 ms / p95 97.972 ms. 추천 순서는 이미지 지원, 크기, 도입 비용에 따른 편집 판단이며 해당 모델들의 MNIST 성능 결과가 아니다.
