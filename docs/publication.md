# 공개판의 보존 범위

2026-10-30에 로컬 실험 묶음을 GitHub 공개판으로 정리하였다. 로컬 원자료 묶음은 그대로 보존하였다.

- 최종 실행 `20261030-010019`의 LoRA, decision head, tokenizer, processor와 보정 설정을 포함한다.
- 과거 실행의 adapter, 모든 중간 checkpoint와 optimizer state는 제외한다. 모델을 이용한 추론과 새 학습은 가능하지만 당시 optimizer 상태에서 정확히 이어서 학습하는 백업은 아니다.
- 9개 실행의 설정·로그·결과, 기록된 770개 microbatch와 20,892개 샘플, 1,000장 테스트 결과를 보존한다.
- 세 개의 데이터 JSONL은 gzip으로 무손실 압축한다. `python tools/restore_data.py`가 SHA-256을 검증하며 복원한다.
- 텍스트에 남은 로컬 사용자 경로 구성요소는 `USER`로 일반화하였다. 측정값과 모델 입력·정답·예측값은 바꾸지 않았다.
- 사이트의 실행별 파일 목록은 공개판에 실제로 포함된 파일만 연결한다. 상대경로를 유지하기 위해 Pages에는 `site/`뿐 아니라 `raw/`, `docs/`, `prompts/` 등 전체 공개 자료 구조를 배포한다.

세부 제외 목록과 압축 전후 해시는 [publication-inventory.json](publication-inventory.json)에 있다. 현재 파일의 무결성 기준은 루트 [manifest.json](../manifest.json)과 [SHA256SUMS.txt](../SHA256SUMS.txt)이다. 과거 `source-copy-verification.json`은 최초 로컬 복사 시점의 검증 기록이며 공개판의 현재 inventory가 아니다.

실험·검증의 한계는 [provenance.md](provenance.md)에 기록하였다. 그 문서의 원본 snapshot 보존 설명보다 이 공개판의 제외 범위가 우선한다. 과거 snapshot에 있던 전체 가중치 목록을 현재 다운로드 가능하다는 뜻으로 해석하면 안 된다.

## 버전 기록

| 상태 | 내용 |
|---|---|
| 로컬 실험 snapshot (2026-10-29–30) | 9개 실행의 당시 잔존 파일을 수집한 원자료 묶음 |
| 기술 문서 개정 (2026-10-30) | 순차적인 실험 설명, 입력·LoRA·head 그림 추가 |
| GitHub 공개판 (2026-10-30) | 최종 모델만 제공, 데이터 압축, 공개 파일 링크 정리, Pages 자동 검증·배포 |

## 외부 자료

이 문서는 Jev 또는 Cloudflare의 공식 튜토리얼이 아니다. 실제 실험 모델은 Qwen3.5-0.8B + Unsloth의 Clef 방식 head + LoRA이다. 포함된 외부 코드의 라이선스와 출처는 [sources.md](sources.md)에 기록되어 있다. 기반 모델의 라이선스는 해당 모델 카드에 따른다. 이 공개 작업에서 전체 자료에 새로운 단일 라이선스를 부여하지 않았다.

GitHub의 [일반 파일 크기 제한](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)과 [Pages 용량 제한](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)을 확인하여 이 구성을 정하였다.
