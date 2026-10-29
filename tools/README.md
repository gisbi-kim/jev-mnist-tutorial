# 공개판 도구

- `verify_tutorial.py`: 원자료와 배치 export, 테스트 rollout 및 정적 링크 검증. 저장소 위치를 자동으로 찾는다.
- `verify_publication.py`: 현재 manifest, 압축 데이터, 공개 파일 링크 검증.
- `restore_data.py`: 세 JSONL을 원본 SHA-256으로 확인하며 복원.
- `prepare_workspace.py`: 새 디렉터리에 코드·MNIST 배열·최종 모델 복사.
- `predict_saved.py`: 복원 workspace에서 쓰는 단건 추론 예제.
- `update_manifest.py`: 공개 소스 수정 후 해시 목록 갱신.
- `build_pages.py`: 추적된 파일을 `_site/`에 복사해 상대경로 유지. GitHub Actions용.

아래의 `package_tutorial.py`와 `finalize_tutorial.py`는 최초 로컬 수집 당시의 보존 코드이다. 원래 경로와 API에 의존하므로 공개판을 읽거나 실행하기 위해 호출하지 않는다.

# 패키징 도구

- `package_tutorial.py`: 원본 실험 폴더를 복사하고 기존 대시보드 API 및 JSONL에서 정적 데이터를 생성한 스크립트. 원래 Windows 경로와 API 주소가 명시되어 있습니다. 다른 컴퓨터에서는 SRC/OUT/ROOT를 수정해야 합니다. HTML/CSS/app.js와 문서는 별도로 작성된 소스입니다.
- `verify_tutorial.py`: 770개 배치 export를 원본과 대조하고 1,000장 모든 rollout·오류 집계·heatmap·영어 입력·로컬 링크를 검증합니다. NumPy/Pillow는 패키징 단계에만 필요합니다.
- `finalize_tutorial.py`: 원본과 복사본의 SHA-256을 대조하고 파일 manifest/checksum 목록을 생성합니다. 원본 실험 파일은 수정하지 않습니다.

이 도구들은 이번 자료 생성 과정을 남긴 것입니다. 원자료 폴더와 사이트를 읽기 위해 실행할 필요는 없습니다. 재실행하면 파생 산출물과 스냅샷을 갱신하므로, 현재 묶음을 보존하려면 먼저 별도 폴더로 복사하세요.
