"""Build a report from frozen evaluation evidence, not from training predictions."""
import json
from pathlib import Path
from collections import Counter
from core import ACTIONS,oracle_action
from data import load_arrays,display_image
from train import dump
p=Path('runs/20261030-010019/full_test')
s=json.loads((p/'summary.json').read_text(encoding='utf-8'));lat=json.loads((p/'latency.json').read_text(encoding='utf-8'))
assert s['image_count']==1000 and s['phase']=='complete' and lat.get('complete')
rs=[json.loads(x) for x in (p/'images.jsonl').read_text().splitlines()]
bad_images=[];consistent=0;high=0
for r in rs:
    bad=[i for i,a in enumerate(r['predictions']) if a!=oracle_action(r['target'],i)]
    high+=sum(r['probabilities'][i][r['predictions'][i]]>=.9 for i in bad)
    if bad:
        bad_images.append(r['image_id'])
        consistent+=any(r['predictions']==[oracle_action(t,i) for i in range(10)] for t in range(10))
findings=[
    '1,000장 × 현재 숫자 10개: 행동 정확도 97.70% (230/10,000 오답), 에피소드 성공률 93.62% (638/10,000 실패).',
    '목표 8: 행동 오류율 5.28%, 에피소드 실패율 14.61%. 목표 2: 각각 3.97%, 10.17%.',
    '목표=현재=8에서 정지 실패 12/89 (13.48%). 전체 목표=현재 상황에서는 정지 실패 49/1,000 (4.90%).',
    '실패 에피소드: 잘못된 숫자에서 정지 434건, 반복 이동 203건, 숫자 범위 이탈 1건. 한 오답 정책이 여러 출발점에 영향을 준다.',
    f'오답 {s["errors"]}건 중 {high}건({high/s["errors"]:.1%})은 선택 확률 90% 이상. 높은 confidence가 정답을 보장하지 않는다.',
    f'이미지 {len(bad_images)}장은 10개 상태 중 적어도 하나에서 오답. 그중 {consistent}장은 다른 목표 숫자를 따르는 정책과 정확히 일치, {len(bad_images)-consistent}장은 어느 한 목표 숫자의 올바른 정책과도 일치하지 않는다. 이는 행동 패턴 관찰이며 인식 오류의 직접 증명은 아니다.',
    '범위: 공식 테스트 앞 1,000장(ID 0–999), 영어 고정 문장, 현재 숫자 0–9. 전체 10,000장/임의 필체/다른 언어 성능으로 일반화할 수 없다. 10,000개 판단은 1,000개 이미지에 묶여 있어 독립 표본 10,000개가 아니다.'
]
analysis=dict(findings=findings,error_image_count=len(bad_images),consistent_wrong_target_images=consistent,
    inconsistent_policy_images=len(bad_images)-consistent,high_confidence_errors=high)
dump(p/'analysis.json',analysis)
out=Path('C:/Users/USER/Documents/Codex/2026-10-29/e/outputs');assets=out/'test-report-images';assets.mkdir(exist_ok=True)
lines=['# MNIST Decision Model — 1,000장 테스트와 실패 분석','',
'평가 모델: `20261030-010019/adapter` · Qwen3.5-0.8B + LoRA + decision head · 학습 완료 후 저장된 확률 보정 모델을 재로드. 평가 중 가중치/보정값을 변경하지 않았다.','',
'## 평가 범위와 방법','',
'공식 MNIST 테스트 ID 0–999의 이미지 1,000장에 현재 숫자 0–9를 각각 적용했다. 모델 입력은 RGB 112×112 이미지, `The current number is {current}.`, 영어 질문과 선택지다. 정답 숫자는 채점용이며 모델 입력에 텍스트로 주지 않았다. 행동은 -1/0/+1 중 하나다.','',
'배치 32로 10,000개 판단을 얻고, 각 이미지의 10개 상태 예측을 캐시해 10개 출발점의 에피소드를 재현했다. 최대 20번 이동, 잘못된 정지·같은 상태 재방문·0–9 범위 이탈을 실패로 처리했다. 단건 API와 배치 추론은 30개 사전 비교에서 행동이 모두 일치했고, 확률 최대 차이는 약 0.000314였다.','',
'사용자 요청에 따라 전체 10,000장 평가는 1,000장에서 중단했다. 무작위 1,000장 추출이 아니라 앞 1,000장이다. 이전 100장 테스트와 표본 및 저장 모델/배치 경로가 달라 점수 차이를 모델 악화로 해석하면 안 된다.','',
'## 결과','',
'| 지표 | 결과 |','|---|---:|',
'| 행동 정확도 | 97.70% (9,770/10,000) |',
'| 에피소드 성공률 | 93.62% (9,362/10,000) |',
'| 10개 상태를 모두 맞힌 이미지 | 917/1,000 (91.7%) |',
'| 잘못된 정지 / 반복 이동 / 범위 이탈 | 434 / 203 / 1 |','',
'## 어떤 숫자에서 실패하는가','',
'| 목표 숫자 | 이미지 수 | 행동 오답/판단 | 행동 오류율 | 에피소드 실패율 |','|---|---:|---:|---:|---:|']
for t,row in enumerate(s['cells']):
    n=row[0]['total'];e=sum(c['errors'] for c in row);f=sum(c['episode_failures'] for c in row)
    lines.append(f'| {t} | {n} | {e}/{10*n} | {e/(10*n):.2%} | {f/(10*n):.2%} |')
lines+=['','목표 8과 2가 가장 취약하다. 목표 숫자 8에서는 이미 목표에 도달한 상태(현재 8)의 오류율이 12/89=13.48%, 현재 9는 10/89=11.24%, 현재 7은 8/89=8.99%였다.','',
'| 목표와 현재의 거리 | 행동 오답/판단 | 오류율 |','|---|---:|---:|','| 같음: STOP 필요 | 49/1,000 | 4.90% |','| 1 차이 | 61/1,821 | 3.35% |','| 2 이상 차이 | 120/7,179 | 1.67% |','',
'목표 근처에서 정지하거나 한 칸 이동하는 판단이 더 취약했다. 쉬운 먼 거리 판단이 전체 정확도를 높게 보이게 할 수 있으므로, 정지 정확도와 에피소드 성공률을 함께 봐야 한다.','',
'## 실제 실패 사례','']
arrays=load_arrays()
for ident,description in [(582,'정답 숫자 8인데, 모든 상태의 행동이 목표 2를 따르는 정책과 일치했다.'),(926,'정답 숫자 2인데, 모든 상태의 행동이 목표 8을 따르는 정책과 일치했다.'),(87,'정답 숫자 3. 현재 6에서 +1, 현재 7에서 -1을 선택해 6→7→6으로 반복했다.')]:
    r=rs[ident];path=assets/f'test-{ident}.png';display_image(arrays['test_images'][ident]).save(path)
    lines += [f'**테스트 #{ident} — {description}**','',f'![테스트 이미지 {ident}]({path.as_posix()})','',f'현재 0~9 순서의 예측: `{", ".join(r["predictions"])}`','']
lines+=['## 실패에서 배운 한계','']+[f'- {x}' for x in findings[3:]]+['',
'27장의 다른 목표를 따르는 패턴은 시각적 숫자 혼동과 일치하는 증거지만, 이 모델은 숫자 인식 결과를 별도 출력하지 않으므로 원인을 인식 단계로 단정할 수 없다. 나머지 56장의 오류도 현재 숫자 해석·상태 의존적인 시각 판단·head의 의사결정 중 어디서 생겼는지 이 테스트만으로 분해되지 않는다.','',
'## 추론 시간','',
'RTX 5070 Ti Laptop GPU에서 측정. CUDA 동기화, 모드별 워밍업 5회. 모델 로딩·디스크 IO 제외. 단건 API는 이미 메모리에 리사이즈된 이미지를 사용하며 토큰화/processor/forward/출력 변환을 포함한다. 배치 logits는 인코딩을 제외하지만 collation·전송·forward·CPU 결과 수집을 포함하므로 순수 CUDA kernel 시간은 아니다. 배치 pipeline은 이미지 리사이즈·인코딩도 포함한다.','',
'| 경로 | 측정 횟수 | 호출당 판단 | 평균 ms/호출 | p50 ms | p95 ms | 판단/초 |','|---|---:|---:|---:|---:|---:|---:|']
for k,v in lat.items():
    if isinstance(v,dict) and 'mean_ms' in v:lines.append(f'| {k} | {v["repeats"]} | {v["decisions_per_call"]} | {v["mean_ms"]:.1f} | {v["p50_ms"]:.1f} | {v["p95_ms"]:.1f} | {v["decisions_per_second"]:.1f} |')
lines+=['',f'평가 전체 경과 시간: {s["elapsed_seconds"]:.1f}초, 10,000판단 / 경과 시간 = {10000/s["elapsed_seconds"]:.1f}판단/초. 입력 준비·채점·파일 저장·사전 단건 비교를 포함하며 모델 최초 로딩은 제외한다.','',
'배치 처리량을 배치 크기로 나눈 값은 동시에 처리할 때의 amortized 비용이다. 실제 사용자 단건 응답 지연은 single_predict 수치를 봐야 한다. batch 측정은 고정된 배치를 반복했으며 서버 동시 요청·네트워크·cold start는 측정하지 않았다.','',
'## 다음 개선 실험 제안 (아직 수행하지 않음)','',
'1. 학습/검증 데이터에서 8·2 및 목표 근처/STOP 사례를 강화하고, 동일 조건의 대조 실험으로 효과를 확인한다. 이번 테스트 오답을 학습에 넣으면 별도의 새 평가셋을 확보해야 한다.',
'2. 이미지 숫자를 직접 분류하는 진단과, 목표 숫자를 텍스트로 제공하는 대조 실험을 분리해 시각 인식과 비교/행동 선택의 병목을 구분한다.',
'3. 학습에 사용하지 않은 검증 데이터로 confidence와 실제 오류의 관계를 평가한다. 이번 결과에서는 90% confidence 임계값만으로 많은 오답을 거를 수 없다.',
'4. 에피소드 수준에서 반복 상태 탐지와 범위 제한을 적용할 수 있으나, 이는 실패를 감지하는 장치이지 정답을 보장하는 해결책은 아니다.','',
'[대시보드에서 숫자·상황별 오답 확인](http://127.0.0.1:7864/training#fullTest)','']
(out/'test-1000-report.md').write_text('\n'.join(lines),encoding='utf-8')
dump(out/'test-1000-evidence.json',dict(summary=s,analysis=analysis,latency=lat))
print('Report saved')
