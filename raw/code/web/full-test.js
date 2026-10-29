(()=>{
const get=id=>document.getElementById(id),make=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
let run='',offset=0,busy=false,stamp='',errorRequest=0;
async function fetchJSON(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error(r.status);return r.json();}
for(const id of ['ftTarget','ftCurrent'])for(let i=0;i<10;i++){const o=make('option',i);o.value=i;get(id).append(o);}
async function errors(){
 if(!run)return;const ticket=++errorRequest;const params=new URLSearchParams({offset,mode:get('ftMode').value});
 for(const [id,key] of [['ftTarget','target'],['ftCurrent','current']])if(get(id).value!=='')params.set(key,get(id).value);
 const d=await fetchJSON(`/api/training/full-test/${run}/errors?${params}`);
 if(ticket!==errorRequest)return;get('ftPage').textContent=`${d.total?offset+1:0}–${offset+d.rows.length} / ${d.total}건`;get('ftPrevious').disabled=!offset;get('ftNext').disabled=offset+12>=d.total;
 get('ftErrors').replaceChildren(...d.rows.map(r=>{
  const card=make('article');card.className='sample';const img=make('img');img.className='digit';img.src=`/api/training/image/test/${r.image_id}`;img.alt=`테스트 이미지 ${r.image_id}`;
  card.append(img,make('h3',`목표 ${r.target} · 현재 ${r.current} · #${r.image_id}`),make('p',r.state),make('b',`정답 ${r.gold} → 예측 ${r.prediction}`),make('p',['-1','0','+1'].map(a=>`${a}: ${(100*r.probabilities[a]).toFixed(1)}%`).join(' / ')));
  const detail=make('details');detail.append(make('summary',`에피소드 ${r.episode.success?'성공':r.episode.reason} · 이동 경로 보기`),make('pre',r.episode.trace.map(t=>`${t.before} → ${t.action} → ${t.after}${t.action==='0'?' (STOP)':''}`).join('\n')));card.append(detail);return card;
 }));
 if(!d.rows.length)get('ftErrors').append(make('p','현재 필터에서 기록된 오류가 없습니다.'));
}
async function refresh(){
 if(busy)return;busy=true;
 try{
 const selected=get('run').value;if(!selected)return;
 if(run!==selected){run=selected;offset=0;stamp='';get('ftTarget').value='';get('ftCurrent').value='';}
 const d=await fetchJSON(`/api/training/full-test/${run}`),s=d.summary;
 get('ftTitle').textContent=`테스트 ${(s?.total_images||d.config?.test_images||0).toLocaleString()}장: 어떤 숫자와 상황에서 틀리나`;
 get('ftInsights').replaceChildren(...(d.analysis?.findings||[]).map(x=>make('p',x)));
 get('ftLatency').textContent=d.latency?Object.entries(d.latency).filter(([k,v])=>v&&v.mean_ms).map(([k,v])=>`${k}: 평균 ${v.mean_ms.toFixed(1)}ms / p50 ${v.p50_ms.toFixed(1)}ms / p95 ${v.p95_ms.toFixed(1)}ms · ${v.decisions_per_second.toFixed(1)} 판단/초 · 호출당 ${v.decisions_per_call}개`).join('\n')+'\n모델 로딩 제외 · CUDA 동기화 · 워밍업 5회. 단건 100회, 배치별 20회 측정. batch는 인코딩 제외, pipeline은 리사이즈·인코딩 포함. 배치 ms는 전체 배치의 시간입니다.':'측정 중 또는 기록 없음';
 get('ftStatus').textContent=d.error?`평가 중단: ${d.error.error}`:!s?(d.config?'모델 확인 및 첫 배치 평가 중':'이 실행에는 전체 테스트 기록이 없습니다.'):`${s.phase==='complete'?'요청 범위 평가 완료':'평가 진행 중 · 아래 수치는 중간 결과'} · ${s.image_count.toLocaleString()} / ${s.total_images.toLocaleString()}장 · ${s.decision_count.toLocaleString()}개 판단${s.phase==='running'?' · 예상 남은 시간 약 '+Math.ceil((s.total_images-s.image_count)/s.images_per_second/60)+'분':''}`;
 get('ftMetrics').textContent=s?`행동 정확도 ${(s.action_accuracy*100).toFixed(2)}% (${s.errors}개 오답) · 에피소드 성공률 ${(s.episode_success_rate*100).toFixed(2)}% (${s.failed_episodes}개 실패)`:'—';
 if(!s){get('ftHeat').replaceChildren();get('ftErrors').replaceChildren();get('ftBreakdown').textContent='';return;}
 const key=run+':'+s.image_count+':'+get('ftMode').value;if(stamp===key)return;stamp=key;
 const table=make('table'),head=make('tr');head.append(make('th','목표 ↓ / 현재 →'));for(let c=0;c<10;c++)head.append(make('th',c));table.append(head);
 const episode=get('ftMode').value==='episode';
 for(let t=0;t<10;t++){
  const tr=make('tr');tr.append(make('th',t));for(let c=0;c<10;c++){
   const x=s.cells[t][c],n=episode?x.episode_failures:x.errors,rate=x.total?n/x.total:0;
   const td=make('td'),b=make('button',`${(100*rate).toFixed(1)}%`);b.title=`목표 ${t}, 현재 ${c}: ${n} / ${x.total}건. 클릭하면 오답 필터`;b.style.background=`rgba(240,95,80,${.06+.7*rate})`;b.onclick=()=>{get('ftTarget').value=t;get('ftCurrent').value=c;offset=0;errors();};td.append(b);tr.append(td);
  }table.append(tr);
 }get('ftHeat').replaceChildren(table);
 const lines=['목표 숫자별 오류 / 전체 판단'];s.cells.forEach((row,t)=>{const n=row.reduce((a,x)=>a+(episode?x.episode_failures:x.errors),0),total=row.reduce((a,x)=>a+x.total,0);lines.push(`${t}: ${n} / ${total} (${total?(100*n/total).toFixed(2):'0'}%)`);});
 lines.push('\n행동 혼동행렬: 행=정답, 열=예측 [-1, 0, +1]');s.confusion.forEach((r,i)=>lines.push(`${s.action_order[i]}: ${r.join(' / ')}`));lines.push('\n에피소드 실패 원인: '+JSON.stringify(s.failure_reasons));get('ftBreakdown').textContent=lines.join('\n');
 await errors();
 }catch(e){get('ftTitle').textContent=`테스트 ${(s?.total_images||d.config?.test_images||0).toLocaleString()}장: 어떤 숫자와 상황에서 틀리나`;
 get('ftInsights').replaceChildren(...(d.analysis?.findings||[]).map(x=>make('p',x)));
 get('ftLatency').textContent=d.latency?Object.entries(d.latency).filter(([k,v])=>v&&v.mean_ms).map(([k,v])=>`${k}: 평균 ${v.mean_ms.toFixed(1)}ms / p50 ${v.p50_ms.toFixed(1)}ms / p95 ${v.p95_ms.toFixed(1)}ms · ${v.decisions_per_second.toFixed(1)} 판단/초 · 호출당 ${v.decisions_per_call}개`).join('\n')+'\n모델 로딩 제외 · CUDA 동기화 · 워밍업 5회. 단건 100회, 배치별 20회 측정. batch는 인코딩 제외, pipeline은 리사이즈·인코딩 포함. 배치 ms는 전체 배치의 시간입니다.':'측정 중 또는 기록 없음';
 get('ftStatus').textContent='전체 테스트 조회 오류: '+e.message;}finally{busy=false;}
}
for(const id of ['ftTarget','ftCurrent'])get(id).onchange=()=>{offset=0;errors();};
get('ftMode').onchange=()=>{offset=0;stamp='';refresh();};get('ftPrevious').onclick=()=>{offset=Math.max(0,offset-12);errors();};get('ftNext').onclick=()=>{offset+=12;errors();};
refresh();setInterval(refresh,5000);
})();
