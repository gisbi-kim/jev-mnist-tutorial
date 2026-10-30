'use strict';
const target = document.getElementById('target');
const current = document.getElementById('current');
function updateRule() {
  const t = Number(target.value), s = Number(current.value), a = Math.sign(t - s);
  document.getElementById('targetValue').textContent = t;
  document.getElementById('currentValue').textContent = s;
  document.getElementById('action').textContent = a > 0 ? '+1' : a < 0 ? '−1' : '0';
  document.getElementById('explanation').textContent = a === 0 ? `현재 숫자와 목표가 ${t}로 같습니다. 더 움직이지 않고 여기서 멈추면 됩니다.` : `목표 ${t}이 현재 ${s}보다 ${a > 0 ? '크므로 1을 더합니다' : '작으므로 1을 뺍니다'}. 다음 숫자는 ${s + a}입니다.`;
  const path = [s];
  while (path[path.length - 1] !== t) path.push(path[path.length - 1] + a);
  const track = document.getElementById('track');
  track.replaceChildren(...Array.from({length: 10}, (_, n) => {
    const item = document.createElement('span'); item.textContent = n;
    if (path.includes(n)) item.classList.add('visited');
    if (n === t) item.classList.add('goal');
    if (n === s) item.classList.add('origin');
    return item;
  }));
  document.getElementById('pathText').textContent = `정답대로 계속 고르면: ${path.join(' → ')} → 멈춤 (0)`;
}
target.addEventListener('input', updateRule);
current.addEventListener('input', updateRule);
updateRule();
const links = Array.from(document.querySelectorAll('aside nav a'));
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    links.forEach(link => {
      const active = link.hash === '#' + entry.target.id;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
    });
  }
}, {rootMargin: '-15% 0px -65% 0px', threshold: 0});
document.querySelectorAll('main section[id]').forEach(section => observer.observe(section));
