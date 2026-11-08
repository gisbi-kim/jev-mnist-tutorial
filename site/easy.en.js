'use strict';
const target = document.getElementById('target');
const current = document.getElementById('current');
function updateRule() {
  const t = Number(target.value), s = Number(current.value), a = Math.sign(t - s);
  document.getElementById('targetValue').textContent = t;
  document.getElementById('currentValue').textContent = s;
  document.getElementById('action').textContent = a > 0 ? '+1' : a < 0 ? '−1' : '0';
  document.getElementById('explanation').textContent = a === 0 ? `The current number equals the target ${t}. Stop here.` : `The target ${t} is ${a > 0 ? 'greater than' : 'less than'} the current number ${s}, so ${a > 0 ? 'add' : 'subtract'} 1. The next number is ${s + a}.`;
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
  document.getElementById('pathText').textContent = `Following the correct actions: ${path.join(' → ')} → STOP (0)`;
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
