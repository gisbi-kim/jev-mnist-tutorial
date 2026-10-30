'use strict';
const modelRows = [...document.querySelectorAll('[data-model-row]')];
const modelSearch = document.getElementById('model-search');
const modelFilter = document.getElementById('model-filter');
function filterModels() {
  const query = modelSearch.value.trim().toLocaleLowerCase();
  const mode = modelFilter.value;
  let visible = 0;
  for (const row of modelRows) {
    const matchesMode = mode === 'all' || (mode === 'vision' ? row.dataset.vision === 'yes' : row.dataset.mode.split(' ').includes(mode));
    row.hidden = !(matchesMode && row.textContent.toLocaleLowerCase().includes(query));
    if (!row.hidden) visible++;
  }
  document.getElementById('model-count').textContent = `${visible} / ${modelRows.length}개 표시`;
  document.getElementById('model-empty').hidden = visible !== 0;
}
if (modelSearch && modelFilter) {
  modelSearch.addEventListener('input', filterModels);
  modelFilter.addEventListener('change', filterModels);
  filterModels();
}
const guideLinks = [...document.querySelectorAll('aside nav a')];
const guideObserver = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    guideLinks.forEach(link => {
      const active = link.hash === '#' + entry.target.id;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
}, {rootMargin: '-15% 0px -65% 0px'});
document.querySelectorAll('main section[id]').forEach(section => guideObserver.observe(section));
let messageTimer;
document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const status = document.getElementById('copy-status');
    try {
      await navigator.clipboard.writeText(document.getElementById(button.dataset.copy).textContent);
      status.textContent = '코드를 복사했습니다.';
    } catch {
      status.textContent = '자동 복사가 안 됩니다. 코드 내용을 선택해서 복사해 주세요.';
    }
    clearTimeout(messageTimer);
    messageTimer = setTimeout(() => { status.textContent = ''; }, 3000);
  });
});
