"""Stage tracked public files for Pages, preserving site/../raw relative paths."""
from pathlib import Path
import shutil, subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / '_site'
if OUTPUT.exists():
    raise SystemExit('_site already exists; use a fresh checkout or a new build directory.')
OUTPUT.mkdir()
paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf8').split('\0')
for name in paths:
    if not name or name.startswith('.github/') or name in {'.gitignore', '.gitattributes'}:
        continue
    source, target = ROOT / name, OUTPUT / name
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
print(f'Pages files prepared: {OUTPUT}')
