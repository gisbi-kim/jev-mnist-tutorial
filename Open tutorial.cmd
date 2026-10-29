@echo off
cd /d "%~dp0"
echo Tutorial: http://127.0.0.1:7865/site/index.html
echo Keep this window open while viewing. Press Ctrl+C to stop.
python -m http.server 7865 --bind 127.0.0.1 --directory "%~dp0."
