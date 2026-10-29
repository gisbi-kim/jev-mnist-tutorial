# MNIST Decision Lab

Image + natural-language current state → a genuine Unsloth Clef decision head → one of `-1`, `0`, `+1`. Applying `0` terminates an episode. The image target label is never passed to inference. No OCR/rule fallback is used.

## Actual status — 2026-10-29

The code and local GUI were authored in a restricted session. **MNIST download, package installation, model training and model evaluation have NOT completed. No trained checkpoint or measured model accuracy is included.** The host has an RTX 5070 Ti Laptop GPU (12 GB), but its active Python environment lacks torch and unsloth. Network DNS and WSL access were blocked in the execution environment. Downloads is outside the session's writable roots.

The unit/API tests exercise task semantics, data split logic, IDX parsing and web behavior using explicitly synthetic fixtures/stubs. They are not evidence that GPU training works. Unsloth integration follows the upstream APIs inspected on 2026-10-29 and still needs an actual GPU run.

Verification: 11 unit/API tests passed, Python files compiled, JavaScript syntax checked. A local server started, but live HTTP access was denied (`WinError 10013`) and the browser timed out. Browser appearance/interactions are therefore unverified; the temporary server was stopped. All UI metrics remain explicitly unmeasured.

## Run in your Downloads folder

Extract this project to `C:\Users\USER\Downloads\mnist-decision-lab` and run in a normal PowerShell terminal:

```powershell
cd "$env:USERPROFILE\Downloads\mnist-decision-lab"
.\setup_and_train.ps1
```

This creates an isolated `.venv`, installs CUDA PyTorch and Unsloth, downloads original MNIST and Qwen3.5-0.8B, generates GT, trains, evaluates, then starts the GUI at http://127.0.0.1:7864. It stops at the first failure. Internet access and a compatible Python/CUDA stack are required. The current official Windows guide uses Python 3.12; Python 3.13 compatibility must be established by installation, not assumed. If installation fails, create the venv with Python 3.12 before rerunning. No API key or paid service is used. Downloads may be several GB; runtime and peak VRAM have not been measured.

For a first GPU smoke run:

```powershell
.\setup_and_train.ps1 -Steps 10 -TrainImages 50 -TestImages 5
```

A smoke run checks execution only, not learning quality. Default: 600 optimizer steps, 1,200 training images, 100 validation images and 100 official test images. All outputs are stored under `runs/<timestamp>`. Each successful run saves its effective package versions and pip freeze. If the installed package lacks `predict(images=...)`, update to an Unsloth version with multimodal decision support; the script refuses to silently train a text-only model.

GUI only (works without torch; inference remains disabled until training succeeds):

```powershell
python -m pip install -r requirements-gui.txt
python server.py
```

## Data and GT

- Original MNIST: 60,000 training / 10,000 test images, checked against the standard MD5 checksums.
- Hold out 500 images per digit from original training: 55,000 train / 5,000 validation / 10,000 official test.
- Expand each image with starting states 0–9: 550,000 / 50,000 / 100,000 decision rows.
- `sign(target - current)` determines GT automatically; text does not contain the target label.
- Four English/Korean training phrasings. Additional held-out wording evaluated separately.
- Training subsets are selected by image ID, then decision classes are balanced through resampling.
- More phrasings and starts are not new independent handwriting examples. GT expansion is cheap, but original image diversity is finite.
- `data/*.jsonl` are image-ID manifests. Training reconstructs PIL image inputs, `state`, `questions`, and `gold`.

## Evaluation

- Per-action confusion matrix, action accuracy and macro recall.
- Exact deterministic rollouts for all ten starting states; episode success requires the correct STOP.
- Wrong-stop rate, cycles, out-of-range failures and turn limit; no clipping erroneous actions.
- NLL, Brier, ten-bin ECE and inference latency; temperature calibrated on validation only.
- Fresh decision head before training versus matched post-training subset (this is NOT a zero-shot VLM baseline).
- Separate full test subset, unseen wording, shuffled-image and blank-image ablations.
- Rollouts cache per-image state predictions: there is no history beyond the current state and the policy is deterministic. This produces the same rollouts without repeated GPU calls.
- No independent CNN comparison has been trained or included.

## GUI behavior

Draw one digit, upload an image, or choose a real MNIST test sample. Enter e.g. `2에서 시작해`, `삼에서 시작해`, or `start from seven`. A restricted deterministic parser extracts the initial number for the environment; the original sentence is still passed to the model for the first decision. Subsequent turns supply the updated state in Korean. This does not claim arbitrary natural-language parsing.

Uploaded drawings are normalized to bright ink on a black 28×28 canvas and resized to 112×112. Real MNIST samples use exactly the same preprocessing as training. Out-of-distribution user handwriting may perform differently. The UI reports a model STOP, not ground-truth success, because uploaded images have no known label.

## Sources inspected

- https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth
- https://github.com/unslothai/unsloth/blob/main/unsloth/models/decision.py
- https://github.com/unslothai/unsloth/blob/main/unsloth/models/decision_from_lm.py
- https://unsloth.ai/docs/get-started/install/windows-installation
- https://huggingface.co/unsloth/Qwen3.5-0.8B

Unsloth decision source files carry AGPL-3.0-only; check upstream terms before redistribution or service deployment.
