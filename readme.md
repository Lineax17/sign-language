# ASL-Alphabet Recognizer

ASL alphabet classification project with split workflows:

- Training and data processing in `apps/training` (target Python 3.13)
- Live inference with webcam in `apps/live` (Python 3.11)

The project uses `uv` for Python version and dependency management.

## Project layout

- `apps/training/src/processing`: data split + model conversion
- `apps/training/src/train`: training entrypoints
- `apps/training/src/eval`: evaluation utilities (confusion matrix)
- `apps/training/src/diagnostics`: GPU diagnostics
- `apps/live/src/run`: live webcam inference
- `apps/live/src/test`: single-image inference smoke test
- `apps/live/src/config`: label mapping
- `models`: shared exported models
- `data`: shared datasets

## Dataset source

The dataset is GPLv2 licensed:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download

## Setup with uv

Run all commands from repository root unless noted.

### 1) Training environment (Python 3.13 target)

```bash
cd apps/training
uv python install 3.13
uv sync
```

If dependency resolution fails on Python 3.13 (for example TensorFlow wheel availability), use the highest compatible version instead:

```bash
uv python install 3.12
uv sync --python 3.12
```

### 2) Live environment (Python 3.11)

```bash
cd apps/live
uv python install 3.11
uv sync
```

## Training workflow

### Prepare split dataset

```bash
cd apps/training
uv run python src/processing/split_data.py
```

### Check CUDA visibility

```bash
cd apps/training
uv run python src/diagnostics/test_cuda.py
```

### Train model

```bash
cd apps/training
uv run python src/train/training_freezed.py
uv run python src/train/training_unfreezed.py
```

### Convert Keras to TFLite

```bash
cd apps/training
uv run python src/processing/convert_model.py
```

### Generate confusion matrix

```bash
cd apps/training
uv run python src/eval/confusion_matrix.py
```

## Live inference workflow

### Run webcam inference

```bash
cd apps/live
uv run python src/run/real_time_with_text.py
```

### Run single-image smoke test

```bash
cd apps/live
uv run python src/test/test_single_image.py
```
