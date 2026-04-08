# Vision Imaging Lab

`vision-imaging-lab` is a basics-focused computer vision and image-processing tutorial repo.

This repository is intentionally kept as a learning lab:
- build concepts step by step
- understand what the code is doing
- connect visual effects with quantitative metrics
- practice organizing small experiments into reusable modules

---

## What This Repo Covers

This repo focuses on foundational topics:

- image loading and grayscale conversion
- brightness and contrast
- histogram analysis
- brightness shift
- contrast scaling
- histogram equalization
- CLAHE
- blur and sharpness
- noise and denoising
- small-dataset analysis
- camera calibration basics

---

## Main Learning Notebooks

### `notebooks/01_image_quality_basics.ipynb`

Covers:
- image representation
- brightness and contrast
- histograms
- intensity transforms
- enhancement methods
- blur and noise
- denoising tradeoffs
- small-dataset reasoning

### `notebooks/02_camera_calibration.ipynb`

Covers:
- chessboard pattern definition
- inner corners vs square counts
- corner detection
- object points and image points
- camera calibration
- camera matrix and distortion coefficients
- undistortion
- interpretation of calibration results

---

## Main Reusable Modules

### `src/image_utils.py`
Basic helpers for:
- loading images
- grayscale conversion
- project paths

### `src/image_metrics.py`
Metrics for:
- brightness
- contrast
- min/max intensity
- sharpness
- estimated noise
- MAE / MSE / PSNR

### `src/image_histograms.py`
Helpers for:
- grayscale histogram computation

### `src/image_transforms.py`
Transforms for:
- brightness shift
- contrast scaling
- centered contrast
- histogram equalization
- CLAHE
- Gaussian / median / bilateral denoising

---

## Repository Structure

```text
vision-imaging-lab/
|-- data/
|   |-- calibration/
|   |-- calibration_synthetic_backup/
|   `-- raw/
|-- notebooks/
|-- outputs/
`-- src/
```

---

## Current Datasets

### Image-quality mini dataset
Used for learning small-dataset analysis:
- `data/raw/blurry/`
- `data/raw/daylight/`
- `data/raw/high_contrast/`
- `data/raw/low_light/`

### Calibration dataset
Used in notebook02:
- real OpenCV sample chessboard images in `data/calibration/`

---

## How To Run

Use your conda environment, then run scripts from the project root.

Examples:

```powershell
python src/shift_brightness.py
python src/scale_contrast.py
python src/centered_contrast.py
python src/clahe_experiment.py
python src/denoising_experiment.py
python src/analyze_dataset.py
python src/compare_clahe_vs_centered_contrast.py
```

To open the notebooks:

```powershell
jupyter notebook
```

---

## What This Repo Teaches

By working through this repo, a beginner should learn that:

- brightness, contrast, sharpness, and noise are different image properties
- histograms often explain image behavior better than a single summary number
- clipping causes information loss
- enhancement is not always improvement
- denoising is a tradeoff, not magic
- metrics and visual inspection should be used together
- calibration needs consistent pattern geometry and successful corner detection

---

## Scope

This repository is intentionally limited to fundamentals and guided practice.

Its purpose is:
- tutorial learning
- concept reinforcement
- notebook-based review
- small reusable building blocks

It should stay focused on basics.
