# Vision Imaging Lab

`vision-imaging-lab` is a step-by-step computer vision and image-processing project.

The goal is to grow from beginner understanding to more advanced imaging practice by building the pipeline manually:
- learn image representation
- measure image properties
- analyze histograms
- study transforms
- compare blur, noise, and denoising methods
- understand camera calibration
- build reusable code from experiments

It is designed to help a beginner grow step by step toward more advanced imaging and computer vision practice.

---

## Project Structure

This project has two stages:

### Stage 1: Learning Tutorial
Build understanding manually, one concept at a time, and turn that learning into reusable code and clear notebooks.

### Stage 2: Ambitious and Valuable Practice
Use stronger datasets, tighter project scope, and more serious evaluation to build a portfolio-quality mini project.

Stage 1 is now complete enough to close as a learning checkpoint.

---

## What Stage 1 Covers

### Notebook 01: Image Quality Basics

Topics covered:
- image loading and grayscale conversion
- brightness and contrast measurement
- histogram analysis
- brightness shift
- contrast scaling
- centered contrast
- histogram equalization
- CLAHE
- blur and sharpness
- noise estimation
- denoising tradeoffs
- small-dataset analysis
- category-based enhancement recommendations

Main lessons:
- brightness, contrast, sharpness, and noise are different image properties
- histograms explain intensity distribution better than one summary number
- clipping causes information loss
- quantization can create empty histogram bins after transforms
- centered contrast is more meaningful than naive contrast scaling
- histogram equalization can over-enhance highlights
- soft CLAHE gives more balanced local enhancement
- denoising is a tradeoff, not magic

### Notebook 02: Camera Calibration Basics

Topics covered:
- chessboard pattern definition using inner corners
- corner detection
- inspection of corner ordering
- failed-detection inspection
- object points and image points
- camera calibration
- camera matrix and distortion coefficients
- undistortion
- interpretation of calibration outputs

Main lessons:
- calibration uses inner corners, not square counts
- object points are defined in the board’s local coordinate system
- the same object-point grid is reused for all images
- calibration estimates one shared camera model and one pose per successful image
- not every calibration image has to succeed
- realistic undistortion can be subtle rather than dramatic

---

## Best Findings So Far

### Contrast and enhancement
- naive contrast scaling increased brightness too much
- centered contrast scaling preserved brightness better and produced a more meaningful contrast increase
- histogram equalization was too aggressive for the current test image
- soft CLAHE gave a better balance between local detail and natural appearance

### Blur, noise, and denoising
- blur changed brightness very little but strongly reduced Laplacian sharpness
- noise falsely inflated sharpness metrics because it adds high-frequency variation
- Gaussian denoise reduced noise strongly but oversmoothed the image
- bilateral denoise gave the best balance in the current denoising experiment

### Small-dataset analysis
- low-light images were clearly the darkest category
- blurry images were clearly the least sharp category
- high-contrast images showed the strongest contrast
- means and medians can tell different stories when one image dominates a category
- enhancement recommendations should depend on image condition, not one universal rule

### Calibration
- a real OpenCV chessboard sample set worked much better for learning realistic calibration behavior
- calibration succeeded on a useful subset of images
- the estimated intrinsics and distortion were numerically plausible
- the undistortion effect was subtle but believable

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

### Important notebooks
- `notebooks/01_image_quality_basics.ipynb`
- `notebooks/02_camera_calibration.ipynb`

### Important reusable modules
- `src/image_utils.py`
- `src/image_metrics.py`
- `src/image_histograms.py`
- `src/image_transforms.py`

### Important analysis scripts
- `src/analyze_dataset.py`
- `src/plot_dataset_summary.py`
- `src/clahe_dataset_evaluation.py`
- `src/plot_clahe_category_summary.py`
- `src/compare_clahe_vs_centered_contrast.py`
- `src/inspect_outliers.py`

---

## Current Datasets

### Image-quality mini dataset
Used for Stage 1 category analysis:
- `data/raw/blurry/`
- `data/raw/daylight/`
- `data/raw/high_contrast/`
- `data/raw/low_light/`

### Calibration dataset
Used in notebook02:
- real OpenCV sample chessboard images in `data/calibration/`
- synthetic calibration images kept in `data/calibration_synthetic_backup/`

---

## How to Run

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

## Learning Path

This project is meant to function as a growth ladder:

- beginner: understand images, grayscale, brightness, contrast, histograms, blur, and noise
- intermediate: build reusable utilities, metrics, transforms, and structured experiments
- advanced direction: compare methods critically, reason about tradeoffs, analyze datasets, and understand camera geometry

The long-term goal is to move from learning scripts toward a small but serious image-processing and computer-vision toolkit.

---

## What Comes Next

Stage 1 is complete enough to close.

Stage 2 will begin with:
- choosing a strong open-source dataset
- defining a focused computer vision task
- planning a more ambitious, portfolio-quality mini project

The final goal is for this repository to become both:
- a strong learning record
- a portfolio-ready computer vision project
