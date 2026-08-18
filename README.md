# FinFET Pattern Localization

A synthetic-data generation and deep-learning pipeline for localizing a reference FinFET pattern within a larger search image.

The project generates paired synthetic SEM-style images consisting of:

- A small **reference image**
- A larger **search image**
- Ground-truth `(x, y)` coordinates of the reference pattern within the search image

A Siamese CNN with cross-correlation is then trained to learn the correspondence between the reference and search images and predict the pattern location.

---

## 1. Project Overview

```text
Synthetic Dataset Generation
            |
            v
   Reference + Search Images
            |
            v
      Siamese CNN
            |
            v
   Multi-scale Feature Extraction
            |
            v
      Cross-Correlation
            |
            v
       Response Heatmap
            |
            v
       Soft-Argmax
            |
            v
      Predicted (x, y)
```

The dataset generator supports multiple DRAM and FinFET structure presets. For the FinFET experiments, the primary architectures used are:

- FinFET 10 nm
- FinFET 14 nm
- FinFET 22 nm

---

## 2. Requirements

### Python Version

This project was developed and tested using:

```text
Python 3.10.0
```

Python 3.10 is recommended for reproducibility.

---

## 3. Create the Virtual Environment

From the project root directory:

```powershell
python -m venv venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Verify the Python version:

```powershell
python --version
```

Expected:

```text
Python 3.10.0
```

---

## 4. Install Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

## 5. PyTorch and CUDA Compatibility

The model is designed to run on an NVIDIA GPU when CUDA is available.

The development environment used:

```text
PyTorch    : 2.1.0+cu118
Torchvision: 0.16.0+cu118
CUDA       : 11.8
GPU        : NVIDIA GeForce RTX 3050 6GB
```

The NVIDIA driver may report a newer CUDA version. For example, the development system reported CUDA 13.0 through `nvidia-smi`.

This does **not** mean that PyTorch must use CUDA 13.0. The `+cu118` PyTorch package contains the CUDA 11.8 runtime required by that PyTorch build, while the installed NVIDIA driver provides backward compatibility.

If the standard `requirements.txt` installation cannot find:

```text
torch==2.1.0+cu118
```

install the CUDA-enabled PyTorch packages separately:

```powershell
python -m pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Verify the installation:

```powershell
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected GPU configuration:

```text
PyTorch: 2.1.0+cu118
Torchvision: 0.16.0+cu118
CUDA: True
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

If CUDA is unavailable, the model can fall back to CPU execution, but GPU execution is recommended for training.

---

## 6. Synthetic Dataset Generation

The dataset is generated using:

```text
generate_dataset.py
```

The generator creates paired:

```text
Reference image
Search image
Ground-truth coordinates
```

and records generation parameters in a CSV manifest.

### Basic Command

For example, to generate 3 FinFET 10 nm samples:

```powershell
python generate_dataset.py --num-samples 3 --architectures finfet_10nm --output-dir ./output --seed 42
```

For a test split:

```powershell
python generate_dataset.py --num-samples 3 --architectures finfet_10nm --split test --output-dir ./output --seed 42
```

### Training Dataset

The main training dataset used in the experiments was generated with:

```powershell
python generate_dataset.py --num-samples 3000 --architectures finfet_10nm finfet_14nm finfet_22nm --split train --output-dir ./output_3000 --seed 42
```

This generates:

```text
output_3000/
└── train/
    ├── reference/
    ├── search/
    └── manifest.csv
```



## 7. Dataset Contents

Each generated split contains:

```text
reference/
    00000.png
    00001.png
    ...

search/
    00000.png
    00001.png
    ...

manifest.csv
```

The manifest records information including:

```text
id
reference_path
search_path
gt_x
gt_y
gt_box_x
gt_box_y
gt_box_w
gt_box_h
architecture
scale_ratio
rotation_deg
noise and imaging parameters
seed
```

The `gt_x` and `gt_y` fields represent the ground-truth center coordinates of the reference pattern in the search image.

---

## 8. Model Training

The deep-learning implementation is contained in the `ml/` directory.

The main training script is:

```text
ml/train.py
```

Run training from the project root:

```powershell
python -m ml.train
```

Running it as a module ensures that project imports are resolved correctly from the repository root.

The training pipeline includes:

- Siamese CNN feature extraction
- Fine/coarse feature processing
- Cross-correlation
- Response heatmap generation
- Soft-argmax localization
- Localization loss
- Validation using Localization Error
- Best-model checkpoint saving

---

## 9. Model Evaluation

The evaluation script is:

```text
ml/test.py
```

Run:

```powershell
python -m ml.test
```

The evaluation reports:

```text
Mean CLE
Median CLE
Accuracy @ 1 px
Accuracy @ 2 px
Accuracy @ 5 px
Accuracy @ 10 px
```

CLE is the Euclidean localization error between the predicted position and ground-truth position.

---

## 10. Single-Pair Localization

The project also provides:

```text
localize.py
```

for inference on a single reference/search image pair.

The script accepts:

```text
Reference image path
Search image path
```

Example:

```powershell
python localize.py --reference ".\smoke_test\reference.png" --search ".\smoke_test\search.png"
```

The model:

1. Loads the reference image.
2. Loads the search image.
3. Extracts CNN features.
4. Performs cross-correlation.
5. Generates a response heatmap.
6. Applies soft-argmax localization.
7. Converts feature-map coordinates back to image coordinates.
8. Reports the predicted `(x, y)` position.

---

## 11. Reproducibility

The dataset generator supports a random seed:

```powershell
--seed 42
```

Example:

```powershell
python generate_dataset.py --num-samples 3000 --architectures finfet_10nm finfet_14nm finfet_22nm --split train --output-dir ./output_3000 --seed 42
```

Using a fixed seed helps reproduce the dataset-generation procedure.

---

## 12. Recommended Fresh-Clone Setup

```powershell
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>

python -m venv venv
.\venv\Scripts\Activate.ps1

python --version

python -m pip install -r requirements.txt
```

If using the CUDA 11.8 PyTorch configuration:

```powershell
python -m pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Generate a small smoke-test dataset:

```powershell
python generate_dataset.py --num-samples 3 --architectures finfet_10nm --split test --output-dir ./smoke_output --seed 42
```

Run model inference:

```powershell
python localize.py --reference ".\smoke_test\reference.png" --search ".\smoke_test\search.png"
```

---

## 13. Project Structure

```text
finfet-pattern-localization/
|
├── generate_dataset.py
├── localize.py
├── requirements.txt
├── README.md
├── .gitignore
|
├── src/
│   ├── pipeline.py
│   ├── presets.py
│   └── sem_imaging.py
|
├── ml/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── test_cpy.py
|
├── model/
│   └── siamese_cnn.pt
|
└── smoke_test/
    ├── reference.png
    └── search.png
```

---

## 14. Notes

- Python 3.10.0 is recommended for the tested environment.
- CUDA-enabled PyTorch is recommended when an NVIDIA GPU is available.
- The CUDA version reported by `nvidia-smi` refers to the maximum CUDA version supported by the installed NVIDIA driver; it does not have to match the CUDA runtime bundled with the PyTorch wheel.
- The full synthetic dataset does not need to be stored in the repository. It can be regenerated using `generate_dataset.py`.
- Generated datasets and virtual environments should remain outside version control.
- Ground-truth coordinates are stored in the generated `manifest.csv`.

---

## 15. Quick Reference

### Install

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Generate Dataset

```powershell
python generate_dataset.py --num-samples 3000 --architectures finfet_10nm finfet_14nm finfet_22nm --split train --output-dir ./output_3000 --seed 42
```

### Train

```powershell
python -m ml.train
```

### Evaluate

```powershell
python -m ml.test
```

### Single-Pair Inference

```powershell
python localize.py --reference ".\smoke_test\reference.png" --search ".\smoke_test\search.png"
```
