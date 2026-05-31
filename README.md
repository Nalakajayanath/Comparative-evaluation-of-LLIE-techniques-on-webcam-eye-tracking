# Comparative Evaluation of Low-Light Image Enhancement Techniques for Webcam-Based Eye Tracking

MSc Advanced Software Engineering Research Project

**Author:** Nalaka Jayanath  
**University of Westminster ID:** W2106771  
**IIT Student ID:** 20240319  
**Supervisor:** Mr. Lakshan Costa

---

## Overview

This research evaluates multiple Low-Light Image Enhancement (LLIE) techniques for improving webcam-based gaze estimation under low-light conditions.

The study uses the MPIIGaze dataset, simulated low-light environments, and a pretrained MPIIGaze ResNet-preact gaze estimation model to compare LLIE methods in terms of:

- Gaze estimation accuracy
- Processing latency
- Frames Per Second (FPS)

Evaluated methods:

- Histogram Equalization (HE)
- CLAHE
- SSR
- MSR
- MSRCR
- Zero-DCE
- EnlightenGAN
- MIRNet

---

## Dataset

This project uses the MPIIGaze dataset.

Place the dataset in:

```text
data/original/MPIIGaze/
```

---

## Quick Start

Create a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Setup dependencies and model files:

```powershell
python scripts/setup_environment.py
```

Extract normalized eye images:

```powershell
python data/extract_normalized_images.py
```

Generate simulated low-light datasets:

```powershell
python data/simulate_low_light.py
```

Run all experiments:

```powershell
python evaluation/run_all.py
```

---

## Experimental Pipeline

1. Load MPIIGaze evaluation samples
2. Generate simulated low-light conditions
3. Apply LLIE methods
4. Predict gaze using MPIIGaze ResNet-preact
5. Calculate angular error
6. Measure latency and FPS
7. Compare results

---

## Evaluation Metrics

- Mean Angular Error (°)
- Average Latency (ms)
- Frames Per Second (FPS)

Lower angular error indicates better gaze estimation accuracy.

---

## Main Findings

- EnlightenGAN achieved the best gaze estimation accuracy.
- Histogram Equalization (HE) achieved the best real-time performance.
- LLIE preprocessing can improve gaze estimation under low-light conditions.
- Accuracy improvements must be balanced against computational cost.

---

## Repository Structure

```text
IMPLEMENTATION/
│
├── data/
├── evaluation/
├── gaze_model/
├── LLIEs/
├── models/
├── results/
├── scripts/
└── requirements.txt
```

---

## Technologies

- Python
- PyTorch
- OpenCV
- NumPy
- SciPy
- ONNX Runtime

---

## Citation

```bibtex
@mastersthesis{jayanath2026llie,
  author = {Nalaka Jayanath},
  title = {Comparative Evaluation of Low-Light Image Enhancement Techniques to Improve Webcam-Based Eye Tracking},
  school = {University of Westminster},
  year = {2026}
}
```

---

## Acknowledgements

- University of Westminster
- Informatics Institute of Technology (IIT)
- MPIIGaze Dataset Authors
- Zero-DCE Authors
- EnlightenGAN Authors
- MIRNet Authors
- Mr. Lakshan Costa
