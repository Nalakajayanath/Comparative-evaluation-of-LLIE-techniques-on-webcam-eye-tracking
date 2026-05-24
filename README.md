# Comparative Evaluation of LLIE Techniques for Webcam-Based Gaze Tracking

## Run order (four commands)

From the **project root**, after placing MPIIGaze in `data/original/MPIIGaze/`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

python scripts/setup_environment.py
python data/extract_normalized_images.py
python data/simulate_low_light.py
python evaluation/run_all.py
```

| Step | Script | What it does |
|------|--------|----------------|
| 1 | `scripts/setup_environment.py` | pip install, clone `vendor/`, licenses, **download gaze weights** |
| 2 | `data/extract_normalized_images.py` | `.mat` → `left_0001.jpg` / `right_0001.jpg` |
| 3 | `data/simulate_low_light.py` | Build `data/low_light_simulated/` |
| 4 | `evaluation/run_all.py` | LLIE + gaze eval → `results/results.csv` |

Optional plots: `python evaluation/generate_summary.py`, `plot_mean_error.py`, …

## Details

- **pip:** `requirements.txt`
- **git clones + licenses:** `scripts/setup_environment.py` → `THIRD_PARTY_NOTICES.md`
- **Gaze model:** inlined ResNet-preact (`gaze_model/`), weights from [ptgaze release](https://github.com/hysts/pytorch_mpiigaze_demo)
