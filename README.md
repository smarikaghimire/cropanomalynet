# CropAnomalyNet

Unsupervised crop disease detection on PlantVillage maize via reconstruction-based anomaly detection and pretrained-feature methods, with diffusion-based synthetic data augmentation.

**Status:** NB1 (autoencoder baseline) and NB2 (PatchCore + brightness ablation) complete. NB3–NB4 in progress.

## Motivation

Supervised crop disease classifiers require thousands of labeled examples per class. We reframe the problem as anomaly detection: train only on healthy leaf images, then flag deviations as diseased. This mirrors industrial defect inspection — abundant "normal" samples, very few labeled defects.

## Pipeline

| Notebook | Method                                         | Status  | Test AUROC |
| -------- | ---------------------------------------------- | ------- | ---------- |
| NB1      | Convolutional Autoencoder (MSE reconstruction) | Done    | 0.9942     |
| NB2      | PatchCore (WR50 layer-2+3, mean-aggregated)    | Done    | **0.9995** |
| NB3      | Stable Diffusion synthetic disease generation  | Planned | —          |
| NB4      | PatchCore + synthetic augmentation ablation    | Planned | —          |

## Dataset

PlantVillage (Hughes & Salathé, 2015), maize subset, color version only. Four classes:

| Class                            | Count |
| -------------------------------- | ----- |
| Corn healthy                     | 1,162 |
| Corn Common rust                 | 1,192 |
| Corn Cercospora (Gray leaf spot) | 513   |
| Corn Northern Leaf Blight        | 985   |

Splits (canonical, in `results/nb1/splits.csv`):

- Train: 929 healthy
- Val: 116 healthy
- Test: 117 healthy + 2,690 diseased (all 3 disease classes)

NB2 reuses the exact same splits as NB1 (deterministic via shared random seed) so all results are directly comparable.

## Results — PlantVillage maize

Per-class test AUROC (117 healthy + 2,690 diseased):

| Class                | Brightness-only | CAE (NB1)  | PatchCore-mean (NB2) |
| -------------------- | --------------- | ---------- | -------------------- |
| Common rust          | 0.9175          | 0.9996     | **1.0000**           |
| Cercospora           | 0.6853          | 0.9948     | **0.9994**           |
| Northern Leaf Blight | 0.6849          | 0.9874     | **0.9990**           |
| **Overall**          | **0.7880**      | **0.9942** | **0.9995**           |

### Key findings

1. **PatchCore-mean matches or exceeds the CAE on every class** while operating on ImageNet-pretrained features. Largest gain is on Northern Leaf Blight (+0.0116), the hardest class for both methods.

2. **Score aggregation matters substantially for diffuse anomalies.** PatchCore's standard `max patch distance` (calibrated for localized industrial defects on benchmarks like MVTec) achieves only 0.9536 on this dataset. `mean` aggregation recovers full performance. Plant disease is spread across the leaf rather than concentrated on a single patch — max picks up edge artifacts, mean integrates the diffuse signal.

   | Aggregation                  | Overall AUROC |
   | ---------------------------- | ------------- |
   | max (PatchCore default)      | 0.9536        |
   | mean                         | **0.9995**    |
   | top-10% mean                 | 0.9967        |
   | top-1% mean                  | 0.9817        |
   | center max (4-pixel margin)  | 0.9904        |
   | center mean (4-pixel margin) | 0.9995        |

3. **Brightness alone is insufficient** (overall 0.7880). Color carries most of the Common Rust signal (0.9175) but is barely above chance for Cercospora and Northern Blight (~0.685). Both learned methods are doing substantial real texture-based detection on the harder classes, not just exploiting a color shortcut.

4. PlantVillage's clean studio backgrounds and uniform lighting likely inflate both methods' AUROC compared to real-field deployment. PlantDoc evaluation (planned) will test which signal generalizes.

## NB1 — Convolutional Autoencoder

Vanilla CAE (1.08M params) trained on 929 healthy maize images for 45 epochs with early stopping. Output in `[0, 1]` via sigmoid; MSE loss; Adam at lr=1e-3 with `ReduceLROnPlateau`.

Per-pixel reconstruction-error maps localize to lesion regions (`results/nb1/cae_reconstructions.png`), indicating genuine texture-based reconstruction failure rather than just global brightness mismatch.

Artifacts in `results/nb1/`:

- `cae_score_histogram.png`, `cae_roc.png`, `cae_reconstructions.png`
- `cae_test_scores.csv`, `splits.csv`
- Checkpoint: `checkpoints/cae_best.pt` (4 MB, included)

## NB2 — PatchCore

WideResNet50 ImageNet-pretrained backbone, frozen. Features from layer 2 (28×28×512) and layer 3 (14×14×1024) captured via forward hooks; layer 3 upsampled to 28×28 via bilinear interpolation, concatenated channel-wise with layer 2, then 3×3 average-pooled for neighborhood-aware patches (Roth et al. 2022). Each image yields 784 patches × 1536 dims.

Memory bank: 728,336 patches from 929 healthy training images. Reduced to 10% (72,833 patches) via greedy farthest-point coreset sampling on Johnson-Lindenstrauss-projected (128-dim) features for tractable distance computation.

Inference: `torch.cdist` between test patches and coreset bank, minimum over the coreset gives a 28×28 anomaly map per image, **mean** over the 784 patches gives the image-level anomaly score. Score-reduction choice (mean vs max) was made by ablation — see Key Finding 2 above.

Artifacts in `results/nb2/`:

- `patchcore_score_histogram.png`, `patchcore_roc.png`, `patchcore_anomaly_maps.png`
- `patchcore_test_scores.csv` (columns: `score_max`, `score_mean`)
- `brightness_baseline_scores.csv` (mean-RGB-distance baseline for ablation)
- Coreset memory bank (~447 MB) regenerable from notebook cells 4–7; excluded via `.gitignore`.

## Reproducing

### NB1

1. Open `notebooks/nb1_autoencoder.ipynb` on Kaggle.
2. Attach the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).
3. Settings → Accelerator → GPU T4 x2 (or P100).
4. Run all cells. End-to-end runtime ~3 minutes.

### NB2

1. Open `notebooks/nb2_patchcore.ipynb` on Kaggle.
2. Attach the same PlantVillage dataset.
3. Settings → Accelerator → GPU T4 x2 (or P100).
4. Run all cells. End-to-end runtime ~15 minutes (memory-bank construction ~15 s, coreset selection ~10 min, inference ~3 min, brightness baseline ~15 s).

## Repository structure

```
cropanomalynet/
├── notebooks/
│   ├── nb1_autoencoder.ipynb
│   └── nb2_patchcore.ipynb
├── results/
│   ├── nb1/
│   │   ├── cae_score_histogram.png
│   │   ├── cae_roc.png
│   │   ├── cae_reconstructions.png
│   │   ├── cae_test_scores.csv
│   │   └── splits.csv
│   └── nb2/
│       ├── patchcore_score_histogram.png
│       ├── patchcore_roc.png
│       ├── patchcore_anomaly_maps.png
│       ├── patchcore_test_scores.csv
│       └── brightness_baseline_scores.csv
├── checkpoints/
│   └── cae_best.pt          # 4 MB — small enough to commit
├── .gitignore
└── README.md
```

## Citation

This work is in progress and does not yet have an associated publication. Code may be referenced as:

Smarika Ghimire, Avinash Gautam, "CropAnomalyNet: Unsupervised crop disease detection," 2026, https://github.com/smarikaghimire/cropanomalynet

## License

MIT
