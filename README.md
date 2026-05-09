# CropAnomalyNet

Unsupervised crop disease detection on PlantVillage maize via reconstruction-based anomaly detection and pretrained-feature methods, with diffusion-based synthetic data augmentation.

**Status:** NB1 (autoencoder baseline) complete. NB2–NB4 in progress.

## Motivation

Supervised crop disease classifiers require thousands of labeled examples per class. We reframe the problem as anomaly detection: train only on healthy leaf images, then flag deviations as diseased. This mirrors industrial defect inspection — abundant "normal" samples, very few labeled defects.

## Pipeline

| Notebook | Method                                         | Status      | Test AUROC |
| -------- | ---------------------------------------------- | ----------- | ---------- |
| NB1      | Convolutional Autoencoder (MSE reconstruction) | Done        | 0.9942     |
| NB2      | PatchCore + PaDiM (WideResNet50 features)      | In progress | —          |
| NB3      | Stable Diffusion synthetic disease generation  | Planned     | —          |
| NB4      | PatchCore + synthetic augmentation ablation    | Planned     | —          |

## Dataset

PlantVillage (Hughes & Salathé, 2015), maize subset, color version only. Four classes:

| Class                            | Count |
| -------------------------------- | ----- |
| Corn healthy                     | 1,162 |
| Corn Common rust                 | 1,192 |
| Corn Cercospora (Gray leaf spot) | 513   |
| Corn Northern Leaf Blight        | 985   |

Splits (in `results/nb1/splits.csv`):

- Train: 929 healthy
- Val: 116 healthy
- Test: 117 healthy + 2,690 diseased (all 3 disease classes)

## NB1 Results

Vanilla convolutional autoencoder (1.08M params) trained on 929 healthy maize images for 45 epochs with early stopping.

| Class                  | AUROC  |
| ---------------------- | ------ |
| Overall (all diseases) | 0.9942 |
| Common rust            | 0.9996 |
| Cercospora             | 0.9948 |
| Northern Leaf Blight   | 0.9874 |

**Diagnosis.** Per-pixel error maps localize to lesion regions (genuine texture-based reconstruction failure). However, per-class AUROC ranks classes by mean-color gap from healthy (brightness contribution). PlantVillage's clean studio backgrounds and class-correlated brightness likely inflate reconstruction-based AUROC. NB2 will evaluate pretrained-feature methods on real-field PlantDoc images to test which signal generalizes.

See `results/nb1/` for figures and per-image scores.

## Reproducing NB1

1. Open `notebooks/nb1_autoencoder.ipynb` on Kaggle.
2. Attach the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).
3. Settings → Accelerator → GPU T4 x2 (or P100).
4. Run all cells. End-to-end runtime ~3 minutes.

## Repository structure

cropanomalynet/
├── notebooks/
│ └── nb1_autoencoder.ipynb
├── results/
│ └── nb1/
│ ├── cae_score_histogram.png
│ ├── cae_roc.png
│ ├── cae_reconstructions.png
│ ├── cae_test_scores.csv
│ └── splits.csv
├── checkpoints/
│ └── cae_best.pt # 4 MB — small enough to commit
└── README.md

## Citation

This work is in progress and does not yet have an associated publication. Code may be referenced as: Smarika Ghimire, Avinash Gautam "CropAnomalyNet: Unsupervised crop disease detection," 2026, https://github.com/<smarikaghimire>/cropanomalynet

## License

MIT
