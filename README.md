# CropAnomalyNet

Unsupervised maize disease detection via reconstruction-based and pretrained-feature anomaly detection, with cross-domain evaluation on PlantDoc field imagery and a controlled background-sensitivity experiment isolating the causal source of cross-domain score inflation.

**Status:** All five notebooks (NB1–NB5) complete. Paper draft in progress.

## Motivation

Supervised crop disease classifiers require thousands of labeled examples per class. We reframe the problem as anomaly detection: train only on healthy leaf images, then flag deviations as diseased. This mirrors industrial defect inspection — abundant "normal" samples, very few labeled defects. We then test whether lab-trained anomaly detectors transfer to real-world field imagery, and design a controlled experiment to identify which failure modes arise.

## Pipeline

| Notebook | Method                                         | Status | Result                                                |
| -------- | ---------------------------------------------- | ------ | ----------------------------------------------------- |
| NB1      | Convolutional Autoencoder (MSE reconstruction) | Done   | AUROC 0.9942                                          |
| NB2      | PatchCore (WR50 layer-2+3, mean-aggregated)    | Done   | AUROC **0.9995**                                      |
| NB3      | Stable Diffusion synthesis evaluation          | Done   | **Negative finding** (FID 323–375)                    |
| NB4      | PlantDoc cross-domain evaluation               | Done   | AUROC **0.9995** (PatchCore cross-domain)             |
| NB5      | Background sensitivity controlled experiment   | Done   | CAE **4.00×** vs PatchCore **0.97×** background ratio |

## Dataset

PlantVillage (Hughes & Salathé, 2015), maize subset, color version. Four classes:

| Class                            | Count |
| -------------------------------- | ----- |
| Corn healthy                     | 1,162 |
| Corn Common rust                 | 1,192 |
| Corn Cercospora (Gray leaf spot) | 513   |
| Corn Northern Leaf Blight        | 985   |

For NB4 cross-domain evaluation, we additionally use **PlantDoc** (Singh et al. 2020) maize subset:

| PlantDoc class      | Count | Maps to              |
| ------------------- | ----- | -------------------- |
| Corn_rust_leaf      | 117   | Common rust          |
| Corn_leaf_blight    | 194   | Northern Leaf Blight |
| Corn_Gray_leaf_spot | 67    | Cercospora           |

PlantDoc has no healthy corn class — see NB4 and NB5 for how this constrains the evaluation and how a controlled stimulus experiment was designed to work around it.

Splits (canonical, in `results/nb1/splits.csv`):

- Train: 929 healthy
- Val: 116 healthy
- Test: 117 healthy + 2,690 diseased (NB1/NB2) or + 2,690 PV + 378 PD diseased (NB4)

All notebooks reuse identical splits (deterministic via shared seed 42).

## Results — PlantVillage maize

Per-class test AUROC (117 healthy + 2,690 diseased):

| Class                | Brightness-only | CAE (NB1)  | PatchCore-mean (NB2) |
| -------------------- | --------------- | ---------- | -------------------- |
| Common rust          | 0.9175          | 0.9996     | **1.0000**           |
| Cercospora           | 0.6853          | 0.9948     | **0.9994**           |
| Northern Leaf Blight | 0.6849          | 0.9874     | **0.9990**           |
| **Overall**          | **0.7880**      | **0.9942** | **0.9995**           |

## Results — Cross-domain on PlantDoc (NB4)

PV healthy training reference vs three test conditions:

| Comparison                               | n_diseased | CAE    | PatchCore  |
| ---------------------------------------- | ---------- | ------ | ---------- |
| In-domain (PV healthy vs PV diseased)    | 2,690      | 0.9945 | **0.9995** |
| Cross-domain (PV healthy vs PD diseased) | 378        | 0.9972 | **0.9995** |
| Unified (PV healthy vs all diseased)     | 3,068      | 0.9948 | **0.9995** |

Per-class cross-domain breakdown:

| Class                | CAE-PV | PC-PV      | CAE-PD | PC-PD      |
| -------------------- | ------ | ---------- | ------ | ---------- |
| Common rust          | 0.9997 | **1.0000** | 0.9940 | **0.9985** |
| Cercospora           | 0.9953 | 0.9994     | 0.9969 | **1.0000** |
| Northern Leaf Blight | 0.9877 | 0.9990     | 0.9992 | **1.0000** |

## Results — Background sensitivity (NB5)

Same healthy leaf, four backgrounds of increasing domain distance:

| Condition                | CAE mean ± std    | PatchCore mean ± std |
| ------------------------ | ----------------- | -------------------- |
| 1. Studio white (native) | 0.00114 ± 0.00031 | 3.2601 ± 0.0364      |
| 2. Solid gray (neutral)  | 0.00151 ± 0.00024 | 3.2201 ± 0.0342      |
| 3. PD field non-corn     | 0.00522 ± 0.00369 | 3.2044 ± 0.1555      |
| 4. PD field corn-disease | 0.00455 ± 0.00184 | 3.1528 ± 0.1357      |

Background sensitivity ratio (Condition 4 / Condition 1):

| Method    | Ratio     | Interpretation                                |
| --------- | --------- | --------------------------------------------- |
| CAE       | **4.00×** | Massive background sensitivity                |
| PatchCore | **0.97×** | None (slightly _less_ anomalous on field bgs) |

### Key findings

1. **PatchCore-mean matches or exceeds the CAE on every class in-domain** while operating on ImageNet-pretrained features. Largest gain is on Northern Leaf Blight (+0.0116), the hardest class for both methods.

2. **Score aggregation matters substantially for diffuse anomalies.** PatchCore's standard `max patch distance` (calibrated for localized industrial defects on benchmarks like MVTec) achieves only 0.9536 on this dataset. `mean` aggregation recovers full performance.

   | Aggregation                  | Overall AUROC |
   | ---------------------------- | ------------- |
   | max (PatchCore default)      | 0.9536        |
   | mean                         | **0.9995**    |
   | top-10% mean                 | 0.9967        |
   | top-1% mean                  | 0.9817        |
   | center max (4-pixel margin)  | 0.9904        |
   | center mean (4-pixel margin) | 0.9995        |

3. **Brightness alone is insufficient** (overall 0.7880). Color carries most of the Common Rust signal (0.9175) but is barely above chance for Cercospora and Northern Blight (~0.685). Both learned methods are doing substantial real texture-based detection on the harder classes.

4. **Off-the-shelf Stable Diffusion v1.5 cannot synthesize convincing maize disease imagery** (NB3). Across four pipeline configurations, FID against real disease images ranged 323–375 versus a healthy intra-class baseline of 32 — an order of magnitude off-distribution.

5. **PatchCore generalizes from lab to field; the CAE's apparent cross-domain success is partly inflated by domain shift** (NB4). PatchCore reaches AUROC 0.9995 cross-domain — identical to its in-domain performance — with 1.0000 per-class on cercospora and northern blight. The CAE's 0.9972 cross-domain AUROC is misleading: its PlantDoc reconstruction errors are 1.4–2.5× its PlantVillage diseased errors for the same underlying disease class, indicating background/lighting novelty contributes substantially to its anomaly score.

6. **A controlled experiment confirms the CAE's background sensitivity is causal, not correlational** (NB5). When the same healthy corn leaf is composited onto four backgrounds of increasing domain distance (studio white → solid gray → PlantDoc non-corn → PlantDoc corn-disease), the CAE's reconstruction error rises 4.00× while PatchCore's score stays flat (ratio 0.97×, slight decrease). Since only the background pixels vary across conditions, this isolates the causal effect: the CAE responds to backgrounds; PatchCore does not. PatchCore actually finds field-style backgrounds _more_ normal than synthetic studio backgrounds, consistent with its ImageNet pretraining having seen extensive outdoor/plant imagery.

## NB1 — Convolutional Autoencoder

Vanilla CAE (1.08M params) trained on 929 healthy maize images for 45 epochs with early stopping. Output in `[0, 1]` via sigmoid; MSE loss; Adam at lr=1e-3 with `ReduceLROnPlateau`.

Per-pixel reconstruction-error maps localize to lesion regions (`results/nb1/cae_reconstructions.png`), indicating genuine texture-based reconstruction failure rather than just global brightness mismatch.

Artifacts in `results/nb1/`:

- `cae_score_histogram.png`, `cae_roc.png`, `cae_reconstructions.png`
- `cae_test_scores.csv`, `splits.csv`
- Checkpoint: `checkpoints/cae_best.pt` (4 MB, included)

## NB2 — PatchCore

WideResNet50 ImageNet-pretrained backbone, frozen. Features from layer 2 (28×28×512) and layer 3 (14×14×1024) captured via forward hooks; layer 3 upsampled to 28×28 via bilinear interpolation, concatenated channel-wise with layer 2, then 3×3 average-pooled for neighborhood-aware patches (Roth et al. 2022). Each image yields 784 patches × 1536 dims.

Memory bank: 728,336 patches from 929 healthy training images. Reduced to 10% (72,833 patches) via greedy farthest-point coreset sampling on Johnson-Lindenstrauss-projected (128-dim) features.

Inference: `torch.cdist` between test patches and coreset bank, minimum over the coreset gives a 28×28 anomaly map per image, **mean** over the 784 patches gives the image-level anomaly score.

Artifacts in `results/nb2/`:

- `patchcore_score_histogram.png`, `patchcore_roc.png`, `patchcore_anomaly_maps.png`
- `patchcore_test_scores.csv` (columns: `score_max`, `score_mean`)
- `brightness_baseline_scores.csv`
- Coreset memory bank (~447 MB) regenerable from notebook cells 4–7; excluded via `.gitignore`.

## NB3 — Stable Diffusion synthesis evaluation

We tested four configurations of Stable Diffusion v1.5 for generating synthetic maize disease images on PlantVillage healthy bases:

1. img2img with descriptive prompts (strength 0.2–0.6)
2. ControlNet (Canny edges) standalone
3. ControlNet + img2img
4. ControlNet with visual-symptom-only prompts (no disease vocabulary)

All four configurations failed to produce convincing maize disease imagery. Failure modes:

- **Wrong species** at high denoising strength: broadleaf garden plants, autumn fallen leaves, rubber plants with pinnate venation — none are corn.
- **Misinterpreted disease vocabulary**: "rust" → rusted metal scales / beetle exoskeletons; "pustules" → broccoli florets.
- **No disease added** at low strength: output essentially preserves the healthy input.

Quantitative results (FID, 50 synthetic vs 200 real per class):

| Comparison                                                    | FID      |
| ------------------------------------------------------------- | -------- |
| Healthy maize, half-A vs half-B (intra-distribution baseline) | **32.2** |
| Synthetic common rust vs real common rust                     | 352.0    |
| Synthetic Cercospora vs real Cercospora                       | 374.9    |
| Synthetic Northern Blight vs real Northern Blight             | 323.8    |

Synthetic-vs-real FID is roughly an order of magnitude above the healthy intra-class baseline, confirming the visual evidence: vanilla SD v1.5 cannot produce in-distribution maize disease images via prompt engineering alone. Domain adaptation (LoRA, fine-tuning, or paired-image conditioning like Dreambooth) would be required.

Artifacts in `results/nb3/`:

- Iteration grids: `sd_sanity_check.png`, `sd_strength_sweep.png`, `sd_controlnet_sweep.png`, `sd_controlnet_img2img_sweep.png`, `sd_final_attempt.png`
- `synthetic_samples.png`, `synthetic_samples/` (15 sample images), `fid_results.json`

## NB4 — Cross-domain evaluation on PlantDoc

We test whether the CAE and PatchCore — both trained/built only on PlantVillage's clean studio healthy images — generalize to real-world maize disease imagery from PlantDoc (Singh et al. 2020). PlantDoc has no healthy corn class, so a pure PD-healthy-vs-PD-diseased AUROC is not computable; we instead evaluate the deployment scenario directly: PV healthy training images as the normal reference, vs three test conditions (in-domain PV diseased, cross-domain PD diseased, and the union).

Test set: 117 PV healthy + 2,690 PV diseased + 378 PD diseased = 3,185 images. Same seed and train/val splits as NB1/NB2 (929 train / 116 val healthy from PlantVillage).

### Score-magnitude evidence for the cross-domain story

The headline AUROC numbers (above, in Results — Cross-domain) understate the difference between the two methods. The score distributions tell the full story:

CAE reconstruction error (mean per class):

| Class           | PV diseased | PD diseased | Ratio |
| --------------- | ----------- | ----------- | ----- |
| Common rust     | 0.00347     | 0.00476     | 1.37× |
| Cercospora      | 0.00282     | 0.00547     | 1.94× |
| Northern blight | 0.00231     | 0.00574     | 2.48× |

PlantDoc images produce reconstruction errors 1.4–2.5× larger than PlantVillage diseased images _for the same underlying disease class_. This indicates the CAE is detecting both disease _and_ background/lighting novelty when scoring PlantDoc.

PatchCore mean-aggregated anomaly score:

| Class           | PV diseased | PD diseased | Ratio  |
| --------------- | ----------- | ----------- | ------ |
| Common rust     | 3.1754      | 3.1608      | 0.995× |
| Cercospora      | 2.9342      | 3.1271      | 1.066× |
| Northern blight | 2.8999      | 3.1685      | 1.093× |

PatchCore's per-class score magnitudes are essentially flat across domains (0.99–1.09×). The PD diseased distribution overlaps almost completely with the PV diseased distribution (see `nb4_score_histograms.png`), strongly suggesting PatchCore detects disease texture genuinely rather than chasing domain novelty.

### Mechanism

PatchCore uses ImageNet-pretrained WideResNet50 features. ImageNet contains diverse real-world imagery (outdoor scenes, dirt, hands, varied lighting), so these features remain discriminative for plant disease texture across the lab-vs-field domain gap. The CAE, trained from scratch on PlantVillage studio healthy images only, has no such prior exposure to field-style imagery, and learns reconstruction filters tuned to the studio distribution. NB5 confirms this mechanism via a controlled experiment.

### Limitation

PlantDoc has no healthy corn class. Without healthy PD images as a control, we cannot directly compute AUROC of "field healthy vs field diseased." NB5 addresses this gap by constructing a controlled stimulus experiment in which only the background varies (the leaf is held identical), allowing us to isolate background sensitivity without requiring a healthy PD image set.

Artifacts in `results/nb4/`:

- `nb4_score_histograms.png` (CAE and PatchCore distributions, three conditions each)
- `nb4_anomaly_maps.png` (PatchCore anomaly maps on PV healthy, PV diseased, PD diseased)
- `cae_scores.csv`, `patchcore_scores.csv` (per-image scores with domain & class labels)
- `manifest.csv` (full test set composition)

## NB5 — Background sensitivity controlled experiment

NB4 demonstrated that the CAE's cross-domain reconstruction errors are systematically inflated compared to its in-domain errors. NB5 directly tests _whether this inflation is caused by background novelty_ via a controlled stimulus experiment in which the leaf content is held constant while the background varies.

### Design

**Stimulus:** 100 segmented healthy maize leaves from PlantVillage (random sample of 1,162; different seed from training-set sampling so the stimulus is unbiased).

For each leaf we generate four composite images by placing the leaf — cropped to its bounding box and scaled to ~55% of canvas dimension — onto four background conditions of increasing domain distance from the PlantVillage studio reference:

1. **Studio white** (240, 240, 240) — near-native background
2. **Solid gray** (128, 128, 128) — neutral, low-frequency, novel but unstructured
3. **PlantDoc non-corn** — random field imagery from PlantDoc's other crop classes (apple, peach, tomato, etc.)
4. **PlantDoc corn-disease** — random field imagery from PlantDoc's three diseased corn classes (the deployment-domain background)

The leaf is pixel-identical across all four conditions; only the surrounding pixels (~70% of image area) change. This isolates the background's contribution to the anomaly score.

Each of the 400 composites (100 leaves × 4 conditions) is scored with both the CAE (same architecture and training procedure as NB1) and PatchCore (same construction as NB2, mean-aggregated).

### Findings

The CAE's reconstruction error increases **4.00×** when the background shifts from studio white to PlantDoc field imagery — even though the leaf is unchanged and was healthy to begin with. The model is calling healthy corn leaves anomalous purely because their surrounding pixels look unfamiliar. This is a direct causal demonstration of the score-inflation pattern observed in NB4.

PatchCore's score actually _decreases slightly_ (0.97×) when moved to field backgrounds. This is consistent with its ImageNet pretraining: outdoor/plant imagery is well-represented in ImageNet while solid-colored backgrounds are rare, so a leaf on a field background looks _more_ in-distribution to PatchCore than the same leaf on solid white. This is the strongest possible form of domain robustness — not merely tolerance of background variation, but actually finding natural backgrounds _more_ normal than synthetic ones.

### Implications

A practitioner planning to deploy a PlantVillage-trained anomaly detector in real-world field conditions should:

- Expect the CAE to produce many false positives on real-field images regardless of disease status, because field backgrounds inflate the anomaly score for _all_ leaves — healthy and diseased alike.
- Expect PatchCore to maintain its in-domain detection performance with no systematic false-positive bias from background variation.
- Prefer pretrained-feature methods (PatchCore family) over from-scratch reconstruction methods (CAE family) for any deployment in conditions visually distinct from the training distribution.

Artifacts in `results/nb5/`:

- `nb5_composite_preview.png` (3 leaves × 4 conditions visual sanity check)
- `nb5_background_sensitivity.png` (the causal-effect boxplot — the paper's confirmatory figure)
- `nb5_results.csv` (400 rows: leaf_idx, condition, cae_score, pc_score)

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
4. Run all cells. End-to-end runtime ~15 minutes.

### NB3

1. Open `notebooks/nb3_stable_diffusion.ipynb` on Kaggle.
2. Attach the same PlantVillage dataset.
3. **Settings → Internet → On** (required for downloading SD v1.5 weights from HuggingFace).
4. Settings → Accelerator → GPU T4 x2 (or P100).
5. Run all cells. End-to-end runtime ~25 minutes.

### NB4

1. Open `notebooks/nb4_plantdoc_eval.ipynb` on Kaggle.
2. Attach both [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and [PlantDoc](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset).
3. Settings → Accelerator → GPU T4 x2 (or P100).
4. Run all cells. End-to-end runtime ~16 minutes.

### NB5

1. Open `notebooks/nb5_background_sensitivity.ipynb` on Kaggle.
2. Attach both PlantVillage and PlantDoc.
3. Settings → Accelerator → GPU T4 x2 (or P100).
4. Run all cells. End-to-end runtime ~13 minutes (CAE training 1.5 min + memory bank 15 s + coreset 10 min + experiment 1 min).

## Repository structure

```
cropanomalynet/
├── notebooks/
│   ├── nb1_autoencoder.ipynb
│   ├── nb2_patchcore.ipynb
│   ├── nb3_stable_diffusion.ipynb
│   ├── nb4_plantdoc_eval.ipynb
│   └── nb5_background_sensitivity.ipynb
├── results/
│   ├── nb1/  (5 files)
│   ├── nb2/  (5 files)
│   ├── nb3/  (8 items, incl. synthetic_samples/)
│   ├── nb4/
│   │   ├── manifest.csv
│   │   ├── cae_scores.csv
│   │   ├── patchcore_scores.csv
│   │   ├── nb4_score_histograms.png
│   │   └── nb4_anomaly_maps.png
│   └── nb5/
│       ├── nb5_composite_preview.png
│       ├── nb5_background_sensitivity.png
│       └── nb5_results.csv
├── checkpoints/cae_best.pt
├── .gitignore
└── README.md
```

## Citation

This work is in progress and does not yet have an associated publication. Code may be referenced as:

Smarika Ghimire, Avinash Gautam, "CropAnomalyNet: Unsupervised crop disease detection," 2026, https://github.com/smarikaghimire/cropanomalynet

## License

MIT
