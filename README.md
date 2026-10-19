# Galaxy Morphology Classification Under Limited Labeling Budgets

### A comparative study of transfer learning, CNN fine-tuning, and probability ensembling on Galaxy10 DECaLS

**Research question.** *How effectively can pretrained convolutional neural networks classify galaxy morphology with a limited labeling budget, and why does performance vary substantially between morphological classes?*

**Headline result:** An equal-weight ensemble of three pretrained CNNs reaches **75.41% held-out test accuracy** and **0.7238 macro-F1** using **2,000 labeled training images per run**. Across three random seeds, the ensemble improves test macro-F1 over the strongest individual architecture (EfficientNet-B0) and shows little aggregate degradation relative to validation. Class-level analysis reveals a striking contrast: **Class 4 achieves F1 = 0.637**, despite having only 50 test examples, whereas **Class 0 achieves F1 = 0.351** with 157 test examples. This contrast motivates an investigation of morphological ambiguity, visual consistency, and class overlap rather than class frequency alone.

> **Scope.** This repository includes an earlier active-learning study and a later, separate, fixed-budget CNN comparison. The **75.41% result is from the CNN ensemble experiment**, not from an active-learning sampling strategy. The label budget is shared *within each seed*: ensemble members train on the same 2,000 labeled examples, not three disjoint sets of 2,000.

---

## Abstract

Galaxy morphology classification is challenging when manually labeled images are limited, particularly when categories differ in visual distinctiveness and prevalence. This study evaluates the effectiveness of transfer learning under a controlled **2,000-image training-label budget** using the Galaxy10 DECaLS dataset. We compare a frozen ResNet18 feature extractor with partially fine-tuned ResNet18, fully fine-tuned MobileNetV3-Small, and fully fine-tuned EfficientNet-B0. We also construct an equal-weight probability ensemble from the three fine-tuned architectures. Evaluation uses a fixed, previously prepared train/validation/test split, three independent training-subset seeds, validation-guided checkpoint selection, and a held-out test set of 2,642 images. The ensemble achieves **75.41% ± 1.18% test accuracy** and **0.7238 ± 0.0078 macro-F1** (mean ± standard deviation across seeds), exceeding EfficientNet-B0 by **1.50 percentage points** in accuracy and **0.0186** in macro-F1. Per-class and visual error analyses show that the ensemble performs unevenly across classes: Class 0 is frequently confused with Classes 7 and 3, while many correctly classified Class 4 examples share a visually elongated appearance. These observations suggest that label efficiency depends not only on the number of labeled examples but also on the discriminability and consistency of class-defining visual features. This is a controlled project-level comparison, not a claim of state-of-the-art performance or a causal test of morphology-related difficulty.

## 1. Introduction

Galaxy morphology provides a visual description of galactic structure and is relevant to studying galaxy formation and evolution. Automated classification can scale morphological analysis beyond what is practical through manual labeling alone, but labeled astronomical images are costly to obtain and may contain visually ambiguous cases. Transfer learning offers a practical approach: start from representations learned on large image collections and adapt them to the target astronomical domain using fewer labeled examples.

This project addresses two questions:

1. **Label-budget effectiveness:** How well do pretrained CNNs and a simple ensemble perform when each training run is restricted to 2,000 labeled galaxy images?
2. **Class-level heterogeneity:** Why can a class with relatively few examples be easier to recognize than a class with more examples, and what do systematic misclassifications reveal about the model's limitations?

The study compares model architectures under a common budget and then investigates errors using per-class metrics, prediction probabilities, confusion patterns, and original galaxy images.

## 2. Dataset and experimental design

### 2.1 Galaxy10 DECaLS and data preparation

- **Dataset:** Galaxy10 DECaLS; ten morphological class IDs (`0`–`9`).
- **Cleaned dataset:** 17,615 retained images after the project's cleaning and duplicate/group checks.
- **Fixed split:** 12,330 training-pool images; 2,643 validation images; **2,642 held-out test images**.
- **Training budget:** **2,000 distinct labeled training images per seed**, sampled from the fixed training pool. This is approximately **16.2% of the training pool**.
- **Random seeds:** `42`, `43`, and `44`. The training subset changes with seed; the held-out test images do not.
- **Image processing:** RGB images resized to **128 × 128** with bilinear antialiasing and scaled to `[0, 1]`. CNN inputs are normalized with ImageNet channel statistics in the model forward pass.
- **Augmentation:** random quarter-turn rotations and horizontal/vertical flips for fine-tuned models during training; no stochastic augmentation during test inference.

The prepared split is reused rather than regenerated. Training and checkpoint selection do not use held-out test labels. The experiment uses an **800-image stratified validation monitor** for early stopping and retains the full validation set for model assessment.

> **Important distinction:** The 157 Class 0 images and 50 Class 4 images discussed below are **test-set supports**, not counts of labeled training images. They describe evaluation prevalence; they do not establish that the model trained on 157 versus 50 examples.

### 2.2 Evaluation protocol

All models within a seed use the same 2,000-image labeled subset. Models are selected by validation-monitor macro-F1, with cross-entropy used to break ties. The reported final test metrics are calculated once from saved best checkpoints. The three-seed standard deviations describe variability across training runs and labeled-subset selections; they are **not confidence intervals from three independent test sets**.

**Primary metric:** macro-F1, which weights all ten classes equally and is more informative than accuracy alone when class frequencies and difficulty vary. **Secondary metrics:** accuracy, per-class precision/recall/F1, cross-entropy, and confusion matrices.

## 3. Methodology

### 3.1 Compared CNN configurations

| Configuration | Backbone | Training strategy | Augmentation |
|---|---|---|---|
| Frozen ResNet18 | ResNet18 | Freeze backbone; train classification head | No |
| Partial ResNet18 | ResNet18 | Fine-tune later backbone blocks and head | Yes |
| MobileNetV3-Small | MobileNetV3-Small | Fine-tune backbone and head | Yes |
| EfficientNet-B0 | EfficientNet-B0 | Fine-tune backbone and head | Yes |

The pretrained architectures start from ImageNet weights. Each model uses a ten-class classification head. The pipeline includes training-subset-based feature standardization, AdamW optimization, warm-up for applicable fine-tuned models, and early stopping on validation-monitor macro-F1. The common comparison runner uses a maximum of **30 epochs**, **patience 6**, **minimum 8 epochs**, and **2 warm-up epochs** (subject to its saved run protocol).

### 3.2 Equal-weight ensemble

The final ensemble combines the **partially fine-tuned ResNet18**, **MobileNetV3-Small**, and **EfficientNet-B0**. Given the predicted class-probability vector from member model \(m\), the ensemble prediction is

\[
\bar{p}(y=c\mid x)=\frac{1}{3}\sum_{m=1}^{3}p_m(y=c\mid x),
\qquad \hat{y}(x)=\arg\max_c\bar{p}(y=c\mid x).
\]

Ensemble weights are fixed at one third each: **no test-set fitting or weight tuning** is performed. Ensemble members are trained on the **same labeled subset within a seed**, so combining them does not require additional distinct labeled images. It does, however, increase model training and inference cost.

### 3.3 Separate earlier active-learning investigation

The repository also contains a separate active-learning investigation using frozen ResNet18 features and a lightweight classifier. At a 2,000-label budget, its held-out macro-F1 results were approximately:

| Sampling strategy | Test macro-F1 (mean ± SD) |
|---|---:|
| Hybrid | 0.4791 ± 0.0032 |
| Diversity | 0.4791 ± 0.0114 |
| Random | 0.4675 ± 0.0155 |
| Uncertainty | 0.4651 ± 0.0064 |

These results document the broader project's exploration of label selection. **They are not a controlled head-to-head comparison with the later fine-tuned CNN ensemble**: representation, classifier, and training protocol differ. Consequently, the present study does not claim that active learning caused the ensemble's improvement. See [`README_ACTIVE_LEARNING.md`](README_ACTIVE_LEARNING.md) for that experiment.

## 4. Experimental results

### 4.1 Held-out test performance

The table reports **mean ± sample standard deviation across seeds 42, 43, and 44**, evaluated on the same 2,642 held-out test images per seed.

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Frozen ResNet18 | 47.59% ± 0.57% | 0.4280 ± 0.0127 |
| Partial ResNet18 | 70.73% ± 0.71% | 0.6796 ± 0.0068 |
| MobileNetV3-Small | 65.35% ± 0.94% | 0.6116 ± 0.0023 |
| EfficientNet-B0 | 73.91% ± 1.20% | 0.7052 ± 0.0096 |
| **Equal-weight ensemble (3)** | **75.41% ± 1.18%** | **0.7238 ± 0.0078** |

**Interpretation.** Fine-tuning substantially outperforms the frozen-feature ResNet18 baseline under this experimental setup. EfficientNet-B0 is the strongest single model, but the ensemble improves upon it by **1.50 percentage points** in test accuracy and **0.0186 absolute macro-F1** (approximately **2.6% relative macro-F1 improvement**). These improvements are descriptive; no formal significance claim is made.

### 4.2 Per-seed ensemble consistency

| Seed | Ensemble test accuracy | Ensemble test macro-F1 |
|---|---:|---:|
| 42 | 76.31% | 0.7302 |
| 43 | 75.85% | 0.7262 |
| 44 | 74.07% | 0.7150 |
| **Mean ± SD** | **75.41% ± 1.18%** | **0.7238 ± 0.0078** |

The ensemble exceeds EfficientNet-B0 in test accuracy and macro-F1 for each of the three seeds. A comparison of the recorded aggregate validation and test results also shows little degradation: **ensemble validation accuracy ≈ 75.66% versus test accuracy 75.41%**, and **validation macro-F1 ≈ 0.7255 versus test macro-F1 0.7238**. This supports stable performance on this held-out split, but does not rule out all forms of overfitting or distribution shift.

### 4.3 Per-class performance

Per-class metrics below are the **mean of each seed's ensemble metrics**; `support` is the number of unique test images in that class, not the sum across seeds.

| Class ID | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **0** | 157 | 0.479 | 0.278 | **0.351** |
| 1 | 276 | 0.786 | 0.814 | 0.797 |
| 2 | 395 | 0.863 | 0.904 | 0.883 |
| 3 | 303 | 0.802 | 0.854 | 0.827 |
| **4** | **50** | **0.672** | **0.613** | **0.637** |
| 5 | 306 | 0.815 | 0.687 | 0.745 |
| 6 | 273 | 0.696 | 0.608 | 0.648 |
| 7 | 389 | 0.571 | 0.686 | 0.621 |
| 8 | 213 | 0.839 | 0.890 | 0.864 |
| 9 | 280 | 0.859 | 0.874 | 0.866 |

The large spread in F1 scores shows why overall accuracy alone is insufficient. Class 2 achieves an F1 of **0.883**, while Class 0 reaches only **0.351**. Class 4 is noteworthy: despite having the smallest test support, its F1 exceeds that of Class 0 and Class 7.

### 4.4 Where the ensemble helps

Comparing per-class F1 against EfficientNet-B0:

| Class | EfficientNet-B0 F1 | Ensemble F1 | Change |
|---|---:|---:|---:|
| 0 | 0.3601 | 0.3510 | −0.0091 |
| 1 | 0.7727 | 0.7966 | +0.0239 |
| 2 | 0.8686 | 0.8826 | +0.0140 |
| 3 | 0.8288 | 0.8267 | −0.0021 |
| **4** | **0.5374** | **0.6368** | **+0.0994** |
| 5 | 0.7461 | 0.7451 | −0.0010 |
| 6 | 0.6255 | 0.6477 | +0.0221 |
| 7 | 0.5947 | 0.6210 | +0.0262 |
| 8 | 0.8584 | 0.8640 | +0.0056 |
| 9 | 0.8598 | 0.8662 | +0.0065 |

The largest per-class improvement occurs in **Class 4 (+0.0994 F1)**, corresponding to roughly **18.5% relative improvement** over EfficientNet-B0. However, this class contains only 50 unique test images, so the estimate should be interpreted cautiously. The ensemble is not uniformly better: Class 0 declines slightly.

A paired comparison of EfficientNet-B0 and ensemble predictions over the three seeds found **468 cases in which the ensemble corrected an EfficientNet error** versus **349 cases in which it introduced an error**, for a net gain of **119 correct predictions across seed-image evaluations**. Because the same test images are evaluated in three seeds, these counts are not 468 or 349 independent images.

## 5. Error analysis and discussion

### 5.1 Why does Class 0 perform worse than Class 4?

Class 0 has **157 test examples** versus **50** for Class 4, yet its ensemble recall is **0.278** versus **0.613** and its F1 is **0.351** versus **0.637**. More evaluation examples do not imply a class is easier to recognize, and evaluation support alone does not measure the number of training examples seen by the model.

Visual inspection of original test images provides a plausible explanation. **Class 0 examples exhibit a heterogeneous range of appearances**, including compact, diffuse, irregular, and relatively smooth-looking structures. Some resemble images assigned to other classes. By contrast, **many correctly classified Class 4 examples share a distinctly elongated, high-aspect-ratio appearance**. This visual regularity may provide a more consistent discriminative cue even under a limited labeling budget.

This is an **interpretation supported by example images and confusion patterns**, not a causal proof. We do not claim that the official Class 0 definition is literally “blurry,” nor that low image quality is the sole explanation. Morphological ambiguity, overlap between class definitions, image resolution, and variation within a class may all contribute. Official dataset descriptions should be consulted before attributing a specific semantic meaning to a numeric class ID.

### 5.2 Dominant confusion patterns

Aggregating the three ensemble confusion matrices, the most frequent error directions include:

| True → predicted | Count across three seeds |
|---|---:|
| 6 → 7 | 202 |
| 5 → 7 | 161 |
| 0 → 7 | 131 |
| 7 → 6 | 108 |
| 7 → 5 | 82 |
| 7 → 0 | 68 |
| 0 → 3 | 63 |

**Class 7 is a frequent destination for incorrect predictions**, helping explain its moderate recall (**0.686**) but lower precision (**0.571**). For Class 0, confusion with Classes 7 and 3 is particularly pronounced. This is consistent with a poorly separated decision region rather than a single isolated failure mode.

### 5.3 Qualitative visual inspection

Visual grids compare correctly classified Class 0 images, Class 0 images misclassified as Classes 7 and 3, and correctly and incorrectly classified Class 4 images. The Class 0 error examples include both compact/smooth and more diffuse appearances; Class 4 correct examples frequently display elongated structure. Class 4 errors also show that elongation alone is not sufficient for reliable recognition.

The figure files are kept in [`results/visual_error_analysis/`](results/visual_error_analysis/) **when uploaded with this report**:

| Visual sample | File |
|---|---|
| Class 0 → Class 7 | [`class0_as7.jpg`](results/visual_error_analysis/class0_as7.jpg) |
| Class 0 correctly classified | [`class0_correct.jpg`](results/visual_error_analysis/class0_correct.jpg) |
| Class 0 → Class 3 | [`class0_as3.jpg`](results/visual_error_analysis/class0_as3.jpg) |
| Class 4 correctly classified | [`class4_correct.jpg`](results/visual_error_analysis/class4_correct.jpg) |
| Class 4 misclassified | [`class4_wrong.jpg`](results/visual_error_analysis/class4_wrong.jpg) |

**Representative comparison** (visible once the accompanying image files are committed):

![Class 0 examples misclassified as Class 7](results/visual_error_analysis/class0_as7.jpg)

![Correctly classified Class 4 examples](results/visual_error_analysis/class4_correct.jpg)

The grids are qualitative illustrations, not representative samples for estimating class-level effect sizes.

## 6. Limitations

1. **Single dataset and fixed test split.** Results demonstrate held-out performance on Galaxy10 DECaLS, not necessarily transfer to different telescopes, surveys, or labeling schemes.
2. **Small evaluation support for Class 4.** Its 50 test examples make per-class estimates more sensitive to a few changed predictions.
3. **Only three training seeds.** The reported standard deviations characterize these runs; they do not establish statistical significance or broad robustness.
4. **Model complexity versus labeling cost.** The ensemble reuses the same labels but requires training and serving three CNNs; label efficiency is not equivalent to computational efficiency.
5. **Qualitative morphological interpretation.** The visual analysis suggests class heterogeneity and overlap but does not formally quantify morphology or verify whether specific examples are ambiguously labeled.
6. **No full budget-performance curve for the final CNNs.** This experiment demonstrates performance *at* 2,000 labels, but cannot establish the minimum necessary budget or the optimal annotation strategy.
7. **Model-selection context.** Architecture exploration and validation comparisons took place during development. The held-out test supports final reporting, but should not be reused to tune future models without establishing a new untouched evaluation set.

## 7. Conclusion

Under a fixed **2,000-label training budget**, pretrained and fine-tuned CNNs provide a practical approach to Galaxy10 morphology classification. The equal-weight ensemble reaches **75.41% test accuracy** and **0.7238 macro-F1**, improving upon the strongest single model without requiring additional distinct training labels. The close validation and test aggregates are encouraging, while the substantial per-class differences highlight a central limitation: **recognition difficulty depends on class separability and visual consistency, not merely on class prevalence**. In particular, Class 4 performs comparatively well despite low test support, whereas Class 0 remains difficult and is frequently confused with Classes 7 and 3. Future work could investigate class-aware calibration, targeted data collection, or more precise morphological features, ideally using a new untouched test split for subsequent model changes.

## 8. Repository guide and reproducibility

The repository contains multiple experimental phases. The most relevant components include:

```text
galaxy_al/                  # Earlier active-learning experiments
galaxy_image_demo/          # Earlier image-model pilot code
image_budget_runner/        # Isolated fixed-budget CNN training and evaluation engine
run_image_2000.py           # Three-seed fixed-budget CNN comparison runner
summarize_image_2000.py     # Cross-seed aggregation (when present)
evaluate_final_test.py      # Held-out test inference from saved checkpoints (new)
visual_error_analysis.py    # Qualitative inspection of saved test predictions (new)
README_ACTIVE_LEARNING.md   # Earlier active-learning investigation
README_IMAGE_2000.md       # Detailed fixed-budget experiment notes
results/
  visual_error_analysis/    # Selected qualitative grids (new)
```

**To reproduce the final test from existing trained checkpoints**, use the same prepared data, configuration, and saved campaign outputs, then run from the project root on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe evaluate_final_test.py --device cpu --batch-size 32 --threads 4
```

**To regenerate the qualitative examples** from saved predictions and original images:

```powershell
.\.venv\Scripts\python.exe visual_error_analysis.py --seed 42 --examples 20
```

These are **evaluation commands**, not training commands. The training comparison is managed separately by `run_image_2000.py`; refer to the saved campaign protocol and `README_IMAGE_2000.md` before attempting a fresh training run. Raw images and large model checkpoints are not required in Git history to read the report, and may be excluded from Git using `.gitignore`.

### Evaluation artifacts

The final evaluation produces a report and machine-readable files in `outputs/final_test/comparison_.../`:

- `test_report.md` — held-out test summary;
- `per_seed_metrics.csv` — model performance for each seed;
- `per_class_metrics.csv` — class-level precision, recall, F1, and support;
- `predictions.csv` — per-image probabilities, labels, and predictions;
- confusion matrix figures and `test_results.json` — diagnostic plots and provenance.

For a lightweight public repository, commit the report, metric CSVs, and selected figures. Consider the size and licensing implications before publishing raw images, full prediction tables, datasets, or pretrained weights.

## Data attribution

This project uses **Galaxy10 DECaLS** and pretrained torchvision models. Please follow the dataset authors' citation and licensing requirements when redistributing images or presenting the work publicly. The numerical class IDs used in the tables follow the prepared dataset mapping; this report deliberately avoids inventing unsupported official class descriptions.

---

**Project status:** Fixed-budget CNN comparison, three-seed held-out test evaluation, and visual error analysis completed. Results are observational and reproducible from the saved protocol, dataset preparation, and checkpoints.
