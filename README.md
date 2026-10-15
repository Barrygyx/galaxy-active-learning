# Galaxy Morphology Classification with Label-Efficient Learning

This project investigates how to classify galaxy morphology with limited labeled images while preserving recognition of rare classes. It combines data auditing, active learning baselines, model diagnostics, and pretrained CNN fine-tuning using the Galaxy10 DECaLS dataset.

**Current results:** using 2,000 randomly selected training images, EfficientNet-B0 achieved **74.55% validation accuracy** and **0.7151 ± 0.0165 validation macro-F1** across three seeds. A fixed three-model ensemble achieved **75.66% accuracy** and **0.7255 ± 0.0070 macro-F1**.

These results establish a promising model-training baseline. The next stage will test whether active learning can reach comparable performance using fewer training labels.

## Research Question

> Can active learning that combines prediction uncertainty and sample diversity reduce the number of labeled galaxy images needed to reach a target classification performance while preserving recognition of rare morphology classes?

The project evaluates overall performance alongside class-level outcomes. Class 4, **cigar-shaped smooth galaxies**, is the prespecified rare-class focus. All ten morphology classes remain in the evaluation.

The hypothesis is tested against random sampling rather than assuming that the hybrid acquisition strategy will perform better.

## Project Progress

| Stage | Status | Purpose |
|---|---|---|
| Dataset inspection and cleaning | Completed | Identify duplicate-linked samples, conflicting labels, and invalid values |
| Fixed data splitting | Completed | Establish a common training pool, validation set, and test set |
| Frozen-feature active learning baseline | Completed, exploratory | Compare random, uncertainty, diversity, and hybrid acquisition |
| Classifier diagnostics and head comparisons | Completed | Investigate generalization gaps and limitations of frozen features |
| Small CNN pilot | Completed | Compare scratch training, frozen encoders, and CNN fine-tuning |
| Three-seed, 2,000-label CNN comparison | Completed | Evaluate model performance and repeatability |
| Active learning with the selected fine-tuned CNN | Planned | Directly test annotation efficiency and rare-class preservation |

**Implementation distinction:** `run_pipeline.py` runs the earlier frozen-feature active learning experiment. `run_image_2000.py` runs the newer CNN comparison; it does not yet perform iterative active learning acquisition.

## Dataset

The project uses [Galaxy10 DECaLS](https://astronn.readthedocs.io/en/latest/galaxy10.html), containing **17,736 astronomical images** of size **256 × 256 × 3** across ten morphology classes.

- Images originate from the DESI Legacy Imaging Surveys.
- Morphology labels are derived from Galaxy Zoo classifications.
- The HDF5 contains `images`, `ans`, `ra`, `dec`, `redshift`, and `pxscale`.
- Current classifiers use the images as predictors and `ans` as the target. Astronomical metadata is not included as a model input.

| Class | Morphology | Original images |
|---|---|---:|
| 0 | Disturbed galaxies | 1,081 |
| 1 | Merging galaxies | 1,853 |
| 2 | Round smooth galaxies | 2,645 |
| 3 | In-between round smooth galaxies | 2,027 |
| 4 | Cigar-shaped smooth galaxies | 334 |
| 5 | Barred spiral galaxies | 2,043 |
| 6 | Unbarred tight spiral galaxies | 1,829 |
| 7 | Unbarred loose spiral galaxies | 2,628 |
| 8 | Edge-on galaxies without bulge | 1,423 |
| 9 | Edge-on galaxies with bulge | 1,873 |

Download the original file separately:

https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5?download=1

Expected SHA256:

```text
19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571
```

The original HDF5 is excluded from Git.

## Data Cleaning and Fixed Split

The raw HDF5 is opened read-only. Cleaning records retain the original HDF5 row indices.

The preparation procedure:

1. Validates image shapes, label alignment, and the ten-class label range.
2. Checks images for invalid pixel values.
3. Identifies byte-identical images using image hashes.
4. Links samples with exactly equal finite RA/Dec coordinates.
5. Combines these links transitively into groups.
6. Excludes entire linked groups containing conflicting labels rather than guessing replacement labels.
7. Preserves natural class proportions and creates fixed partitions.

The published cleaning run excluded **121 images**, all associated with conflicting labels in linked groups, leaving **17,615 images**.

Missing redshift values were recorded but did not cause otherwise valid images to be removed because redshift is not a classifier input.

| Partition | Images |
|---|---:|
| Training pool | 12,330 |
| Validation | 2,643 |
| Test | 2,642 |
| **Total retained** | **17,615** |

The original preparation directory is:

```text
data/processed/prepared_20261014T042419_288789Z/
```

The SHA256 of its `splits.csv` is:

```text
39c4d6ecc7c0d823bafece1a0858b908d495a86e5dff3692c981808c705f6703
```

A versioned copy of the preparation records is stored in `results/data_preparation/`. Exact image hashing and coordinate linkage reduce known duplication risks but do not guarantee detection of every near-duplicate or related astronomical object.

## Initial Active Learning Baseline

The first experiment used an ImageNet-pretrained ResNet18 with frozen convolutional parameters. Images were resized to 224 × 224, and cached 512-dimensional features were passed to a small MLP classifier.

The acquisition strategies were:

- **Random:** uniformly sample from the remaining training pool.
- **Uncertainty:** select samples with the highest predictive entropy.
- **Diversity:** use greedy farthest-first selection in normalized feature space.
- **Hybrid:** create an uncertainty shortlist and select a diverse query batch from that shortlist.

The original protocol used 200 initial labels, query batches of 200, and a maximum of 2,000 labels. Strategies shared the initial labeled subset within each seed.

At 2,000 labels, the historical mean validation macro-F1 was **0.4802 for random sampling** and **0.4883 for hybrid sampling**. Hybrid improved rare-class validation F1 but also reduced performance on some other classes. The experiment did not establish the intended validation-target label savings within the tested budget.

These findings motivated further model diagnostics and CNN fine-tuning. The earlier frozen-feature results and current CNN results use different training protocols and should not be interpreted as a controlled comparison of acquisition strategies.

## Current CNN Comparison

The current experiment compares four image-model training schemes at a shared training-label budget.

| Setting | Value |
|---|---|
| Training images | 2,000 uniformly sampled images per seed |
| Seeds | 42, 43, 44 |
| Image size | 128 × 128 |
| Maximum epochs | 30 |
| Minimum training length | 8 epochs |
| Head warm-up | 2 epochs |
| Batch size | 32 |
| Optimizer | AdamW |
| Classifier-head learning rate | 0.001 |
| Pretrained-backbone learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Gradient norm limit | 5 |
| Early-stopping criterion | Validation-monitor macro-F1 |
| Early-stopping patience | 6 epochs |
| Early-stop monitor | 800 stratified validation images |
| Reported evaluation | Complete 2,643-image validation set |

Within a seed, all candidate models share the same training images and validation monitor. Across seeds, the labeled subset, initialization randomness, and monitor selection can change.

### Models

- **Frozen ResNet18:** updates only the classifier head.
- **Partial ResNet18:** updates `layer3`, `layer4`, and the classifier head.
- **EfficientNet-B0:** fine-tunes the pretrained backbone and classifier head.
- **MobileNetV3 Small:** fine-tunes the pretrained backbone and classifier head.

Fine-tuned candidates use independent quarter-turn rotations and horizontal/vertical flips. The frozen baseline does not use image augmentation.

Pretrained BatchNorm running statistics remain fixed. Feature-normalization statistics are fitted only on the current labeled training subset.

A fixed ensemble averages the ten-class prediction probabilities from partial ResNet18, EfficientNet, and MobileNet. No ensemble weights are optimized. The ensemble uses the same unique training labels but requires more computation.

## Results Across Three Seeds

All results below are measured on the **complete validation set**. Standard deviations describe variation across seeds; they are not confidence intervals.

| Model | Mean accuracy | Macro-F1, mean ± SD | Class 4 F1 | Class 4 recall | Class 5 F1 |
|---|---:|---:|---:|---:|---:|
| Frozen ResNet18 | 47.21% | 0.4265 ± 0.0057 | 0.1799 | 0.1400 | 0.4174 |
| Partial ResNet18 | 70.66% | 0.6812 ± 0.0077 | 0.6414 | 0.6133 | 0.6339 |
| **EfficientNet-B0** | **74.55%** | **0.7151 ± 0.0165** | **0.6396** | **0.6067** | **0.7037** |
| MobileNetV3 Small | 65.94% | 0.6272 ± 0.0076 | 0.5306 | 0.5400 | 0.5541 |
| **Equal-probability ensemble** | **75.66%** | **0.7255 ± 0.0070** | **0.6693** | **0.6400** | **0.7032** |

EfficientNet was the strongest individual model in all three seeds. The ensemble achieved the highest average macro-F1 with lower overall seed variability.

### EfficientNet by Seed

| Seed | Accuracy | Macro-F1 | Class 4 correct / 50 | Best epoch | Epochs trained |
|---|---:|---:|---:|---:|---:|
| 42 | 76.28% | 0.7328 | 30 / 50 | 18 | 24 |
| 43 | 74.23% | 0.7126 | 30 / 50 | 20 | 26 |
| 44 | 73.14% | 0.7000 | 31 / 50 | 11 | 17 |

Class 4 had 36, 44, and 39 training examples in the respective labeled subsets. Its validation recall remained between 60% and 62%.

The results support pretrained CNN fine-tuning as a practical model foundation. However, difficult categories remain: EfficientNet's average F1 for disturbed galaxies, class 0, is approximately **0.3104**.

The mean EfficientNet training-minus-validation macro-F1 gap is approximately **0.2144**. Early stopping limits further training after validation performance plateaus, but it does not eliminate overfitting.

![Model comparison](results/image_models_2000/figures/model_comparison.png)

Detailed results are available in:

- [Experiment report](results/image_models_2000/image_2000_report.md)
- [Seed-level metrics](results/image_models_2000/metrics_by_seed.csv)
- [Means and standard deviations](results/image_models_2000/metrics_mean_std.csv)
- [Per-class metrics](results/image_models_2000/per_class_metrics.csv)
- [Class supports](results/image_models_2000/class_counts.csv)

## Interpretation and Evaluation Limits

The current experiments support the repeatability of the fine-tuning approach. They do not yet establish annotation savings from active learning.

Several boundaries matter:

- Validation is used for early stopping and model selection. These are development-validation scores rather than independent confirmation.
- All three seeds share the same full validation set. Seed variability does not measure all evaluation-sample uncertainty.
- The 2,000-label budget counts training labels. Validation and test labels are additional fixed evaluation resources.
- ImageNet pretraining is a shared external resource.
- The original frozen-feature experiment evaluated the test partition. Later CNN development did not evaluate it; the project should not describe that partition as never inspected during the entire research process.
- The 512-label pilot and 2,000-label experiment changed multiple settings, so their difference cannot be attributed solely to additional training labels.
- Comparisons with published benchmark scores require matching dataset versions, splits, training budgets, and evaluation protocols.
- The current results apply to the curated Galaxy10 subset and do not establish cross-survey generalization.

## Reproducing the Current Experiment

### Environment

The working environment uses Windows PowerShell, Python 3.11, and CPU PyTorch. CUDA is optional.

Create an environment for a fresh checkout:

```powershell
py -3.11 -m venv .venv

.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

.\.venv\Scripts\python.exe -m pip install -r requirements_active_learning.txt

.\.venv\Scripts\python.exe -m pip install -r image_budget_runner\requirements_image_demo.txt
```

Existing experiments should continue using their working environment. Changing software versions can change numerical results and experiment fingerprints.

### Data Setup

Download the original HDF5 separately and place it at:

```text
data/raw/Galaxy10_DECals.h5
```

In `al_config.json`, check these paths:

```json
{
  "data_file": "data/raw/Galaxy10_DECals.h5",
  "prepared_dir": "data/processed/prepared_20261014T042419_288789Z"
}
```

The full configuration contains additional settings for the earlier active learning pipeline. The current image runner has its own training arguments.

For a fresh checkout, restore the versioned preparation records to the configured location:

```powershell
$Config = Get-Content .\al_config.json | ConvertFrom-Json

New-Item -ItemType Directory -Force -Path $Config.prepared_dir | Out-Null

Copy-Item .\results\data_preparation\* -Destination $Config.prepared_dir -Recurse -Force
```

Alternatively, inspect and prepare the dataset again:

```powershell
.\.venv\Scripts\python.exe .\inspect_data.py
.\.venv\Scripts\python.exe .\prepare_data.py
```

A new preparation writes a timestamped directory. Update `prepared_dir` intentionally and keep each run's audit and indices together.

### Run the Model Comparison

Run seed 42:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_image_2000.ps1
```

Run or extend to three seeds:

```powershell
.\.venv\Scripts\python.exe .\run_image_2000.py --seeds 42 43 44
```

Find the generated report directory:

```powershell
Get-Content .\outputs\latest_image_2000.json
```

Completed models are verified and reused. Interrupted models resume from the last committed epoch. An unfinished epoch is repeated. Avoid concurrent processes writing to the same experiment.

Regenerate the summary without retraining:

```powershell
$Run = Get-Content .\outputs\latest_image_2000.json | ConvertFrom-Json

.\.venv\Scripts\python.exe .\summarize_image_2000.py --campaign $Run.campaign_directory
```

### Legacy Active Learning

```powershell
.\.venv\Scripts\python.exe .\run_pipeline.py --smoke
.\.venv\Scripts\python.exe .\run_pipeline.py
```

These commands run the earlier frozen-ResNet protocol, including its original evaluation sequence. They do not launch fine-tuned EfficientNet active learning.

### Tests

```powershell
.\.venv\Scripts\python.exe -m unittest test_active_learning -v

.\.venv\Scripts\python.exe -m unittest test_image_2000.py -v

.\.venv\Scripts\python.exe -m unittest discover -s image_budget_runner -p test_image_demo.py -v
```

The current image workflow includes checks for label isolation, trainable convolutional parameters, augmentation, checkpoint recovery, metric aggregation, artifact integrity, and completed-seed reuse.

Small synthetic runs validate engineering behavior; their scores are not scientific results.

## Main Files

| File or directory | Purpose |
|---|---|
| `inspect_data.py` | Raw-data inspection |
| `prepare_data.py` | Cleaning, linkage audit, and fixed splitting |
| `al_config.json` | Dataset paths and original AL configuration |
| `run_pipeline.py`, `galaxy_al/` | Frozen-feature active learning implementation |
| `diagnose_models.py` | Classifier and generalization diagnostics |
| `run_head_ablation.py` | Earlier classifier-head comparisons |
| `run_image_demo.py`, `galaxy_image_demo/` | Small image-model screening |
| `run_image_2000.py` | Current shared-budget, multi-seed experiment |
| `run_image_2000.ps1` | Windows execution wrapper |
| `image_budget_runner/` | CNN training, ensemble, and reporting implementation |
| `summarize_image_2000.py` | Cross-seed metrics and figures |
| `results/` | Versioned research summaries and preparation records |
| `outputs/` | Generated local experiment artifacts |
| `cache/` | Generated image/features caches and downloaded weights |

The `.gitignore` excludes raw datasets, virtual environments, caches, and large model artifacts. Research tables and selected figures are copied into `results/` for publication.

## Next Steps

The next stage will keep the selected CNN training scheme fixed and compare random, uncertainty, diversity, and hybrid acquisition under matched conditions.

The planned evaluation will examine:

- Macro-F1 learning curves across multiple label budgets.
- Class 4 precision, recall, and F1.
- Performance across all ten classes.
- The number and distribution of labels acquired by each strategy.
- Labels required to reach a prespecified performance target.
- Training and acquisition computation.

Annotation savings will be reported only when the relevant methods reach the specified target under the same evaluation rules. A hybrid strategy that fails to outperform random sampling remains a valid experimental finding.

## Acknowledgments

- [Galaxy10 DECaLS and astroNN](https://astronn.readthedocs.io/en/latest/galaxy10.html)
- [Galaxy10 dataset construction repository](https://github.com/henrysky/Galaxy10)
- [Galaxy Zoo](https://www.galaxyzoo.org/)
- [DESI Legacy Imaging Surveys](https://www.legacysurvey.org/)
- [PyTorch](https://pytorch.org/) and [torchvision](https://docs.pytorch.org/vision/stable/)

Dataset and pretrained-model resources retain their original terms and attribution requirements. This project is an application study; it does not claim a new CNN architecture or a previously unknown acquisition algorithm.