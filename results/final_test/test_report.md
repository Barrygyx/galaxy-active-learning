# Galaxy10 Held-Out Test Evaluation

Test images: 2642; training label budget: 2000 per model.
No test-set training, hyperparameter tuning, or ensemble weight fitting.

| Model | Test Accuracy (mean ± SD) | Test Macro F1 (mean ± SD) |
|---|---:|---:|
| resnet18_frozen | 0.4759 ± 0.0057 | 0.4280 ± 0.0127 |
| resnet18_partial_aug | 0.7073 ± 0.0071 | 0.6796 ± 0.0068 |
| mobilenet_small_aug | 0.6535 ± 0.0094 | 0.6116 ± 0.0023 |
| efficientnet_b0_aug | 0.7391 ± 0.0120 | 0.7052 ± 0.0096 |
| equal_ensemble_3 | 0.7541 ± 0.0118 | 0.7238 ± 0.0078 |

SD is across experiment seeds, not a test-sampling confidence interval.
Each seed ensemble averages ResNet18 Partial, MobileNet Small and EfficientNet-B0 probabilities equally.

See per_seed_metrics.csv, per_class_metrics.csv, predictions.csv, and confusion matrices.
