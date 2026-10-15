# Galaxy10 同预算图像模型对照

训练标签=2000；分辨率=128；最大 epochs=30；patience=6；已完成 seeds=3。

这是模型开发实验。所有单模型在每个 seed 内共用相同训练图、验证 monitor 和训练规则；原 test 未评估。
训练图仍从原 train_pool 均匀随机抽取，未使用隐藏类别挑选初始标签。跨 seed 会改变训练子集、初始化及 monitor。
集成对三个预先指定的微调模型概率等权平均，标签预算不变，但计算成本高于单模型。

| 方法 | Accuracy mean | Macro-F1 mean ± SD | Class 4 F1 mean | Class 4 recall mean | Class 5 F1 mean | Train−val F1 gap mean |
|---|---:|---:|---:|---:|---:|---:|
| resnet18_frozen | 0.4721 | 0.4265 ± 0.0057 | 0.1799 | 0.1400 | 0.4174 | 0.4623 |
| resnet18_partial_aug | 0.7066 | 0.6812 ± 0.0077 | 0.6414 | 0.6133 | 0.6339 | 0.1380 |
| efficientnet_b0_aug | 0.7455 | 0.7151 ± 0.0165 | 0.6396 | 0.6067 | 0.7037 | 0.2144 |
| mobilenet_small_aug | 0.6594 | 0.6272 ± 0.0076 | 0.5306 | 0.5400 | 0.5541 | 0.2473 |
| equal_ensemble_3 | 0.7566 | 0.7255 ± 0.0070 | 0.6693 | 0.6400 | 0.7032 | 0.2156 |

单 seed 的 SD 显示 0 只是没有重复测量，不能解释为结果稳定。多 seed 的 SD 是样本标准差，不是置信区间或显著性检验。

## 每个模型的停止情况

| Seed | 方法 | 最佳 epoch | 已训练 epochs | 停止原因 |
|---:|---|---:|---:|---|
| 42 | resnet18_frozen | 9 | 15 | validation_macro_f1_patience |
| 42 | resnet18_partial_aug | 12 | 18 | validation_macro_f1_patience |
| 42 | efficientnet_b0_aug | 18 | 24 | validation_macro_f1_patience |
| 42 | mobilenet_small_aug | 14 | 20 | validation_macro_f1_patience |
| 43 | resnet18_frozen | 9 | 9 | validation_macro_f1_patience |
| 43 | resnet18_partial_aug | 23 | 29 | validation_macro_f1_patience |
| 43 | efficientnet_b0_aug | 20 | 26 | validation_macro_f1_patience |
| 43 | mobilenet_small_aug | 13 | 19 | validation_macro_f1_patience |
| 44 | resnet18_frozen | 7 | 13 | validation_macro_f1_patience |
| 44 | resnet18_partial_aug | 19 | 25 | validation_macro_f1_patience |
| 44 | efficientnet_b0_aug | 11 | 17 | validation_macro_f1_patience |
| 44 | mobilenet_small_aug | 19 | 25 | validation_macro_f1_patience |

## 稀有类支持数

| Seed | Split | Class 4 图数 | Class 5 图数 |
|---:|---|---:|---:|
| 42 | train | 36 | 228 |
| 42 | monitor | 15 | 93 |
| 42 | validation | 50 | 306 |
| 43 | train | 44 | 227 |
| 43 | monitor | 15 | 93 |
| 43 | validation | 50 | 306 |
| 44 | train | 39 | 227 |
| 44 | monitor | 15 | 93 |
| 44 | validation | 50 | 306 |

## 解释边界

- 同预算对照主要看 paired_differences_vs_frozen.csv；class 4 和总体 F1 必须一起检查。
- 相比旧 512-label demo，本轮同时增加标签、训练上限和验证 monitor，不能把变化单独归因于其中一个因素。
- 早停 monitor 包含在完整 validation 中，完整 validation 也用于比较方法；没有独立确认成绩。
- 到达 max_epochs 不代表模型已经收敛；早停也不能保证消除过拟合。观察训练曲线、最佳 epoch 和逐类结果。
- 训练 F1 来自恢复后的完整训练子集；曲线中的 train monitor 通常只有 256 张，因此终点数值可不同。
- 该实验尚未执行主动学习；不能据此宣称节省标签、达到论文 accuracy 或证明某种选样策略有效。
- 该流程读取原清洗审计记录中的标签和哈希以验证文件，但模型拟合、归一化和训练权重只使用当前训练子集。

## 回传结果

- image_2000_report.md
- metrics_by_seed.csv
- per_class_metrics.csv
- class_counts.csv
- figures/training_curves_seed_42.png
- figures/confusions_seed_42.png
