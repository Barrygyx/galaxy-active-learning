# 真实 Galaxy10 图像模型小实验

已经在真实图像上完成 9 个训练方案，以及一个三模型等权概率集成。结果支持继续研究可学习的图像表征；稀有类问题仍未解决。

## 相同的数据与训练预算

- 沿用原数据清洗和固定划分：train_pool 12,330；validation 2,643；test 2,642。
- 从 train_pool 均匀随机选择相同的 512 张图，seed 42，不使用类别标签来挑训练图。
- 128×128 完整视野缩放；最多 6 epochs；预训练可微调模型先 warm-up 1 epoch；至少运行 4 epochs；patience 4。
- 用原 validation 中固定的 400 张图监测 macro-F1 并恢复最佳 checkpoint；表中分数来自完整 2,643 张 validation。
- 所有候选使用同一子集；不评估 test。完整 validation 用于模型比较，属于开发数据。
- 预训练权重来自 ImageNet；没有使用已在 Galaxy10 上训练过的权重。

## 实际结果

Accuracy 用百分数；macro-F1 与类别 F1 用 0–1。

| 方法 | Validation accuracy | Macro-F1 | Class 4 F1 | Class 5 F1 |
|---|---:|---:|---:|---:|
| 三模型等权集成 | 61.03% | 0.5226 | 0.0000 | 0.5147 |
| ResNet18 部分微调＋增强 | 56.75% | 0.4828 | 0.0000 | 0.5038 |
| EfficientNet-B0 微调＋增强 | 56.53% | 0.4822 | 0.0385 | 0.4848 |
| MobileNetV3 Small 微调＋增强 | 54.52% | 0.4679 | 0.0000 | 0.4255 |
| ResNet18 部分微调，无增强 | 50.89% | 0.4502 | 0.0000 | 0.3941 |
| ResNet18 部分微调＋增强＋类别加权 | 51.19% | 0.4443 | 0.1644 | 0.4536 |
| ResNet18 全部微调＋增强 | 52.25% | 0.4417 | 0.0000 | 0.2694 |
| ResNet18 冻结卷积层，仅训练分类头 | 42.60% | 0.3617 | 0.0000 | 0.3448 |
| ResNet18 从零训练＋增强 | 30.38% | 0.2470 | 0.0000 | 0.1643 |
| 小 CNN 从零训练＋增强 | 23.31% | 0.0786 | 0.0000 | 0.0000 |

集成对部分微调 ResNet18、MobileNet、EfficientNet 的十类预测概率等权平均，没有搜索权重，也没有新增训练标签；总计算成本高于单个模型。

## 能支持什么判断

1. 相同的 512 张训练图上，冻结 ResNet18 到无增强的部分微调，macro-F1 从 0.3617 提升到 0.4502；再加入旋转/翻转增强提升到 0.4828。这个 pilot 表明，允许卷积层适配星系图像有价值，旧冻结特征方案存在限制。
2. 低预算下，部分微调优于本次全部微调；这只描述当前超参数与训练时长，不代表架构最终排名。
3. MobileNet 是计算受限时值得保留的候选。训练耗时不能直接移植到用户的 Windows/AMD 笔记本。
4. 从零训练只有 512 张图、最多 6 epochs，训练不足的可能性很大。较低分数不能证明 CNN 不适合该数据。
5. 训练子集中 class 4 只有 6 张，完整 validation 有 50 张。最佳总体方案的该类 F1 为 0，类别加权能提高到 0.1644，但整体 macro-F1 降低。原 RQ 所需的稀有类识别尚未达到要求。

## 不能据此宣称什么

- 单 seed 探索性结果没有估计随机波动；多个候选也使用了同一个验证集。
- 不能与旧 2,000-label AL 轨迹、全标签参考或文献的 73.9% accuracy 直接比较：数据预算、训练方案和评估协议不同。
- 当前集成的 accuracy 61.03% 不是 macro-F1 0.6103，实际 macro-F1 为 0.5226。
- 部分候选的最佳 epoch 正好为上限 6，尚不能判断收敛上限。
- 这还不是主动学习实验，不证明节省标签，也不证明保留了稀有类表现。

## 建议的下一阶段

优先在相同的 2,000 张训练图上比较部分微调 ResNet18、EfficientNet-B0、MobileNet，增加训练上限但保留验证早停，并使用多个随机种子。继续同时报告 macro-F1、class 4/5 F1。先确定可靠的图像训练方案，再接回 random/uncertainty/diversity/hybrid 主动学习流程；所有查询策略必须共用相同训练规则、预算和初始标签。

不要根据尚未查询的标签挑选稀有类来改善主动学习曲线。类别加权只能使用当前已标注子集的标签。

## 复现这次 512 张图实验

将包内容放入原 galaxy-active-learning 根目录。保留已有 al_config.json、原 HDF5 和 prepared 目录。先按 README 安装依赖，再执行：

```powershell
.\.venv\Scripts\python.exe run_image_demo.py --suite all --train-size 512 --monitor-size 400 --image-size 128 --epochs 6 --min-epochs 4 --warmup-epochs 1 --batch-size 32 --threads 4 --device cpu
.\.venv\Scripts\python.exe analyze_image_ensemble.py
```

本包默认 PowerShell 参数是 1,000 张图／10 epochs，与本页结果的 512／6 不同。不同设备和软件版本也可能带来数值差异。

逐类结果见 per_class_metrics.csv；全部指标见 all_methods_including_ensemble.csv；图表在 figures；子集原始行号见 subsets.csv；配置和核验记录见 locked_protocol.json、planned_protocol.json、verification.json。大体积原始 HDF5、预训练权重和已训练模型不在交付包中，脚本可重新获取权重和训练。
