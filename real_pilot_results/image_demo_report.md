# Galaxy10 图像模型 Demo

**这是单 seed、小预算开发实验；不是完整主动学习结果。**

Train labels=512；early-stop validation monitor=400；最终完整 validation=2643；image size=128。

所有方法共享相同训练图和验证划分。训练子集均匀随机抽取，没有用真实类别平衡初始标注。测试集未评估。

## 完整验证集结果

| Model | Accuracy | Macro-F1 | Class 4 F1 | Class 5 F1 | CE | Best epoch / trained | Training min |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18_partial_aug | 0.5675 | 0.4828 | 0.0000 | 0.5038 | 1.2100 | 6 / 6 | 0.8 |
| efficientnet_b0_aug | 0.5653 | 0.4822 | 0.0385 | 0.4848 | 1.2265 | 6 / 6 | 0.9 |
| mobilenet_small_aug | 0.5452 | 0.4679 | 0.0000 | 0.4255 | 1.2892 | 5 / 6 | 0.2 |
| resnet18_partial_noaug | 0.5089 | 0.4502 | 0.0000 | 0.3941 | 1.4127 | 5 / 6 | 0.8 |
| resnet18_partial_balanced_aug | 0.5119 | 0.4443 | 0.1644 | 0.4536 | 1.3204 | 6 / 6 | 0.7 |
| resnet18_full_aug | 0.5225 | 0.4417 | 0.0000 | 0.2694 | 1.3252 | 4 / 6 | 1.1 |
| resnet18_frozen | 0.4260 | 0.3617 | 0.0000 | 0.3448 | 1.5906 | 4 / 6 | 0.0 |
| resnet18_scratch_aug | 0.3038 | 0.2470 | 0.0000 | 0.1643 | 1.8328 | 4 / 6 | 1.2 |
| tiny_cnn_aug | 0.2331 | 0.0786 | 0.0000 | 0.0000 | 2.0328 | 6 / 6 | 0.3 |

本次完整验证集 macro-F1 排名第一：**resnet18_partial_aug**。单 seed 排名可能随样本和初始化变化，不代表稳定优势。

## 类别支持数

| Class | Train | Monitor | Full validation |
|---|---:|---:|---:|
| 0: Disturbed | 31 | 24 | 158 |
| 1: Merging | 61 | 42 | 276 |
| 2: Round smooth | 76 | 60 | 395 |
| 3: In-between round smooth | 71 | 46 | 303 |
| 4: Cigar shaped smooth | 6 | 8 | 50 |
| 5: Barred spiral | 55 | 46 | 306 |
| 6: Unbarred tight spiral | 53 | 41 | 273 |
| 7: Unbarred loose spiral | 73 | 59 | 389 |
| 8: Edge-on without bulge | 27 | 32 | 213 |
| 9: Edge-on with bulge | 59 | 42 | 280 |

## 训练规则

- 真正训练卷积参数：scratch 从零开始；partial 更新 ResNet layer3/layer4；full 更新整个 backbone。frozen 只训练分类头，作为对照。
- 所有方法用同一个 128 维隐藏层分类头；预训练模型用当前已标注训练图的初始特征 mean/std 固定归一化，scratch 使用恒等特征归一化。
- 微调有 head warm-up，随后使用较小的 backbone 学习率；预训练 BN 的 running stats 固定，选定层的 affine 参数可学习。
- 增强只在训练时应用独立的 90°倍数旋转、水平/垂直翻转；没有裁剪或颜色扰动。
- balanced 变体只根据当前训练标签计算平方根逆频率权重；不增加标签预算。
- 早停只监控验证 monitor 的 macro-F1，保留观察到的最佳 checkpoint；CE 仅记录，没有上一轮的 CE 强制停止条件。
- 为保证微调阶段有学习机会，warm-up 后有最小训练长度。小预算 demo 到达最大 epoch 并不证明模型充分收敛。

## 怎样与旧结果和文献比较

本次默认 1,000 个训练标签、128 像素与单 seed，训练样本也不一定与原 Random 查询轨迹相同，不能直接与旧 2,000-label 分数判断谁优谁劣。应先比较本表内的共同协议结果。
文献的 73.9% 是其他图像构造和划分、更多训练数据下的 accuracy，不是本 demo 的目标 macro-F1。此处同时报告两个指标以免混淆。
monitor 是完整 validation 的子集，完整 validation 的评分并非完全独立确认：其中部分图像已参与早停，整体又用于候选比较。
如果模型有明显潜力，之后应提高分辨率/训练预算、增加 seeds，再在原 train_pool 内重新执行主动学习。不得用全标签微调后的 encoder 冒充小标签预算。

## 回传文件

- `image_demo_report.md`
- `model_comparison.csv`
- `figures/image_model_comparison.png`
- `figures/image_training_curves.png`
- `figures/image_class4_class5.png`
