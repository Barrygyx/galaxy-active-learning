# Galaxy10：直接训练图像的 CNN Demo

这次真正训练卷积层，不再只调缓存特征上的 MLP。保留旧数据划分、原始 HDF5、原 AL 实验和旧特征缓存。

## 先运行 5 个模型的小规模对照

把 ZIP 内文件和 `galaxy_image_demo` 文件夹放到 `C:\Users\Barry\Desktop\galaxy-active-learning` 根目录，与原 `run_pipeline.py` 同一层。

```powershell
cd ([Environment]::GetFolderPath("Desktop"))
cd .\galaxy-active-learning
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_image_demo.ps1
```

默认 quick：同一组 1,000 张 train_pool 图、seed 42、128×128、最多 10 epochs、batch 32。每轮用同一组 600 张分层抽取的验证图早停，最后每个候选都在完整 2,643 张验证图上评分。训练过程监测 256 张训练图，最终训练指标在全部本次训练图上计算。

**真实训练比原来的 MLP 更耗 CPU。** 每个 epoch 和 batch 会打印进度及耗时；不会承诺固定完成时间。自动使用可用 CUDA，否则 CPU；AMD RX6700S 的 Windows 环境默认走 CPU。

### 全部 9 个组合

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_image_demo.ps1 -Suite all
```

quick 与 all 的共同模型自动复用，不再训练。各候选的结果分目录记录，合并报告在 reports/quick 或 reports/all。

| 模型 | 起点 | 哪些卷积参数学习 | 训练增强 | Suite |
|---|---|---|---|---|
| tiny_cnn_aug | 小 CNN 随机初始化 | 全部 | 旋转/翻转 | quick |
| resnet18_frozen | ImageNet ResNet18 | 无，仅分类头 | 无 | quick |
| resnet18_partial_aug | ImageNet ResNet18 | layer3、layer4 | 旋转/翻转 | quick |
| resnet18_full_aug | ImageNet ResNet18 | 整个 backbone | 旋转/翻转 | quick |
| mobilenet_small_aug | ImageNet MobileNetV3 Small | 整个 backbone | 旋转/翻转 | quick |
| resnet18_scratch_aug | ResNet18 随机初始化 | 全部 | 旋转/翻转 | all |
| resnet18_partial_noaug | ImageNet ResNet18 | layer3、layer4 | 无 | all |
| efficientnet_b0_aug | ImageNet EfficientNet-B0 | 整个 backbone | 旋转/翻转 | all |
| resnet18_partial_balanced_aug | ImageNet ResNet18 | layer3、layer4 | 旋转/翻转＋类别权重 | all |

这些组合分别检查：从零训练与迁移学习、固定与可学习图像表征、增强、不同架构、类别加权。

PowerShell 入口还会自动做等权概率集成：quick 使用部分微调 ResNet18＋MobileNet，all 再加入 EfficientNet。没有搜索集成权重或训练新模型，训练标签数仍相同，但总模型训练/推理成本更高。结果目录记录在 outputs/latest_image_ensemble.json。

如果直接用 Python 跑模型，完成后可单独计算集成：

```powershell
.\.venv\Scripts\python.exe analyze_image_ensemble.py
```

所有模型使用同样的 128 维隐藏层分类头。小 CNN 使用 GroupNorm 和保留 2×2 空间网格的池化；ResNet 等保留各自默认池化。此处比较训练方案，不是每个参数都严格匹配的架构消融。

## 早停和微调

- 只根据验证 monitor 的 macro-F1 早停，patience=4、min_delta=0.0001。CE 仅监测，不触发上次那种硬停止。
- 全部已观察 epoch 中最高 monitor F1 的 checkpoint 被恢复；平分时选 monitor CE 更低者。
- 预训练可微调模型先进行 2 epochs 分类头 warm-up，再更新选中的卷积层。至少训练 6 epochs，给微调阶段学习机会。
- AdamW：head lr=0.001；预训练 backbone lr=0.0001；scratch lr=0.001；weight decay=0.0001；梯度范数上限 5。
- 预训练 BatchNorm 的 running statistics 固定；被选中卷积块中的 affine 参数仍训练。scratch ResNet 的 BN 正常更新。
- 所有预训练模型都用本次训练图的初始特征 mean/std 归一化。这些统计仅由当前 1,000 张已标注训练图拟合，后续微调时固定；scratch 使用恒等特征归一化。
- frozen 用一次性初始特征缓存加速分类头训练；其余模型每批真正通过图像和可学习卷积层反向传播。
- 增强：每张训练图独立进行 90°倍数旋转、水平/垂直翻转。验证不增强。没有随机裁剪或颜色变化。
- balanced 权重来自本次训练标签的平方根逆频率，未见类别权重为 0；不查看未查询训练标签来增加稀有类样本。

## 所需原文件

`al_config.json` 中原 data_file 和 prepared_dir 路径保持有效；需要原 HDF5 和准备目录的全部审计文件。无需旧的 CNN 特征缓存或 AL checkpoint。

只从 train_pool 均匀随机选训练图，不根据类别平衡抽取；不同训练预算使用同一随机排列的前缀，因此更大预算包含小预算。验证 monitor 独立于训练预算，保持相同。验证标签仅用于分层选择 monitor 和评估。

原始图像按 SHA256 验证后，以完整视野缩放、float16 [0,1] 缓存在新的 cache/image_demo；训练批次转换 float32，再做固定 ImageNet 归一化。float16 缓存有微小量化误差，所有候选统一使用。

第一个完整 image cache 默认约 360 MB；各模型完成后保留一个最佳 checkpoint，中间进度在完成后删除。预训练权重首次使用自动下载（ResNet18 可复用旧权重），网络失败不会偷偷改成随机权重。

## 扩大有潜力的实验

先看小 demo，再考虑：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_image_demo.ps1 -Suite all -TrainSize 2000 -ImageSize 224 -Epochs 20
```

这会创建新实验目录、图像缓存，计算量明显上升。无需一开始运行。只训练指定模型也可以：

```powershell
.\.venv\Scripts\python.exe run_image_demo.py --models resnet18_partial_aug efficientnet_b0_aug --train-size 1000
```

改变 seed、分辨率、预算、训练规则或代码会形成新实验身份；仅增加 suite/model 列表可复用共同候选。所有输出位于 outputs/image_demo，最后路径写入 outputs/latest_image_demo.json。

## 中断和恢复

每个 epoch 结束后保存模型、optimizer、最佳模型和随机数状态；重新执行同一命令可从已提交 epoch 继续，已完成模型直接复用。未完成的当前 epoch 会重做。不要同时运行两个写同一目录的进程。

进度文件可能较大，因为包含 AdamW 状态及当前/最佳卷积参数；完成后会删除它，只保留最佳模型与预测。请留出至少几 GB 空间用于图像缓存及多种模型。

## 看哪些结果

终端最后给出 `reports/quick/image_demo_report.md` 或 `reports/all/image_demo_report.md` 的完整路径。请发回：

1. image_demo_report.md
2. model_comparison.csv
3. figures/image_model_comparison.png
4. figures/image_training_curves.png
5. figures/image_class4_class5.png

表格同时报告完整验证集的 accuracy、macro-F1、class 4、class 5、CE、训练时间与选中 epoch。不要把 accuracy 当作 macro-F1。

## 结果边界

- 这是模型可行性 pilot，不是重新运行主动学习。若某模型有收益，之后再让它参与每轮选样。
- 默认 1,000 图、128 像素、单 seed，不直接与原 2,000-label 查询轨迹或论文的全数据 accuracy 比分。需要在同一协议下比较。
- 早停 monitor 是完整 validation 的子集。最终完整 validation 也用于选方法，因此它不是独立确认集。代码不评估 test。
- 小 monitor 的稀有类数量少，单 seed 排名不稳定。报告同时给出各类别支持数，最终 class 4 分数使用完整 validation 的 50 张图。
- 训练最多 10 epochs 是节省计算的初筛，到达上限不代表该架构充分收敛。从零训练模型尤其可能需要更多数据和训练。
- 不直接使用已在 Galaxy10 上训练的公开权重；这样避免把额外任务标签混入小标签预算。
- 早停和增强不能保证消除过拟合，也不保证达到文献成绩。

## 工程测试

```powershell
.\.venv\Scripts\python.exe -m unittest test_image_demo.py -v
```

仅测试接口（不作为研究结果）可运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_image_demo.ps1 -Suite all -SmokeOnly
```

smoke 使用至多 40 训练/20 monitor、64 像素、2 epochs，保留完整验证集评分，输出到独立 smoke 目录。它仍加载真实官方预训练权重，检查全部网络能否训练。

包内 `real_pilot_results` 还包含已实际运行的真实 Galaxy10 小实验：官方原文件 SHA256 已核对，沿用你的原清洗划分，512 个训练标签、400 张早停验证图、完整 2,643 张验证图、128 像素、最多 6 epochs、seed 42。包含 9 个模型和一个三模型等权集成，测试集未评估。实际分数与限制见 `real_pilot_results/SUMMARY_REAL_PILOT.md`。

本包默认本机运行参数为 1,000 标签/10 epochs，和附带 512 标签结果不同。复现实验包内的 512 标签设置：

```powershell
.\.venv\Scripts\python.exe run_image_demo.py --suite all --train-size 512 --monitor-size 400 --image-size 128 --epochs 6 --min-epochs 4 --warmup-epochs 1 --batch-size 32 --threads 4 --device cpu
.\.venv\Scripts\python.exe analyze_image_ensemble.py
```

合成 smoke 测试仅验证工程行为；真实 pilot 是单 seed 的探索性结果，不能作为稳定泛化优势或主动学习节省标签的证据。
