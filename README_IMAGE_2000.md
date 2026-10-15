# Galaxy10：2,000 标签图像模型对照

这次对照检查：增加可用标签和训练时间后，CNN 微调能否提高整体和稀有类别的识别能力。不是主动学习重跑，也不预设一定达到论文成绩。

## 放到哪里

把本包**内部的全部文件和 image_budget_runner 文件夹**放入：

```text
C:\Users\Barry\Desktop\galaxy-active-learning
```

run_image_2000.ps1 应与原 al_config.json、.venv、data、outputs 在同一层。本包使用独立的 image_budget_runner 保存完整训练代码，不覆盖旧 run_image_demo.py 或旧 galaxy_image_demo。原 HDF5、prepared 清洗记录和 al_config.json 必须仍可访问。

## 直接运行

```powershell
cd ([Environment]::GetFolderPath("Desktop"))
cd .\galaxy-active-learning
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_image_2000.ps1
```

先使用已经工作的 .venv，检查 torch/torchvision 可导入，安装少量其余依赖。不会主动重装 PyTorch。默认 CPU，适用于当前 Windows/AMD 环境；每轮显示进度，训练总耗时以本机实测为准。

## 本轮固定设置

| 项目 | 设置 |
|---|---|
| 模型 | 冻结 ResNet18、部分微调 ResNet18、EfficientNet-B0、MobileNetV3 Small |
| 训练标签 | 从原 train_pool 均匀随机取 2,000 张，同一 seed 内四个模型完全相同 |
| 随机种子 | 默认先做 42；可补 43、44 |
| 图像 | 128×128、完整视野缩放 |
| 训练上限 | 30 epochs，至少 8 epochs |
| 分类头 warm-up | 前 2 epochs；之后更新预定卷积层 |
| 早停 | 固定 800 张分层 validation monitor，macro-F1 patience=6，min_delta=0.0001 |
| 最佳 checkpoint | 所有观察到的 epoch 中 monitor macro-F1 最高；平分时 CE 较低者 |
| 最终开发评分 | 原完整 validation，实际行数与支持数自动记录；用户当前为 2,643 |
| 训练 | AdamW，head lr=0.001、backbone lr=0.0001、weight decay=0.0001、batch=32、clip norm=5 |
| 增强 | 微调模型使用 90°倍数旋转及翻转；冻结对照无增强 |
| 预训练 | 官方 ImageNet 权重；BN running stats 固定；未使用 Galaxy10 预训练权重 |
| 集成 | 三个微调模型固定等权概率平均，不搜索权重，不增加独特训练标签数 |
| Test | 不评估 |

学习率、结构和特征归一化沿用上一版。报告生成器纠正了旧的“默认 1,000 标签”文字，全部按实际预算解释；不对论文的具体设置作未经核对的断言。

与旧 512-label demo 相比，本轮同时改变标签数、最大训练长度、warm-up、patience 和 monitor 大小，因此这是下一阶段的可行性对照，不能单独分离每个因素的因果作用。原 2,000-label AL 查询样本不保证与本轮随机前缀相同。

每个 seed 的子集、模型、缓存、软件版本及代码哈希都被记录。模型归一化只由该 seed 的这 2,000 张训练图拟合。没有使用未查询训练标签来挑选稀有类。

800 张 monitor 属于原 validation；完整 validation 同时用于模型比较，因此评分不是独立确认。增大 monitor 旨在减少旧 400 张 monitor 上稀有类过少的影响，但实际稀有类数量仍须查看 class_counts.csv。

## 结果位置与回传文件

运行结束会打印完整路径，也可查看：

```powershell
Get-Content .\outputs\latest_image_2000.json
```

汇总位于 outputs/image_2000/comparison_.../，包含：

- image_2000_report.md：总体结论所需指标、每个 seed 的停止原因、稀有类支持数及解释边界。
- metrics_by_seed.csv：各模型与集成的 train/validation accuracy、macro-F1、class 4/5 F1/recall、训练验证差距。
- metrics_mean_std.csv：各 seed 的均值与样本标准差。只有一个 seed 时 SD=0 表示没有重复，不表示稳定。
- per_class_metrics.csv：十类逐类 precision/recall/F1/support，包含集成。
- class_counts.csv：实际 train、monitor、validation 各类图数。
- paired_differences_vs_frozen.csv：同一 seed 内各方法相对冻结基线的差值。
- top_confusions.csv：class 0、4、5、6 的主要误判去向。
- figures/model_comparison.png、rare_and_barred_f1.png。
- figures/training_curves_seed_42.png：训练/验证 monitor F1、CE、class 4/5 F1；虚线为选中 epoch。
- figures/per_class_f1_seed_42.png、confusions_seed_42.png：逐类表现与全部五种方法的混淆矩阵。

图表同时保存 PDF。请优先回传 image_2000_report.md、metrics_by_seed.csv、per_class_metrics.csv、class_counts.csv，以及 training_curves_seed_42.png 和 confusions_seed_42.png。

原始每模型结果仍在 outputs/image_demo/pilot_... 中，runs.json 记录确切路径。所有训练/验证预测、最佳权重和逐 epoch 历史会保留。

## 中断恢复与追加种子

重新执行同一命令：已完成模型校验后复用，未完成模型从最后提交的 epoch 恢复；当前未提交 epoch 会重做。请勿并行启动多个 image demo 写同一项目。图像缓存和多个 checkpoint 需要预留几 GB 空间。

默认 seed 42 完成后，如果结果值得进一步验证，可追加两个种子。使用同样代码、环境和配置时，下面命令复用已完成的 42。直接使用 Python 入口可避免 Windows PowerShell 子进程的数组参数解析差异：

```powershell
.\.venv\Scripts\python.exe .\run_image_2000.py --seeds 42 43 44
```

每完成一个 seed 都更新汇总；seeds 清单中的请求与已完成清单分别记录。均值/SD 汇总 runs.json 中同一协议下的全部已完成 seeds。多 seed 改变训练抽样、初始化及 monitor；该 SD 表示整套流程的变动，不能当作独立测试图像带来的不确定性。

只重新生成汇总、不训练：

```powershell
$Result = Get-Content .\outputs\latest_image_2000.json | ConvertFrom-Json
.\.venv\Scripts\python.exe .\summarize_image_2000.py --campaign $Result.campaign_directory
```

## 下一步如何判断

先看同预算微调相对冻结基线的改善，再看 class 4 的 recall/F1、class 5 及 class 0/6，检查训练验证差距和是否仍碰到上限。不根据验证表现删除困难类别或改变原清洗规则。

本轮仍没有学习到稀有类时，要区分稀有训练支持不足、类别混淆和优化问题；不会把低分自动归咎于数据标签。当前只有课程项目级可行性验证，尚未证明主动学习标签节省。

本包不附带新的 2,000-label 真实成绩，也不保证 F1 一定上升；需要在本机运行后再分析。工程检查记录见 VALIDATION_IMAGE_2000.md。
