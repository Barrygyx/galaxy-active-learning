# Galaxy10 DECaLS: uncertainty, diversity, and rare-class coverage

## 现在怎么运行

将 ZIP 内的文件直接解压到你现有的 `galaxy-active-learning` 项目根目录。
正确的位置是 `galaxy-active-learning/run_pipeline.py`，不要再多套一层同名目录。
保留现有的 `.venv`、`data/raw/` 和已通过验收的 `data/processed/`。
本代码不重新清理数据、不重新划分数据，也不会覆盖原始 HDF5。

PowerShell：

```powershell
cd ([Environment]::GetFolderPath("Desktop"))
cd .\galaxy-active-learning
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1
```

这条命令依次安装依赖、运行一个小型集成检查、运行正式实验。
`ExecutionPolicy Bypass` 只作用于这次子进程，不修改系统的永久策略。
脚本遇到失败会停止，不会把失败的 smoke test 当成正式结果。

默认输入：

```text
data/raw/Galaxy10_DECals.h5
data/processed/prepared_20261014T042419_288789Z/
```

如果你移动/改名了这两处路径，只修改 `al_config.json` 的 `data_file` 和 `prepared_dir`。
脚本根据自身位置解析相对路径，与终端当前目录无关。
启动前会检查清理文件校验和、CSV/NPY 一致性、重复组、源数据尺寸和标签。
提取特征时还会逐张核对图片哈希，防止数据文件和索引不匹配。

首次运行需要联网下载 PyTorch 和约 45 MB 的 ImageNet 预训练权重。
权重自动保存在 `cache/torch/`。断网或下载失败时会明确停止，不会改用随机权重。
本版默认 CPU，适用于没有可用 CUDA 的 Windows 环境，不要求 NVIDIA 显卡。

## 断点续跑

安装成功后，中断或重启可直接运行：

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

已完成的特征批次会复用；已经完整保存的训练轮次也会复用。
正在执行但尚未提交的训练轮次会从头重跑。程序不承诺逐个 minibatch 续训。
请不要同时打开两个进程运行同一个项目；它们会写入同一个缓存/实验目录。
修改协议参数或关键算法代码会生成新的实验目录，避免把不同实验混在一起。

其他命令：

```powershell
# 只检查小型流程；结果明确标为 smoke，不用于研究结论
.\.venv\Scripts\python.exe run_pipeline.py --smoke

# 不重复安装，按 smoke + 正式实验的顺序启动
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -SkipInstall

# 算法级自检（不需要下载数据或预训练权重）
.\.venv\Scripts\python.exe -m unittest test_active_learning -v
```

## Research question

Can combining prediction uncertainty with sample diversity reduce annotation
effort while preserving recognition of rare galaxy morphology classes?

本项目检验这一假设，不预设 hybrid 一定胜出。默认组合方法也是已有思想的
可复现实验实现，不应未经文献核对就宣称算法首次提出或达到 SOTA。

## 固定实验协议

| 项目 | 默认设置 |
|---|---|
| 数据 | 已验收的 17,615 张清理后图片，原始 HDF5 索引不变 |
| 数据划分 | 训练池 12,330；验证 2,643；测试 2,642 |
| 初始标注 | 每个 seed 在训练池均匀随机抽 200 张，不利用隐藏类别做平衡 |
| 查询批次 | 每轮 200 张，总标注量到 2,000 张 |
| 随机种子 | 42、43、44；每个 seed 下四种策略共享初始样本 |
| 策略 | random、uncertainty、diversity、hybrid |
| 稀有类别 | 预先指定第 4 类；不是运行后根据效果挑选 |
| 主指标 | macro-F1 学习曲线与归一化 AULC |
| 稀有类指标 | recall、precision、F1，以及已查询样本中的稀有类数量 |
| 附加对照 | 每个 seed 另训一个使用全部训练池标签的 reference |

初始 200 张计入训练标注预算。验证、测试标签是固定的额外标注资源；
报告标签节省时必须单独披露这两部分成本。ImageNet 标签则是所有方法共同使用的外部预训练资源。

## 模型与预处理

1. 原始 uint8 图片全画幅 resize 到 224 × 224，不做中心裁剪，不随机增强。
2. 像素缩放到 [0,1]，再使用固定 ImageNet mean/std。
3. 使用 `ResNet18_Weights.IMAGENET1K_V1`，移除最后分类层，提取 512 维特征。
4. encoder 全程冻结且保持 eval 模式。验证和测试图片只做前向计算，不训练
   encoder、不更新 BatchNorm，不使用这些图片拟合表示或标准化参数。
5. 每一轮仅用当前已揭示标签的训练样本拟合特征 mean/std，再从头训练
   `512 → 128 → ReLU → Dropout(0.1) → 10` 的 MLP。
6. Adam；lr=0.001；weight decay=0.0001；batch=128；最多 40 epochs；patience=7。
7. 根据 validation macro-F1 选 checkpoint，平分时比较 validation loss。
   所有策略共享超参数和同一轮的模型初始化/训练随机种子。

全画幅 resize 有意不同于 torchvision 默认的 resize+center-crop，目的是避免裁掉
星系边缘；这一变更对所有方法一致，并会写入缓存指纹。
`outputs/preprocessing_preview_*.png` 显示训练图片处理前后；为了可视化，图中反转了归一化。

冻结特征让本地计算可行，也便于将差异归因于采样策略。其局限是 ImageNet 表示
未必充分捕捉天文形态。微调 CNN 可以作为后续稳健性实验，不在这版中混入。

官方模型文档：https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

## 四种选择方法

- **Random**：从剩余训练池均匀随机选择，不放回。
- **Uncertainty**：对预测的 10 类概率计算 Shannon entropy，选择熵最大的样本。
- **Diversity**：对冻结特征 L2 归一化；以已标注样本为中心，执行最远点优先的
  greedy k-center，逐次选择距离最近中心最远的候选。
- **Hybrid**：先按熵取前 `5 × batch_size` 个候选，再用上述 greedy k-center
  从候选中选出一个 batch。距离参照包含全部已标注样本及本批次已选样本。

候选倍数 5 是预设参数，不根据测试结果选择。策略函数的输入不包含真实标签。
`LabelOracle` 只允许查询训练池中从未查询的图片，拒绝读取未揭示标签。
真实标签只在选中后进入训练与事后覆盖率分析。

所有 seeds/策略的采样轨迹完成后才统一计算测试指标。
validation 曲线用于后续方法选择；test 曲线作为锁定协议后的统一评估。
不能看过 test 后反复调整参数并仍把它当成未见测试集。

## 输出怎么看

`outputs/latest_run.json` 给出最近一次运行的实际目录。
smoke 和 research 使用不同目录与缓存，不会混淆结果。

| 输出 | 含义 |
|---|---|
| `experiment_report.md` | 自动生成的研究结果说明、协议和局限 |
| `figures/test_learning_curves.png` | 相同标注预算下的 macro-F1、稀有类 recall/F1 |
| `figures/validation_learning_curves.png` | 验证集对应曲线 |
| `figures/rare_class_label_coverage.png` | 选中并揭示标签后，累计覆盖多少稀有类样本 |
| `metrics.csv` | 每个 seed/方法/轮次的指标 |
| `metrics_mean_std.csv` | 各 seed 的均值和样本标准差 |
| `aulc_by_seed.csv` | 每个 seed 的归一化 macro-F1 学习曲线面积 |
| `paired_differences_vs_random.csv` | 同一 seed 下与 random 的表现差值 |
| `validation_label_savings.csv` | 达到预设 validation F1 阈值时的首次观测标签预算与节省比例 |
| `per_class_metrics.csv` | 全部 10 类的 precision/recall/F1/support |
| `locked_protocol.json` / `environment.json` | 配置、代码/划分指纹、软件环境 |
| `seed_*/策略/queries.csv` | 每轮实际查询的原始图片编号与已揭示标签 |
| `seed_*/策略/round_*.json` / `.pt` | 每轮日志与 checkpoint，支持复核及续跑 |

归一化 AULC = 梯形面积 / (最大预算 − 初始预算)。
阈值节省率 = (random 首次达到阈值的预算 − 当前方法首次达到阈值的预算) / random 预算。
阈值固定为 validation macro-F1 的 0.5、0.6、0.7、0.8。
任一方法未达到阈值则结果留空并注明，不外推、不虚报节省。
曲线阴影是 seeds 之间 ±1 样本标准差，不是置信区间。
3 个 seeds 的结果用于探索性比较，不能据此直接宣称统计显著或正式非劣效。
全部标签 reference 是额外的大预算对照，并非保证达到的理论上限。

第 4 类测试集仅 50 张，recall 每错/对一张会变化 2 个百分点。
解释小幅提升时必须结合多个 seeds、precision/F1 及该样本量限制。

## 计算与保存

最慢的通常是首次对全部图片提取 CNN 特征。程序会输出图片/秒和动态 ETA，
不预设你的电脑运行时间。随后 120 次小型 MLP 训练及 3 次全标签对照共享缓存。
特征主体约 36 MB；模型与每轮日志另占空间。保留 cache 可避免重复提取。
更改图像尺寸、权重协议、相关代码或划分会建立新的缓存。

## 文件结构

| 文件 | 职责 |
|---|---|
| `run_all.ps1` | 安装、smoke、正式实验的一键入口 |
| `run_pipeline.py` | Python 总入口、参数检查 |
| `al_config.json` | 固定实验设置 |
| `galaxy_al/dataset.py` | 数据与索引一致性验证 |
| `galaxy_al/features.py` | 图片预处理预览、特征提取、缓存续跑 |
| `galaxy_al/model.py` | 冻结 encoder、MLP 训练和 checkpoint |
| `galaxy_al/acquisition.py` | 四种不读取真实标签的采样策略 |
| `galaxy_al/experiment.py` | 标签 oracle、循环、预算和测试评估隔离 |
| `galaxy_al/metrics.py` | 十类指标和 AULC |
| `galaxy_al/results.py` | 图表、配对比较和报告 |
| `test_active_learning.py` | 关键算法与标签隔离检查 |

## 当前验证范围

交付前在 Linux/Python 3.12 环境验证官方预训练权重的真实前向计算，
并用小型模拟 HDF5 数据跑通多 seed、全部策略、全标签 reference、输出图表及续跑。
算法单元检查覆盖 oracle 标签权限、熵排序、k-center、多样性退化情况、
hybrid 候选限制、随机可复现性、预算和指标。
用户的完整原始 HDF5 位于用户电脑，未在此处执行正式数据训练；
Windows 的实际环境由启动脚本中的 smoke test 进一步检查。
