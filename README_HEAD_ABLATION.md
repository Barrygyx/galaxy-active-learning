# 固定标注子集上的分类器与早停对照

本包新增运行入口，保留原实验。复用原来的冻结 ResNet18 特征，不提取新图像特征；不需要原始 HDF5、网络下载权重或 GPU。

## Windows 一次运行

把 ZIP 内的文件放到 `C:\Users\Barry\Desktop\galaxy-active-learning` 根目录，与 `run_pipeline.py`、`al_config.json` 同一层。
保留 `cache/features`、原 `data/processed/prepared_...`、原 `outputs/active_learning/research_...` 及 `.venv`。不用修改原 `galaxy_al`、配置或诊断文件。

```powershell
cd ([Environment]::GetFolderPath("Desktop"))
cd .\galaxy-active-learning
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_head_ablation.ps1 -Run "outputs\active_learning\research_6e91d238c2999303"
```

默认使用原实验最终预算（本项目 2,000 标签），3 seeds × 4 原选样策略。每组复用原 checkpoint，并新训练 4 个 MLP + 3 个固定 C 的线性分类器，共 48 个 MLP、36 个线性拟合；不重跑 10 轮主动学习。

### 较小预算的快速开发实验

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_head_ablation.ps1 -Run "outputs\active_learning\research_6e91d238c2999303" -Labels 1000
```

这会准确使用各策略原轨迹中第 1,000 标签预算对应的数据和旧 checkpoint，不临时抽样、不重划 split。只能指定原实验保存过的预算点，否则报错。1,000 与 2,000 标签实验自动输出到不同目录。

**减少训练样本主要用来降低计算量和研究预算效应，不是防过拟合的保证。** 样本更少反而可能让泛化更难。验证集保持完整（本项目 2,643 张，其中 class 4 有 50 张），不为加速删掉稀有类验证样本。

安装过依赖后可以直接运行 Python：

```powershell
.\.venv\Scripts\python.exe run_head_ablation.py --run "outputs\active_learning\research_6e91d238c2999303"
```

## 候选和控制变量

| Variant | 隐藏层 | Dropout | Adam weight decay | 说明 |
|---|---:|---:|---:|---|
| original_saved | 原设置 128 | 原设置 0.1 | 原设置 0.0001 | 复用旧模型/旧早停 |
| mlp_early | 128 | 0.1 | 0.0001 | 只换新早停规则 |
| mlp_small32 | 32 | 0.1 | 0.0001 | 在新规则下只缩小容量 |
| mlp_dropout03 | 128 | 0.3 | 0.0001 | 在新规则下只加强 dropout |
| mlp_wd1e3 | 128 | 0.1 | 0.001 | 在新规则下只加强 weight decay |
| linear_C0p001 | 无隐藏层 | — | — | Logistic Regression，C=0.001 |
| linear_C0p01 | 无隐藏层 | — | — | Logistic Regression，C=0.01 |
| linear_C0p1 | 无隐藏层 | — | — | Logistic Regression，C=0.1 |

MLP 的未改参数来自原锁定配置，包括 lr=0.001、batch_size=128；表中原设置对应你的实验。每个新 MLP 使用该策略/seed/预算原来的 training_seed。不同维度网络的参数初始化不可能逐项相同，但使用相同随机种子。

同一策略/seed 的所有候选使用完全相同的已查询标签、完整验证集和冻结特征。训练特征的 mean/std 只在本次已标注子集上拟合。无类别重加权或训练集重平衡。

线性模型这次也只使用当前预算的标签，不再使用全部 12,330 个训练标签。三个 C 是预先固定的独立候选，全部报告，不给不同策略单独挑最有利的 C。

## 更积极的早停

默认 max_epochs=40、patience=3（原等待 7 轮）。每轮记录 train eval loss、val loss、train/val macro-F1，以及验证 class 4/5 F1。

训练在以下任一条件满足时结束：

1. 验证 macro-F1 连续 3 轮没有相对进度基准累计超过 0.001 的提升；
2. 验证 cross-entropy 连续 3 轮没有相对进度基准超过 0.0001 的下降；
3. 达到最大 epoch。

始终恢复所有已观察 epoch 中验证 macro-F1 最好的 checkpoint；F1 相同则选验证 CE 更低者。进度阈值只控制等待多久，不丢弃幅度较小但确实更好的已观察 checkpoint。

CE 保护可能阻止后续 F1 的进一步提高；这是此次明确测试的取舍，不保证更好。`mlp_early` 与旧 checkpoint 的对照能单独检查改变早停带来的影响。线性模型使用 LBFGS 的数值收敛标准，不应用神经网络的 epoch 早停。

## 输出和续跑

输出路径：`outputs/head_ablation/fixed_<budget>_<protocol_hash>/`。
同一命令重新运行，会校验并复用已提交的模型；中断时未提交的单个 fit 会重训。不同参数、代码、依赖版本、源特征或预算自动形成新的实验身份，不混合结果。不要在同一个输出目录同时运行两个进程。

完成后终端打印报告位置，`outputs/latest_head_ablation.json` 也记录目录。失败/中断的输出没有完成标记；没有 `RUN_COMPLETE.json` 就不算完成。

建议发回：

- `ablation_report.md`
- `metrics_mean_std.csv`
- `paired_vs_original.csv`
- `figures/validation_model_comparison.png`
- `figures/class4_class5_comparison.png`
- `figures/training_curves_hybrid_seed_42.png`

还输出每 seed 的指标、per-class 指标、训练历史、类别标注数量、模型、验证/训练预测。PNG 同时保存 PDF。

只按所有策略和 seed 的平均验证 macro-F1 选择**一个共享候选**；平分时比较验证 CE。不收敛的线性候选不得获选。若旧模型最好，报告保留旧模型，不强行宣称改进。`shared_validation_selection.json` 给出全部排名。

## 实验边界

- 本次为固定标注数据的分类器消融，不重新执行主动学习选样。新 head 如有效，之后仍需参与每一轮选样并重跑完整轨迹，才能回答更新后的主动学习方法表现。
- 不计算测试指标；原测试集已经查看过，后续报告要标明开发/探索性质。
- 验证集已经用于早停、比较和选候选，最高验证分数有选择偏差。3 seeds 和小幅分差不能自动解释为显著提升。
- 判断是否改善，要看验证 macro-F1、class 4、class 5 和验证 CE。训练分数下降或 train–val 差距变小，本身不足以说明改善。
- 早停、dropout、weight decay 是待验证的设置，无法承诺消除过拟合。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest test_head_ablation.py -v
```

交付环境只有先前 120 行合成缓存和测试 checkpoint，没有你电脑上的真实 17,615 行缓存。本包会在合成旧实验上做端到端与续跑验证；这些测试分数不能作为 Galaxy10 结果。

官方 API 说明：

- https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout
- https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
