# Galaxy10：第一步模型诊断

这是一套新增代码，复用已完成的主动学习实验。不会重新提取 ResNet 特征、重新跑主动学习、重新划分数据或修改已有 checkpoint。

## 放置与运行（Windows PowerShell）

解压后，把本包中的文件放到 `C:\Users\Barry\Desktop\galaxy-active-learning` 根目录。
确认 `diagnose_models.py`、`run_diagnostics.ps1`、`requirements_diagnostics.txt` 和原来的 `run_pipeline.py` 在同一层。不要替换 `galaxy_al`、`al_config.json`、`data` 或 `cache`。

```powershell
cd ([Environment]::GetFolderPath("Desktop"))
cd .\galaxy-active-learning
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_diagnostics.ps1
```

只安装新增的 scikit-learn，复用现有 `.venv` 的 torch/numpy/matplotlib/threadpoolctl。
首次安装需要网络；诊断本身无需网络、无需 GPU，无需原始 HDF5 文件。

自动读取 `outputs/latest_run.json`。若它指向 smoke，则只有在恰好存在一个完成的 research 实验时自动选取。多个 research 实验时明确指定：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_diagnostics.ps1 -Run "outputs\active_learning\research_6e91d238c2999303"
```

已安装依赖后直接运行：

```powershell
.\.venv\Scripts\python.exe diagnose_models.py
```

## 必須保留的原文件

- 原实验目录的 `locked_protocol.json`、`RUN_COMPLETE.json`、各 seed 的 full_reference 与四个 strategy 子目录。
- `full_reference/reference.json`、`reference.pt`；四个 strategy 的 `round_*.json`、最终 `round_*.pt`、`trajectory_complete.json`。
- 原 `data/processed/prepared_...` 中的 cleaning_report、split CSV 和索引等审计文件。
- `cache/features/<原key>/` 中 features.npy、source_indices.npy、done.npy、feature_manifest.json。
- `.venv` 原环境。若 features 已被删掉，脚本会停止，不会静默重新下载权重或提取特征。

## 做什么

1. 验证原 split、缓存特征和 checkpoint 的校验和与行顺序。
2. 使用原 checkpoint 重新计算：全标签 MLP 和四种主动学习方法最终预算下的 **已标注训练集**、**共同验证集** 表现。
3. 绘制原 full-reference 的每 epoch train loss / validation loss / validation macro-F1 曲线，标出选中的 epoch。历史中没有逐 epoch train macro-F1，脚本不会编造。
4. 用同一份缓存特征、全部 train_pool 标签训练一个多类线性 Logistic Regression。StandardScaler 只 fit 训练集。固定 C=[0.01,0.1,1,10]，只依据 validation macro-F1 选 C；平分时选更低 validation CE，再选更小 C。默认无类别权重。
5. 输出 per-class F1、混淆矩阵、验证错误索引，以及中文解释报告。

线性模型与 **full-reference MLP** 都使用全部训练标签；与 2,000-label AL 的直接差异包含标签数量因素。线性模型只训练一次，不把同一个确定性结果复制成三个 seed。这里对照的是两套训练程序，不是只控制网络结构的严格消融。

LBFGS 每个 C 最多 2,000 次迭代；不收敛时重试 4,000 次，并记录警告。不收敛候选不参与选型，全部不收敛则停止。耗时取决于 CPU 和收敛情况，代码会打印进展。

## 输出

新增 `outputs/model_diagnostics/research_<key>_<UTC时间>/`。每次运行创建新目录，旧结果保留。成功后有 RUN_COMPLETE.json 和 outputs/latest_diagnostic.json。

先看这四个文件：

- `diagnostic_report.md`：中文报告（可用 VS Code 预览）。
- `figures/train_validation_comparison.png`：相同 checkpoint 的训练/验证差距。
- `figures/full_reference_training_curves.png`：原训练是否停得过早、是否有验证 loss 反弹。
- `figures/validation_per_class_f1.png`：class 0、4、5 等的变化。

此外有 6 张混淆矩阵 PNG/PDF；`diagnostic_metrics.csv`、`diagnostic_per_class.csv`、`training_history.csv`、`train_validation_gaps.csv`；每个 seed 验证预测 NPZ；`validation_errors.csv`；`linear_candidates.csv`、`linear_selection.json`、`linear_model.npz` 和记录环境/协议的 JSON。

`linear_model.npz` 保存 coef、intercept、训练 mean/scale 和 classes。复用时：z=(features−mean)/scale，logits=z@coefficients.T+intercept，再 softmax。不要在验证集上重新 fit 标准化。

## 怎样解释

- 看 full-reference 的 train 与 val 差距、历史曲线，再看线性分类器是否接近或超过它；这些证据帮助决定先改训练还是研究特征。
- 不同 AL 方法选到的训练样本不同，train 分数不能直接用于策略排名。
- 混淆矩阵 **行是真实类，列是预测类**，图中数字是行百分比。跨 seed 汇总的是重复评估，不代表更多独立验证样本。
- 没有训练新 CNN，所以本诊断不能单独证明特征提取器是性能瓶颈。
- 不评估测试集；之前测试集已经查看过，后续开发仍应依赖验证集，不能把同一测试集重新称为未见过的确认集。

## 验证范围

附带 `test_diagnostics.py` 可运行：

```powershell
.\.venv\Scripts\python.exe -m unittest test_diagnostics.py -v
```

交付前会对旧项目格式的合成数据端到端验证、检查输入文件不被修改，并测试校验和/非训练查询的拒绝行为。真实 17,615 行特征与 checkpoint 在你的电脑上，交付环境没有它们，因此无法预先声称已在真实文件上跑完。Windows PowerShell 包装脚本会检查安装及 Python 的退出码。

官方方法说明：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
