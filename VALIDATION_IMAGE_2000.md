# 验证记录

本包交付新的 2,000-label 运行流程；没有预先生成新的真实科研成绩。

## 已完成

- 原训练核心 10 项检查通过：预算嵌套与标签隔离、训练增强、选定卷积层梯度更新、warm-up、BN 固定、完整 epoch 恢复、训练统计独立、输入哈希、类别权重及显式十类指标。
- 新流程 6 项检查通过：默认协议、包含失败类别的 macro-F1、两 seed 均值及样本 SD、被修改预测拒绝、验证图像不一致拒绝、报告/CSV/图表输出。
- 在 120 行合成清洗格式数据上实际运行：每个 seed 40 张训练图，20 张 validation，64 像素，2 epochs，seed 42 与 43。使用真实官方 ImageNet 预训练权重，四种模型实际完成训练，两个 seed 都完成等权集成与汇总。
- 先完成 seed 42，再追加 42、43。日志有四次 completed-model 复用；42 的 16 个已完成模型文件，内容 SHA256 和修改时间均保持不变。
- 汇总中的逐类/整体指标均与十类混淆计数相符；每个模型的历史、预测及权重文件哈希核对通过；ensemble 成员的预测哈希与模型一致。
- 跨 seed 原始划分哈希和完整 validation IDs 相同；每个 seed 的 train 与 validation 不交叉，monitor 只来自 validation。
- 双 seed 流程输出 8 对 PNG/PDF；训练曲线经过视觉检查。
- 原 galaxy_image_demo.zip 内容 SHA256 未改变。代码放在新 image_budget_runner 文件夹中，无需覆盖原训练脚本和包。
- 源码语法、ZIP CRC 及打包清单在交付前核验。

## 限制

- 合成数据只用于检验工程行为，其分数没有科研含义，未作为模型成绩随包发布。
- 工程环境为 Linux / Python 3.12 / PyTorch 2.14.1+cpu / torchvision 0.29.1+cpu / h5py 3.16.0。Windows PowerShell 包装脚本只做静态检查；真实训练由已有 Windows Python 环境执行。
- 本次未在真实 Galaxy10 上跑新的 2,000-label / 30-epoch 实验。请运行后回传报告，不能从工程检查推断 F1 改善。
- 早停只能根据已有验证表现选择 checkpoint，不保证泛化、不保证消除过拟合。主报告明确记录停止原因与达到上限的情况。

## 可选本机检查

```powershell
.\.venv\Scripts\python.exe -m unittest test_image_2000.py -v
.\.venv\Scripts\python.exe -m unittest discover -s image_budget_runner -p test_image_demo.py -v
```

这些检查不运行完整真实数据训练。实际拟合的输出位置见 README_IMAGE_2000.md。
