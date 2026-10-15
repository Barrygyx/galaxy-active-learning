# 交付前验证记录

- 6 项单元测试通过：十类指标、非法概率、smoke 拒绝、非训练/重复查询拒绝、checkpoint 篡改拒绝、线性模型导出与训练集标准化。
- 使用前一版 AL 的真实文件结构进行合成数据端到端验证：120 个 512 维特征，train 80 / validation 20 / test 20，2 seeds，4 AL 策略及 full-reference。
- 输出 22 行模型/分割指标、220 行类别指标、4 个 C 候选、9 对 PNG/PDF、每模型验证预测和中文报告；所有候选收敛。
- 所有已有 MLP 的验证 macro-F1 和 accuracy 与保存记录匹配。
- 对 131 个原实验、缓存及 preparation 文件核对运行前后的 SHA256 和修改时间，均未改变。
- 在临时复制的缓存中分别篡改 features 和 source_indices，均被拒绝。
- 已检查对照图和混淆矩阵的文字、图例与布局。
- 验证环境：Linux / Python 3.12 / PyTorch 2.14.1+cpu / scikit-learn 1.8.0 / numpy 2.3.5。未在 Windows 执行 PowerShell，包装脚本使用普通 PowerShell 参数及退出码检查。
- 用户真实数据和模型在其电脑，未在交付环境运行真实 17,615 图像实验。合成数据分数没有科研含义，没有包含在交付代码中。

代码是对原结果的新增诊断，不改变 RQ、原主动学习实验或原测试结论。
