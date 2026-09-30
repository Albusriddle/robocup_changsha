# 第三方依赖与许可声明

## 推理期（正式评测）

- Python 标准库，以及官方评测环境 `requirements-official.lock` 已提供的 `numpy`（2.5.3）。
- 官方仓库公开接口 `coverage_bench.protocol`（数据类型）与 `coverage_bench.spaces`（未使用展平）。
- 本提交的 `requirements-infer.lock` 为**零额外依赖**（仅注释行，不含任何包行）。
- 未引入任何第三方代码、预训练模型、数据集或外部下载；`artifacts/` 中只有本方案自己的冻结配置 `rule-config.json`。

## 训练期

- 本方案为**纯规则方法**，提交内**不包含训练代码**：模板自带的 PPO 训练脚本 `train.py` 已从本提交中删除，理由见
  `REPORT.md` 的"材料变更说明"。
- 因此 stable-baselines3 / SuperSuit / PyTorch 等训练侧依赖既不参与本提交，也不出现在推理路径中；
  正式推理只执行 `entry.py` 中的 numpy 前向规则逻辑。

## 许可证

- 本目录（`participant/P666/`）中由参赛者编写的代码按随附 `LICENSE`（MIT）授权。
- 官方材料（`coverage_bench/`、`configs/`、`docs/`、`scripts/`、`participant/_template/`）的版权与许可归组织方所有，
  不在本声明范围内；本提交未修改、未再分发这些材料。
- 未复制或改写任何第三方源码，因此没有需要额外署名的第三方许可项。
