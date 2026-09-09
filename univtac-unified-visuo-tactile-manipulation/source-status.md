# 公开实现状态快照（2026-09-02）

## 仓库

- URL：<https://github.com/univtac/UniVTAC>
- commit：`05bcd3edb92237107efa40105292a24f1a9fd761`
- 最新提交信息：`fix installation typo`，时间 2026-06-20（仓库快照检查于 2026-09-02）
- 许可证文件：Apache License 2.0

## 可以直接看到的实现

- `encoder/network.py`：ResNet-18 共享编码器；RGB（marked/unmarked）、depth、marker 和 pose 头；MSE 加权损失。
- `encoder/train.py`：`shape_pathway`、`contact_pathway`、`contact_shape` 和 full ablation 配置；full 配置使用 marked RGB、RGB、depth、marker、pose。
- `docs/Collection.md`：`collect_data.sh`、`parallel_collect.sh`、任务配置字段和 HDF5 数据结构。
- `eval_policy.sh` / `parallel_eval.sh`：统一策略评测入口。
- `data/download.sh`：通过 ModelScope 下载数据集。

## 需要特别核对的差异

1. README 宣称三类传感器，但 TODO 说明当前采集/评测只支持 `gsmini`。
2. README 许可证文字写 MIT，与根目录 Apache-2.0 LICENSE 不一致。
3. 项目页/README 的安装要求包括 Linux、Python 3.10、Isaac Sim 4.5、Isaac Lab 2.1.1、cuRobo 和项目修改过的本地 TacEx；公共 TacEx 不能直接替代。
4. Hugging Face viewer 本次检查出现 `DatasetGenerationError/CastError`，应以下载后的原始数据和 loader 验证为准。

## 不在本次工作范围内

没有启动 Isaac Sim、编译 TacEx/UIPC、下载 3D/触觉资产、训练真实 encoder、运行 8 任务策略或连接 Marvin 真机。因此本资料的代码 demo 只用于解释数据结构、三路径监督和自适应抓取逻辑，不能声称复现论文成功率。
