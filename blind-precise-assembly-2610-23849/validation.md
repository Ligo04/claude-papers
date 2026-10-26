# 验证记录

日期：2026-10-20。以下检查针对学习材料和教学示例，不是原论文复现。

| 检查 | 结果 | 范围 |
|---|---|---|
| PDF 与元数据 | 通过 | 21 页；标题、4 位作者、v1 和日期对照 arXiv / PDF；保存 SHA-256 |
| 原图与表格 | 通过 | 提取 110 个嵌入图像，另渲染 4 个关键页面；目视核对图 12 和表 7 |
| 论文数据 | 通过 | 3 个任务各 7 条成功率；真机计数 173/237；保留 `>99.9` |
| 问答与标签 | 通过 | 15 道题、15 个折叠答案；meta 与根索引均为 2 个不同标签 |
| 几何演示 | 通过 | Python 标准库运行；检查顶点满足半平面约束、角点误差公式及随边数加密递减 |
| 信息流演示 | 通过 | CPU，PyTorch 2.14.0+cu130；辅助损失对编码器和 decoder 有梯度，对 actor 动作头无梯度 |
| 删除 decoder | 通过 | 相同输入下 actor 前向输出逐值不变，输出形状 4×7 |
| HTML 脚本语法 | 通过 | `node --check` |
| HTML 独立性 | 静态检查通过 | CSS、JS 和数据均内联；无外部资源、fetch 或 localStorage |
| 页面初始呈现 | 通过 | Codex 内置浏览器打开，截图见下方 |
| 训练 / 部署切换 | 通过 | 部署时监督分支消失，解释文字更新 |
| 泛化任务 / 条件选择 | 通过 | 切换插销、齿轮、螺母；齿轮保留不等号；螺母强深度变化显示 −0.6 pp |
| 真机筛选 | 通过 | 全部为 173/237；C 为 19/29、65.5%；Wilson 区间相应更新 |
| 几何双滑块 | 通过 | n=281、c=20 μm 时显示额外游隙约 0.0012 μm，SVG 的可访问描述同步变化 |

运行命令：

```bash
python3 ~/claude-papers/papers/blind-precise-assembly-2610-23849/code/clearance_geometry_demo.py
python3 ~/claude-papers/papers/blind-precise-assembly-2610-23849/code/proprioceptive_information_flow.py
```

几何示例默认 c=100 μm、n=7，公式额外游隙为 10.991626 μm。该数值不是碰撞基准的实测沉降误差。

输出：

- [可行平移区域 SVG](outputs/clearance-feasible-region.svg)
- [公式计算 CSV](outputs/clearance-derived.csv)
- [信息流检查 JSON](outputs/information-flow-check.json)
- [页面截图](outputs/explorer-preview.png)
- [材料结构与索引检查](outputs/material-check.json)

没有执行原作者 PPO 训练、原作者 MuJoCo / Warp benchmark、GPU 动力学、机器人部署或实物标定。没有独立验证移动端布局或严格 `sandbox="allow-scripts"` iframe 嵌入；页面无外部依赖的结构满足该用途，浏览器交互检查是在本地顶层页面完成。

本地阅读入口为 [index.html](index.html)。预览服务只监听 `127.0.0.1:8767`；即使预览服务停止，也可直接打开 HTML 文件。

## 后续 mjfit 几何专题验证

- 固定源码 SHA 为 `eee986cc087d14458337fdc195d5cf03abc6aa6d`，`mjfit 0.1.0`。
- 在学习目录内单独 venv 运行，MuJoCo 3.16.0。该版本与论文报告环境不能自动视为相同。
- 调用作者生成器生成星形销孔，20 个凸 mesh 分属两个 body，编译检查通过。
- 边法向间隙为 0.5 mm；板壁与插销的分块面积分别匹配目标区域。
- 编译后凸包体积与棱柱体积的最大相对差约 `8.41e−8`。
- 原库 15 项聚焦几何测试通过，0.79 秒，日志见 [mjfit-focused-tests.log](outputs/mjfit-focused-tests.log)。
- 基于实际凸片坐标的交互示意已在 sandbox iframe 内检查二维、棱柱与拆开三种模式。
- 没有推进 MuJoCo 动力学，没有运行 GPU benchmark、全量测试或真机实验。
