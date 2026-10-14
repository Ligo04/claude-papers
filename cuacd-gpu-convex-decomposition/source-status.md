# 验证状态与复现边界

日期：2026-10-14。

| 项目 | 状态 / 证据 |
|---|---|
| 原文 | arXiv v1；15 页；PDF 完整解析；SHA256 见 sources.md |
| 元数据 | 首页与 arXiv 交叉核对；修正 parser 作者与摘要错误 |
| 核心学习材料 | README、summary、insights、method、mental-model、reflection、qa、source-study、sources 均存在 |
| 问答 | 基础/中级/高级各 5，总数 15 |
| 原图 | 抽取 140 张嵌入图像；额外渲染关键矢量图与表格；抽取图片数不等于 figure 数 |
| 数据 | metrics.json 收录 Tables 1、2、4、5；HTML 内嵌同一数据 |
| 索引 | 从原索引读取并追加；旧 55 条保持不变，新总数 56；meta/index 均恰好两个相同标签 |
| 教学 demo | 标准库运行、py_compile 通过；12 组排序/压缩和 7 组凸包语义检查通过 |
| demo 输出 | 固定 seed=7，合成二维点 2048 → 69 幸存点，最终 23 个 hull 顶点；预筛前后 hull 相同 |
| HTML | CSS/JS/图片/笔记均内嵌；无外部资源依赖；无 fetch/localStorage/sessionStorage；inline JS 语法通过 |
| 浏览器行为 | 六章节切换、数据集/指标、搜索消融、凸包分布/点数、ballot、笔记选择、问答筛选与展开均检查 |
| HTTP | localhost:8765 页面 200，响应与本地 HTML 字节一致 |
| iframe | sandbox=allow-scripts，无 allow-same-origin；交互和内嵌图像可用 |
| 视觉 | 1440×1080 桌面、430×932 窄屏，最终 HTML 重新截图；见 output/playwright/ |
| 页面控制台 | 主页面 0 errors / 0 warnings；独立 iframe 验证容器的初始 favicon 404 已补 data URI |
| 源码 | 固定 revision 13a00e25477f53a3533938b0325e2d575e3fd68b，只读静态检查；摘录与校验和已保存 |

## 发现与限度

论文所述上传后无中间 host 同步，与所查源码的 C 外层循环、计数读回和 cuStreamSynchronize 不一致。Python 入口还有 SciPy 初始凸包。API/CLI 默认候选与 merge 开关不同；论文/README 显存池默认 70%，C 实现为 80%。这些是特定 revision 的静态观察。

没有安装或运行官方 CuACD CUDA、未复现论文性能、未下载评测资产集、未接入 ManaEnv/IsaacLab/Newton。所有性能数来自作者论文；demo 的点数压缩结果仅属于合成二维教学输入，不可当作 GPU 加速结果。

页面截图来自本地浏览器。完整离线 HTML 为 index.html；运行 HTTP 服务只是便于浏览，不是其依赖。
