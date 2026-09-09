# 图像说明

`page_<页码>_img_<序号>.png` 是技能脚本从 PDF XObject 直接提取的原图，共 40 个。以下别名保留同一像素内容，便于学习材料引用：

| 描述性文件 | 来源 | 内容 |
|---|---|---|
| `max-overlap-correction.png` | `page_6_img_1.png` | 两条边附近的重复障碍区域与减项示意（论文 Figure 4.1） |
| `sliding-block-setup.png` | `page_21_img_1.png` | 2D 网格平面滑动方块设置 |
| `stacked-cubes-initial.png` | `page_23_img_1.png` | 三立方体细化实验初始状态 |
| `stacked-cubes-equilibrium.png` | `page_23_img_2.png` | 三立方体稳定平衡状态 |
| `sphere-collision-stress.png` | `page_24_img_1.png` | 高速双球碰撞时 von Mises 应力 |
| `sphere-separation-stress.png` | `page_24_img_2.png` | 双球分离后的应力波 |
| `gyroid-microstructure.png` | `page_26_img_1.png` | gyroid 微结构网格/几何图 |

Figure 4.2 的 12 个障碍场子图位于 `page_7_img_1.png` 至 `page_7_img_12.png`；它们分别展示不同折角下直接求和、精确 `max` 与论文修正的场分布。文件提取顺序不等同于论文阅读顺序，使用时应对照 PDF 第 7 页的整版排布。

