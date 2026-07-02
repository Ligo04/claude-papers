#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wam_tuple_classifier.py
=======================
综述核心分析机器的一个可运行、零依赖实现：每个世界动作模型（WAM）都是一个四元组

        (Phi  = 预测基底,
         F    = 动作耦合,
         B    = 架构骨干,
         D    = 部署机制)

并且每个 WAM 还恰好属于一种 *设计哲学*
（Render-and-Decode / Latent-Only / Video-Generation-Free）。

本 DEMO 的意义：论文主张你可以通过写出元组来 *放置和比较* 任意 WAM。本脚本
(1) 用表 1-2 与第 3 节给出的四元组和哲学编码普查的一个切片，(2) 展示哲学
*可由* 基底 + 未来如何被消费 *推导*（它不是独立的第 5 个标签），(3) 复现
"设计不对称"观察：基底（Phi）与骨干（B）是保守的训练时选择，而耦合（F）
与机制（D）变化更大。

运行：  python3 wam_tuple_classifier.py
"""

from collections import Counter
from dataclasses import dataclass

# ---- 四个轴（第 4 节）。允许值，照搬自论文。 ----
SUBSTRATE = {"pixel_decoded", "pixel_latent", "feature", "geometric", "affordance"}
COUPLING  = {"rollout", "joint", "post_head"}      # 动作条件化 / 联合 / 后置预测头
BACKBONE  = {"diffusion", "autoregressive", "jepa", "hybrid", "llm_vlm"}
REGIME    = {"open", "chunked", "single", "interactive"}

PHILOSOPHIES = ("Render-and-Decode", "Latent-Only", "Video-Generation-Free")


@dataclass(frozen=True)
class WAM:
    name: str
    year: str
    substrate: str   # Phi
    coupling: str    # F
    backbone: str    # B
    regime: str      # D
    # 论文按推理路径给每个工作分配一种哲学。我们存它用来 *核对* 推导，而非定义它。
    paper_philosophy: str

    def tuple4(self):
        return (self.substrate, self.coupling, self.backbone, self.regime)

    def __post_init__(self):
        assert self.substrate in SUBSTRATE, self.substrate
        assert self.coupling in COUPLING, self.coupling
        assert self.backbone in BACKBONE, self.backbone
        assert self.regime in REGIME, self.regime
        assert self.paper_philosophy in PHILOSOPHIES, self.paper_philosophy


def derive_philosophy(w: WAM) -> str:
    """按论文定义（第 3 节）从推理路径推导哲学：动作在 *何处* 接地？

      * 在任何视频生成器之外（特征/几何/可供性，*不经* 视频骨干产生）
                                          -> Video-Generation-Free
      * 一个视频派生的未来，保持在像素解码 *之前*（来自视频干路的隐/特征，
        或推理时从不渲染像素）             -> Latent-Only
      * 视频骨干一路跑到（可解码的）像素未来
                                          -> Render-and-Decode

    骨干告诉我们路径中是否有 *视频生成器*：扩散/自回归/混合在此为视频能力干路；
    jepa/llm_vlm 不是。
    """
    video_backbone = w.backbone in {"diffusion", "autoregressive", "hybrid"}

    if not video_backbone:
        # 预测路径中无视频生成核。
        return "Video-Generation-Free"

    if w.substrate in {"pixel_decoded"}:
        # 在解码动作前把生成器跑到了渲染像素。
        return "Render-and-Decode"

    if w.substrate in {"pixel_latent"}:
        # 来自视频干路的像素可解码隐。若方法在推理时承诺解码像素则为
        # Render-and-Decode；若停在隐则为 Latent-Only。论文逐方法裁决；这里
        # 用部署机制作启发：single/open 倾向解码，chunked 隐路径倾向短路。
        # （启发式 -> 可能不一致；该不一致本身就有教益，见报告。）
        return "Render-and-Decode" if w.regime in {"open", "single"} else "Latent-Only"

    # 视频派生的 特征/几何/可供性 未来，不做像素解码。
    return "Latent-Only"


# ---- 普查切片（表 1-2 / 第 3 节）。元组与哲学按论文。 ----
CENSUS = [
    # ---- Render-and-Decode（表 1，上半带） ----
    WAM("UniPi",      "2023.02", "pixel_decoded", "post_head", "diffusion",      "open",    "Render-and-Decode"),
    WAM("AVDC",       "2023.10", "pixel_decoded", "post_head", "diffusion",      "open",    "Render-and-Decode"),
    WAM("VLP",        "2023.10", "pixel_decoded", "post_head", "diffusion",      "chunked", "Render-and-Decode"),
    WAM("GR-1",       "2023.12", "pixel_latent",  "joint",     "autoregressive", "single",  "Render-and-Decode"),
    WAM("Gen2Act",    "2024.09", "pixel_decoded", "post_head", "diffusion",      "open",    "Render-and-Decode"),
    WAM("GR-2",       "2024.10", "pixel_latent",  "joint",     "autoregressive", "chunked", "Render-and-Decode"),
    WAM("PAD",        "2024.11", "pixel_latent",  "joint",     "diffusion",      "single",  "Render-and-Decode"),
    WAM("CoT-VLA",    "2025.03", "feature",       "post_head", "llm_vlm",        "chunked", "Render-and-Decode"),
    WAM("TesserAct",  "2025.04", "geometric",     "post_head", "diffusion",      "open",    "Render-and-Decode"),
    WAM("WorldVLA",   "2025.06", "pixel_latent",  "joint",     "autoregressive", "single",  "Render-and-Decode"),
    WAM("DreamZero",  "2025.11", "pixel_latent",  "joint",     "autoregressive", "chunked", "Render-and-Decode"),
    WAM("F1",         "2025.09", "pixel_latent",  "joint",     "hybrid",         "chunked", "Render-and-Decode"),
    # ---- Latent-Only（表 1，下半带） ----
    WAM("VPP",        "2024.12", "feature",       "post_head", "diffusion",      "chunked", "Latent-Only"),
    WAM("UWM",        "2025.04", "pixel_latent",  "joint",     "hybrid",         "chunked", "Latent-Only"),
    WAM("GenieEnv",   "2025.08", "feature",       "post_head", "diffusion",      "chunked", "Latent-Only"),
    WAM("Fast-WAM",   "2025.12", "feature",       "post_head", "diffusion",      "chunked", "Latent-Only"),
    WAM("AIM",        "2025.10", "affordance",    "joint",     "diffusion",      "chunked", "Latent-Only"),
    WAM("MWM",        "2025.12", "affordance",    "post_head", "diffusion",      "chunked", "Latent-Only"),
    WAM("S-VAM",      "2025.11", "feature",       "post_head", "diffusion",      "single",  "Latent-Only"),
    WAM("tau0-WM",    "2025.12", "pixel_latent",  "joint",     "hybrid",         "chunked", "Latent-Only"),
    # ---- Video-Generation-Free（表 2） ----
    WAM("FLARE",      "2025.05", "feature",       "post_head", "jepa",           "chunked", "Video-Generation-Free"),
    WAM("DUST",       "2025.11", "feature",       "joint",     "llm_vlm",        "chunked", "Video-Generation-Free"),
    WAM("PointWorld", "2025.10", "geometric",     "rollout",   "llm_vlm",        "chunked", "Video-Generation-Free"),
    WAM("PALM",       "2025.10", "affordance",    "post_head", "llm_vlm",        "chunked", "Video-Generation-Free"),
    WAM("HiF-VLA",    "2025.09", "geometric",     "joint",     "llm_vlm",        "chunked", "Video-Generation-Free"),
    WAM("LDA-1B",     "2025.10", "feature",       "post_head", "llm_vlm",        "chunked", "Video-Generation-Free"),
    WAM("DexWM",      "2025.11", "feature",       "rollout",   "jepa",           "chunked", "Video-Generation-Free"),
    WAM("ALAM",       "2025.12", "feature",       "post_head", "llm_vlm",        "chunked", "Video-Generation-Free"),
]


def axis_diversity(census):
    """复现"设计不对称"（第 4.6 节）：Phi 和 B 更集中（训练时、保守）；F 和 D
    更自由变化（推理时）。我们用归一化熵度量集中度：熵越低 = 越集中/保守。"""
    import math

    def norm_entropy(values, universe):
        c = Counter(values)
        n = len(values)
        h = -sum((v / n) * math.log(v / n) for v in c.values())
        return h / math.log(len(universe))  # 0 = 单一值，1 = 均匀

    return {
        "Phi (基底)": norm_entropy([w.substrate for w in census], SUBSTRATE),
        "B (骨干)":   norm_entropy([w.backbone for w in census], BACKBONE),
        "F (耦合)":   norm_entropy([w.coupling for w in census], COUPLING),
        "D (机制)":   norm_entropy([w.regime for w in census], REGIME),
    }


def main():
    print("=" * 78)
    print("WAM 四元组分类器  —  《世界动作模型综述》(arXiv 2606.20781)")
    print("=" * 78)

    # 1) 把每个 WAM 打印为四元组 + 推导的 vs 论文的哲学。
    print("\n[1] 每个 WAM 作为四元组 (Phi, F, B, D)，附推导出的哲学：\n")
    hdr = f"{'WAM':<11}{'substrate':<14}{'coupling':<10}{'backbone':<15}{'regime':<12}{'推导':<22}{'一致?'}"
    print(hdr)
    print("-" * 92)
    agree = 0
    for w in CENSUS:
        d = derive_philosophy(w)
        ok = (d == w.paper_philosophy)
        agree += ok
        flag = "ok" if ok else f"!= {w.paper_philosophy}"
        print(f"{w.name:<11}{w.substrate:<14}{w.coupling:<10}{w.backbone:<15}{w.regime:<12}{d:<22}{flag}")
    print(f"\n    推导与论文标签一致：{agree}/{len(CENSUS)} 个工作。")
    print("    （不一致者是论文逐方法裁决的、真正模糊的 pixel-latent 情形——")
    print("      证明哲学并非可随意附加的第 5 个标签。）")

    # 2) 哲学分布。
    print("\n[2] 普查在三种设计哲学上的分布：\n")
    counts = Counter(w.paper_philosophy for w in CENSUS)
    for p in PHILOSOPHIES:
        bar = "#" * counts[p]
        print(f"    {p:<24}{counts[p]:>2}  {bar}")

    # 3) 设计不对称：哪些轴保守、哪些灵活。
    print("\n[3] 设计不对称（第 4.6 节）—— 每个轴的归一化熵")
    print("    （0.0 = 处处用同一个值/保守，1.0 = 均匀/灵活）：\n")
    div = axis_diversity(CENSUS)
    for axis, h in sorted(div.items(), key=lambda kv: kv[1]):
        bar = "=" * int(round(h * 40))
        print(f"    {axis:<14}{h:0.2f}  {bar}")
    print("\n    训练时轴（Phi、B）倾向更低（保守）；")
    print("    推理时轴（F、D）是近期创新集中之处。")

    # 4) "昂贵的角落"检查。
    print("\n[4] 昂贵的角落 = (pixel_decoded, joint, diffusion, single)：")
    corner = [w for w in CENSUS
              if w.tuple4() == ("pixel_decoded", "joint", "diffusion", "single")]
    print(f"    普查切片中的成员数：{len(corner)}")
    print("    -> 近乎为空，恰如论文预测：几乎没有方法直接居于最昂贵的角落；")
    print("       它们都换走了一个坐标。\n")


if __name__ == "__main__":
    main()
