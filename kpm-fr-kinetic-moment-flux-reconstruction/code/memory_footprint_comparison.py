"""
内存占用对比 demo —— KPM-FR vs LBM 家族
========================================

复现论文的核心内存论点:
  KPM-FR = 42 字节/点
         = 标准图形 LBM 的 18%
         = 高保真 in-place streaming CFD LBM 的 34%
         = MR-LBM 的 53%
  相比标准 D3Q27 LBM 每点内存减少 5 倍以上。

本 demo:
  1. 从"每点存什么"推算各方法的每点字节数;
  2. 对比不同网格规模下的总内存, 说明为何 KPM-FR 能把
     十亿级仿真塞进消费级显卡显存(如 RTX 4090 的 24 GiB);
  3. 复现赛车 demo 的内存估算 (2000×332×800, 21 GiB)。

纯标准库:  python memory_footprint_comparison.py
"""


def bytes_per_point():
    """各方法每个 solution point / lattice node 的内存(字节)。
    数值以论文给出的相对比例 (KPM-FR=42B, 占标准图形LBM 18%) 反推。"""
    kpm_fr = 42                                  # 论文明确: 42 字节/点
    graphics_lbm = round(kpm_fr / 0.18)          # KPM-FR 是其 18%
    cfd_inplace_lbm = round(kpm_fr / 0.34)       # KPM-FR 是其 34%
    mr_lbm = round(kpm_fr / 0.53)                # KPM-FR 是其 53%
    # 供参考: 朴素 D3Q27 双缓冲 float32 = 27*2*4 = 216 B (与"标准图形LBM"量级一致)
    naive_d3q27_fp32 = 27 * 2 * 4
    return {
        "KPM-FR (本文)": kpm_fr,
        "MR-LBM [Li 2023]": mr_lbm,
        "in-place streaming LBM (CFD) [Geier 2025]": cfd_inplace_lbm,
        "标准图形 LBM [Lyu 2023]": graphics_lbm,
        "朴素 D3Q27 双缓冲 fp32 (参考)": naive_d3q27_fp32,
    }


def human(nbytes):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PiB"


def main():
    bpp = bytes_per_point()
    print("=" * 66)
    print("每点内存占用 (bytes/point)")
    print("=" * 66)
    kpm = bpp["KPM-FR (本文)"]
    for name, b in bpp.items():
        ratio = kpm / b * 100
        bar = "█" * int(b / 5)
        tag = f"  KPM-FR 是它的 {ratio:.0f}%" if name != "KPM-FR (本文)" else "  ← 基准"
        print(f"  {name:<42} {b:>4} B {bar}{tag}")

    print("\n" + "=" * 66)
    print("不同网格规模的总内存 (仅存状态, 不含临时)")
    print("=" * 66)
    grids = [
        ("256³ (1600 万点)", 256**3),
        ("512³ (1.34 亿点)", 512**3),
        ("1024³ (10.7 亿点)", 1024**3),
        ("18 亿点 (论文最大规模)", 1_800_000_000),
    ]
    methods = ["KPM-FR (本文)", "标准图形 LBM [Lyu 2023]"]
    print(f"  {'网格':<28} " + " ".join(f"{m.split(' ')[0]:>14}" for m in methods))
    for gname, npts in grids:
        cells = []
        for m in methods:
            cells.append(f"{human(npts*bpp[m]):>14}")
        print(f"  {gname:<28} " + " ".join(cells))

    print("\n" + "=" * 66)
    print("复现赛车 demo 的内存估算")
    print("=" * 66)
    nx, ny, nz = 2000, 332, 800
    npts = nx*ny*nz
    mem_kpm = npts * kpm
    print(f"  网格: {nx}×{ny}×{nz} = {npts:,} solution points ({npts/1e6:.0f} M)")
    print(f"  KPM-FR 状态内存 ≈ {human(mem_kpm)}  (论文报告 21 GiB, 含临时/边界)")
    print(f"  若用标准图形 LBM: ≈ {human(npts*bpp['标准图形 LBM [Lyu 2023]'])}"
          f"  → 远超 RTX 4090 的 24 GiB 显存!")
    print("\n  结论: '不存分布函数 + 矩表示'把高分辨率气动仿真压进桌面级显存预算,")
    print("        这是 KPM-FR 三大优势(低耗散/高吞吐/小内存)中'小内存'的直接来源。")


if __name__ == "__main__":
    main()
