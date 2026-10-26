#!/usr/bin/env python3
"""核对学习包数字与结构，不替代物理实验。仅使用 Python 标准库。"""

import hashlib
import json
import math
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "sources/evidence-official.json").read_text())
    meta = json.loads((root / "meta.json").read_text())
    study = json.loads((root / "sources/study-data.json").read_text())
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    required = ["README.md", "summary.md", "insights.md", "method.md", "mental-model.md", "qa.md", "reflection.md", "engineering.md", "evidence.md", "index.html", "paper.pdf", "meta.json", "deep-dive-bond-math.md", "dynamic-collar-implementation.md", "code/bond_hessian.py", "code/triangular_collar.py"]
    check("学习文件齐全", all((root / p).is_file() for p in required))
    check("标签恰好两个且互异", len(meta["tags"]) == len(set(meta["tags"])) == 2)
    check("校正后有十三位作者", len(meta["authors"]) == 13)
    check("版本水印为 v2", "arXiv:2609.28766v2" in (root / "sources/paper-layout.txt").read_text())
    check("PDF 摘要值一致", hashlib.sha256((root / "paper.pdf").read_bytes()).hexdigest() == meta["pdfSha256"])
    check("自测恰好十五问", len(re.findall(r"^### \d+\.", (root / "qa.md").read_text(), flags=re.M)) == 15)
    check("不宣称存在作者源码入口", meta["codeLinks"] == [] and meta["githubLinks"] == [])

    paired = data["closed_loop"]
    check("四任务阈值乘四配置", len(paired) == 16)
    for row in paired:
        n = row["n"]
        check(f"{row['task']}/{row['wrap_turns']}/{row['model']} 混淆矩阵与派生指标一致", (
            row["tp"] + row["fp"] + row["fn"] + row["tn"] == n
            and row["real_successes"] == row["tp"] + row["fn"]
            and row["sim_successes"] == row["tp"] + row["fp"]
            and math.isclose(row["agreement"], (row["tp"] + row["tn"]) / n, abs_tol=1e-12)
            and math.isclose(row["balanced_accuracy"], 0.5 * (row["tp"] / (row["tp"] + row["fn"]) + row["tn"] / (row["tn"] + row["fp"])), abs_tol=1e-12)
        ))
    for model in data["models"]:
        means = [x[model]["active_mean_px"] for x in data["replay"]["episodes"].values()]
        check(f"{model} 回放五段等权平均", math.isclose(sum(means) / len(means), data["replay"]["models"][model]["active_mean_px"], abs_tol=1e-12))
    check("均值而非每段都改善", data["replay"]["episodes"]["stick_tape_on_stand__017"]["Bond-R"]["active_mean_px"] > data["replay"]["episodes"]["stick_tape_on_stand__017"]["Coh-F"]["active_mean_px"])
    check("作者数据复制无改写", all(study[k] == data[k] for k in ["models", "costs", "closed_loop", "replay", "provenance"]))
    check("主 Peel 和黏附敏感性使用各自样本数", all(r["n"] == 100 for r in paired if r["task"] == "Peel") and study["adhesion"]["n"] == 98)
    pack = study["packaging"]
    check("封箱步数和时间一致", math.isclose(pack["steps"] * pack["dt_s"], pack["sim_time_s"], abs_tol=1e-12))
    check("封箱实时倍速保留原文精度", round(pack["sim_time_s"] / pack["wall_time_s"], 3) == pack["reported_real_time_factor"])
    check("不补造 Coh-F 固定容差精确成本", study["fixed_tolerance_32"]["Coh-F"] is None)

    class InspectHTML(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids, self.remote_assets = [], []

        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            if "id" in d:
                self.ids.append(d["id"])
            if tag in ("script", "img", "link", "iframe", "audio", "video", "source") and any(x in d for x in ("src", "href")):
                self.remote_assets.append(d)

    html = (root / "index.html").read_text()
    parser = InspectHTML()
    parser.feed(html)
    check("页面 ID 不重复", len(parser.ids) == len(set(parser.ids)))
    check("无外部资源依赖", parser.remote_assets == [])
    check("无 fetch/storage/API 请求", not re.search(r"\b(fetch|XMLHttpRequest|localStorage|sessionStorage|WebSocket)\b", html))
    script = re.search(r"<script>([\s\S]*?)</script>", html).group(1)
    check("页面内嵌完整真实数据", "__STUDY_DATA__" not in script and '307.77' in script and '130.43' in script)
    script_path = root / "sources/page-script.js"
    script_path.write_text(script)
    syntax = subprocess.run(["node", "--check", str(script_path)], capture_output=True, text=True)
    check("内嵌 JavaScript 语法有效", syntax.returncode == 0)

    demo = json.loads((root / "code/demo-validation.json").read_text())
    check("教学行为检查全部通过", len(demo["checks"]) == 27 and all(x["passed"] for x in demo["checks"]))
    math_report = json.loads((root / "code/bond-math-validation.json").read_text())
    mesh_report = json.loads((root / "code/collar-mesh-validation.json").read_text())
    check("二阶数学检查全部通过", math_report["passed"] == len(math_report["checks"]) == 18 and all(x["passed"] for x in math_report["checks"]))
    check("三角网格检查全部通过", mesh_report["passed"] == len(mesh_report["checks"]) == 80 and all(x["passed"] for x in mesh_report["checks"]))
    deep = json.loads((root / "sources/deep-dive-data.json").read_text())
    check("交互网格状态来自实际运行报告", deep["mesh"]["datasets"] == mesh_report["datasets"])
    check("交互法向阈值来自实际求根", deep["math"]["critical_normal_stretch"] == math_report["critical_normal_stretch"])
    embedded = re.search(r"const DEEP = (.*?);\nfunction bondScalar", script).group(1)
    check("深入页内嵌数据与报告副本一致", json.loads(embedded) == deep)
    check("三角网格示例尺寸与单步推进一致", all(len(dataset["rest_m"]) == 93 and len(dataset["triangles"]) == 120 and all(len(s["update"]["accepted"]) == 3 and len(s["update"]["thawed"]) == 4 for s in dataset["states"][1:]) for dataset in deep["mesh"]["datasets"].values()))
    index = json.loads((root.parents[1] / "index.json").read_text())
    entry = [x for x in index["papers"] if x.get("id") == root.name]
    check("索引中仅有一个新条目且标签一致", len(entry) == 1 and entry[0]["tags"] == meta["tags"])
    before = json.loads((root / "sources/index-before.json").read_text())
    check("既有索引条目完整保留", all(x in index["papers"] for x in before["papers"]))
    result = {"scope": "学习材料结构、原始统计与派生计算", "checks": checks, "passed": len(checks), "physical_benchmark_reproduced": False}
    (root / "sources/study-validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"通过 {len(checks)} 项资料、统计与页面静态核对。")


if __name__ == "__main__":
    main()
