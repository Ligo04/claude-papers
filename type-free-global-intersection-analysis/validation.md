# 验证记录与复现边界

## 本轮已完成

### 论文材料

- PDF 下载成功，文件为 17 页；使用 `pdfinfo` 和 `pdftotext -layout` 复核标题、作者、公式上下文、表 2–6 和局限段落。
- 自动解析器给出了错误的标题/作者 metadata，已以首页正文纠正，并在 `meta.json` 标记。
- 使用 `extract-images.py` 抽取 71 张图片，并复制了 pipeline、raycast、adaptive k-ring、benchmark 代表图。

### 源码静态检查

固定仓库 `d361b71e4ae97b2604c3b704fe31299e3b5dfb2f`，本轮对 README、`ResearchLog/prp_pipeline.md`、`ResearchLog/RayCasting_Untangling_Standalone.md`、PRP resolver、Python demos 和 CMake/CI 配置做了交叉阅读。

独立检查结果（在临时 clone 中）：

```text
python3 -m compileall -q PythonBindings/tests PythonBindings/python Solver/Scripts Example2D     PASS
cd PythonBindings/tests && python3 -m unittest -v test_demo_runner                                  7/7 PASS
python3 PythonBindings/tests/test_batch_method_comparison_enhanced.py   --backend cuda --begin 1 --end 59 --dry_run                                                     PASS
```

Jang dry-run 只验证样例选择、manifest/hash 和命令路径，不加载 `lcs_py`、不运行 GPU solver。当前仓库还没有 `build/bin`；`ext/luisaCompute` 和 `external/instant-mesh-intersection-repair` submodule 默认未初始化。CMake 在这种状态下按预期报告缺失 LuisaCompute。

### 教学 demo

```bash
cd /home/ligo/claude-papers/papers/type-free-global-intersection-analysis
python3 code/linear_displacement_demo.py --output /tmp/untangling_demo.png
python3 -m py_compile code/linear_displacement_demo.py
```

结果：选中 120°，`max depth=1.0208`，`J=7.7104`，生成约 115 KB PNG。它只是二维矩形/点 hit 的概念代理，不实现三角形 barycentric、真实 VF/EE/FV、BVH、cluster culling 或 Newton/PCG。

## 尚未完成

- 没有在本机完成 LuisaCompute 子模块初始化、CUDA C++/Python binding 编译或真实 GPU/CPU solver acceptance。
- 没有跑论文的大型 Synthetic、Round Table、Cloth Drop、Hair Card 或完整 59-case 数据集；论文表格数字是作者报告，不是本轮重测。
- 没有验证 CPU/GPU PRP 的逐 hit parity、CCR 模式对论文表格的影响，亦没有证明任何全局解缠性质。
- CTest 配置会编译 unit-test binary，但仓库 `UnitTest/CMakeLists.txt` 没有把这些 main 注册成 `add_test`；CI 的 `ctest` 不是所有 C++ 自检都已执行的证据。

## 最小真实复现路线（需要 CUDA/编译环境）

```bash
git clone https://github.com/ChengzhuUwU/Untangling26
cd Untangling26
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -U pip numpy trimesh triangle polyscope ninja
cmake -S . -B build -G Ninja   -D CMAKE_BUILD_TYPE=Release   -D LCS_BUILD_MAIN_APPLICATION=OFF   -D LCS_BUILD_PYBINDINGS=ON   -D LCS_PYTHON_EXECUTABLE="$PWD/.venv/bin/python"   -D LUISA_COMPUTE_ENABLE_CUDA=ON   -D LUISA_COMPUTE_ENABLE_METAL=OFF   -D LUISA_COMPUTE_ENABLE_DX=OFF   -D LUISA_COMPUTE_ENABLE_VULKAN=OFF
cmake --build build --target lcs_py -j 2
.venv/bin/python PythonBindings/tests/demo_unit.py   --backend cuda --headless --scene_id 0 --advance_frames 100   --use_gpu_untangling 1 --export_debug
```

运行后应记录：commit SHA、submodule SHAs、Python interpreter、backend、`use_gpu_untangling`、CCR mode/tau、direction optimization 次数、initial/final EF、frame count、wall time 和 debug JSON。否则“跑过 demo”不足以和论文表格比较。
