# RoboDojo 论文到源码的实现映射

> 这份映射来自论文发布后的官方仓库快照，只用于理解“论文概念在代码中落在哪里”，不把后续实现状态当成 v3 当日的复现实验。RoboDojo 快照：`57b8c52424ff1198ade242505656511515c58792`（2026-08-19）；XPolicyLab 快照：`85ff90d4092a9530860c9efe96a732d9f9150954`（2026-08-18）。

## 1. 从命令到任务实例

```text
scripts/robodojo.sh
  └─ scripts/internal/run_policy_eval.sh
      ├─ task/RoboDojo/task_registry.py       动态解析任务类和 YAML
      ├─ task/RoboDojo/config/_task.yml       公共默认值和任务级覆盖
      ├─ task/RoboDojo/config/<task>.yml      资产/场景规格
      └─ task/RoboDojo/tasks/<task>.py        成功条件与任务逻辑
```

- [`scripts/robodojo.sh`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/scripts/robodojo.sh) 是公开入口。
- [`task_registry.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/task/RoboDojo/task_registry.py) 依据任务名加载同名 Python 类和 YAML。
- [`_task.yml`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/task/RoboDojo/config/_task.yml) 保存 `data_source`、场景/机器人/相机配置、`eval_nums` 等公共口径，并为 random 变体或特殊任务覆盖。
- 具体任务把“配置”和“判定逻辑”分开。例如 [`stack_bowls.yml`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/task/RoboDojo/config/stack_bowls.yml) 选择碗资产与布局，而 [`stack_bowls.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/task/RoboDojo/tasks/stack_bowls.py) 实现任务行为/成功条件。

## 2. MagicSim/Isaac Lab 底座和 manager 分层

```text
TaskEnv
  ├─ RobotManager        机器人、动作控制、规划器
  ├─ SceneManager        场景对象和每环境布局
  │   └─ LayoutManager   刚体/铰接/布料/流体/背景/灯光
  ├─ CameraManager       每环境相机随机化与观测
  └─ TiledCaptureManager 批量图像采集
```

- [`TaskEnv`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/env/environment/task_env.py#L10-L100) 组合 Robot/Scene/Camera/Capture 四类 manager，并在 `reset()` 中按顺序重载场景、复位机器人、做 300 个无渲染稳定步、再复位相机与采集器。
- [`SceneManager`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/env/scene_manager/scene_manager.py) 和 [`LayoutManager`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/env/scene_manager/layout_manager.py) 对应论文的配置驱动场景和资产抽象。
- 资产类型不是文档层的口号：[`env/scene_manager/objects/`](https://github.com/RoboDojo-Benchmark/RoboDojo/tree/57b8c52424ff1198ade242505656511515c58792/env/scene_manager/objects) 中存在 `rigid.py`、`articulation.py`、`garment.py`、`fluid.py`、`background.py`、`light.py` 等具体实现。

## 3. 异构并行的代码抓手

论文的核心定义是：并行环境共享向量化 step 接口，但每个环境保留独立的场景配置、任务状态、seed、观测和成功条件。

- [`BaseEnv.launch_sim()`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/env/environment/base_env.py#L114-L145) 构造 `InteractiveSceneCfg` 时设置 `replicate_physics=False`。
- [`RobotManager._get_SceneCfg()`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/env/robot_manager/robot_manager.py#L527-L535) 的默认值同样是 `replicate_physics=False`。
- [`IsaacRLEnv._setup_scene()`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/env/environment/isaac/isaac_rl_env.py#L38-L41) 仍调用 Isaac Lab 的 `clone_environments(copy_from_source=True)`；关键不是“不 clone”，而是关闭严格 physics replication 后，由 manager 为环境装配/重载各自场景状态。
- [`TaskEnv.update_seed()`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/env/environment/task_env.py#L74-L81) 把每环境 seed 传播到 Scene/Robot/Camera manager。
- [`BaseEnv` 的 seed 管理](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/env/environment/base_env.py#L249-L305) 明确区分 global seed 与长度必须等于 `num_envs` 的 per-env seed 列表。

因此，旧学习笔记里“需要特殊 batched kernel/自定义资产内存布局”只能算推测，官方代码证据更具体：Isaac Lab 向量化接口 + `replicate_physics=False` + manager 驱动的每环境场景/seed 管理。

## 4. 观测、动作与 XPolicyLab 接口

论文表 12 的单环境/批量接口直接落在以下文件：

- [`model_template.py`](https://github.com/XPolicyLab/XPolicyLab/blob/85ff90d4092a9530860c9efe96a732d9f9150954/model_template.py#L1-L104)：`update_obs`、`update_obs_batch`、`get_action`、`get_action_batch`、`reset`，另有论文后新增的 `prepare_case` 和 `on_trial_end` 生命周期 hook。
- [`policy/demo_policy/model.py`](https://github.com/XPolicyLab/XPolicyLab/blob/85ff90d4092a9530860c9efe96a732d9f9150954/policy/demo_policy/model.py)：最小可运行 adapter，展示 joint/EE 两种动作键和 action chunk 返回格式。
- [`policy/demo_policy/deploy.py`](https://github.com/XPolicyLab/XPolicyLab/blob/85ff90d4092a9530860c9efe96a732d9f9150954/policy/demo_policy/deploy.py#L1-L42)：单环境与批量评测循环。批量循环持续维护 active env index，只给尚未结束的环境发送后续 action chunk。
- [`policy/demo_policy/deploy.yml`](https://github.com/XPolicyLab/XPolicyLab/blob/85ff90d4092a9530860c9efe96a732d9f9150954/policy/demo_policy/deploy.yml)：协议、host/port、benchmark、任务、本体、动作类型、seed 和 `eval_batch` 的运行时配置。

## 5. 远程通信协议

```text
RoboDojo / RealEval environment client
  ⇄ WebSocket binary frames
  ⇄ MessagePack + NumPy payload
XPolicyLab policy server
```

- [`client_server/ws/model_client.py`](https://github.com/XPolicyLab/XPolicyLab/blob/85ff90d4092a9530860c9efe96a732d9f9150954/client_server/ws/model_client.py) 把同步环境调用桥接到后台 asyncio WebSocket loop。
- [`client_server/ws/model_server.py`](https://github.com/XPolicyLab/XPolicyLab/blob/85ff90d4092a9530860c9efe96a732d9f9150954/client_server/ws/model_server.py) 负责模型方法分发、请求去重、重连语义和 bounded shutdown。
- [`protocol/codec.py`](https://github.com/XPolicyLab/XPolicyLab/blob/85ff90d4092a9530860c9efe96a732d9f9150954/client_server/ws/protocol/codec.py#L1-L124) 使用 MessagePack + `msgpack_numpy` 编码 NumPy/Torch 张量，并拒绝 object-dtype 数组。
- 这解释了论文所谓“同一策略实现最小改动连接仿真和真机”：环境侧换掉，策略服务和接口契约不变。

## 6. 分数聚合和验证脚本

- [`scripts/internal/summarize_result.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/scripts/internal/summarize_result.py) 汇总 case 结果并形成表格。
- [`scripts/internal/task_inventory.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/scripts/internal/task_inventory.py) 解析可用任务清单。
- [`scripts/internal/verify_install.sh`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/scripts/internal/verify_install.sh) 做安装检查；[`smoke_all_tasks.sh`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/57b8c52424ff1198ade242505656511515c58792/scripts/internal/smoke_all_tasks.sh) 批量冒烟任务。

快照中的 RoboDojo 主仓库没有匹配 `*test*.py` 的测试文件，因此公开验证主要体现为安装/任务 smoke 脚本，而不是单元测试套件。XPolicyLab 含大量策略上游测试，但没有看到独立覆盖核心 WebSocket 协议的顶层测试目录；这不等于“代码没有被测试”，只是公开树中可直接审计的自动化测试边界。

## 7. 论文与活跃仓库的差异

- 论文冻结口径：XPolicyLab 集成 30 个策略。
- 2026-08-20 检查的官方 README：40+；源码快照 `policy/` 下有 41 个一级 adapter 目录。
- 论文没有写当前 release 的“eval-only”边界；活跃 RoboDojo README 明确写明 simulator client、任务、资产/配置验证和结果产物属于 RoboDojo，策略训练/服务由 XPolicyLab 负责。
- 这些是论文后的工程演进，学习论文结论时仍应使用 2026-07-03 冻结榜单。
