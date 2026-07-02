"""
MagicSim 核心思想的可运行概念实现
====================================

论文: MagicSim — One Runtime, One MDP, Three Drivers (arXiv:2606.17511v1)

本 demo 用纯 Python 标准库(无任何外部依赖)复刻 MagicSim 的三个核心机制,
让你在没有 Isaac Sim/GPU 的情况下也能"跑通"其设计直觉:

  1. 一个 MDP,三个驱动器 (§3.4)
     —— 同一个 TaskEnv 被 RL / AutoCollect / Replay 三种 driver 复用。

  2. Agent 级异步:物理时间同步,语义时间异步 (§3.5)
     —— 多个子环境跨越同一个 batched sim_step() 屏障,
        但各自的语义阶段(采样指令/技能/规划/执行)独立推进。
     —— planner 用 submit-and-poll 的 future,慢规划不冻结整个 batch。

  3. Episode 而非 Frame + 成功门控录制 (§3.6)
     —— 只有成功且结束的 episode 才被保存为轨迹;失败的缓冲被丢弃。

注意: 这是教学用的"骨架"实现,刻意省略真实物理/IK/渲染,
      用计数器和随机延迟模拟"规划耗时不固定"。重点在**控制流与状态机**,
      这正是 MagicSim 的系统贡献所在。

运行:  python3 one_mdp_three_drivers_demo.py
"""

import random
import collections

# 用固定种子保证"确定性是运行时契约"(§5)——可复现的随机流。
RNG = random.Random(20260616)


# =============================================================================
# 1. 共享 MDP 边界:TaskEnv  (对应论文 TaskBaseEnv §3.3)
# =============================================================================
class TaskEnv:
    """一个 Gym 风格的批处理任务 MDP。

    obs/action/reward/termination/success 在这里定义一次,
    然后被三个 driver 复用。这就是"One MDP"。

    任务语义(玩具版): 机器人要把末端执行器从 pos 移动到 goal。
    每个子环境(env_id)有独立的 pos/goal/step 计数 —— per-env 状态切片(§3.2)。
    """

    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.reset()

    def reset(self, env_ids=None):
        """确定性 reset:可选择只 reset 部分 env_ids(对应 reset_idx §5)。"""
        if env_ids is None:
            env_ids = range(self.num_envs)
            self.pos = [0.0] * self.num_envs
            self.goal = [0.0] * self.num_envs
            self.steps = [0] * self.num_envs
        for e in env_ids:
            self.pos[e] = 0.0
            self.goal[e] = RNG.uniform(3.0, 6.0)   # seeded DR:可复现的目标随机化
            self.steps[e] = 0
        return self._get_obs()

    def _get_obs(self):
        # 观测 = (当前位置, 目标, 到目标的距离)
        return [(self.pos[e], self.goal[e], self.goal[e] - self.pos[e])
                for e in range(self.num_envs)]

    def step(self, actions, env_ids=None):
        """批处理物理 step:所有 env 跨越同一个 sim_step() 屏障(§3.2 物理时间同步)。

        action[e] 是该环境本 tick 施加的位移增量(由 driver/planner 决定)。
        """
        if env_ids is None:
            env_ids = range(self.num_envs)
        obs, reward, terminated, success = {}, {}, {}, {}
        for e in env_ids:
            self.pos[e] += actions.get(e, 0.0)
            self.steps[e] += 1
            dist = abs(self.goal[e] - self.pos[e])
            reward[e] = -dist                              # 越近奖励越高
            success[e] = dist < 0.15                       # 成功条件
            terminated[e] = success[e] or self.steps[e] >= 18   # 成功或超时(预算有限)
        # 返回与 obs 对齐的批量信息(对应 TaskBaseEnv.step 返回 obs/reward/term/success)
        return self._get_obs(), reward, terminated, success


# =============================================================================
# 2. 异步规划服务:submit-and-poll future + 微批求解  (§3.5 / §6.7)
# =============================================================================
class PlannerFuture:
    """非阻塞 future:提交后不立刻有结果,需轮询 done() 才能取 result()。"""
    def __init__(self, target_action, latency_ticks):
        self._target = target_action
        self._remaining = latency_ticks   # 模拟"规划耗时不固定"

    def tick(self):
        if self._remaining > 0:
            self._remaining -= 1

    def done(self):
        return self._remaining <= 0

    def result(self):
        assert self.done(), "future 未就绪就取结果 —— 这正是同步阻塞要避免的!"
        return self._target


class AsyncPlannerService:
    """异步微批求解农场(§6.7 的玩具版)。

    各 env 提交请求 -> 进入请求队列 -> 按微批窗口聚合 -> 批量"求解"(返回 future)。
    关键属性: 仿真主循环永不被慢解阻塞;请求被聚合成更大批量。
    """
    def __init__(self, microbatch_window=3):
        self.pending = collections.deque()   # 已提交、尚未开始求解的请求
        self.in_flight = []                   # 正在求解(倒计时)的 future
        self.microbatch_window = microbatch_window
        self.solve_count = 0   # 统计:发生了多少次微批求解

    def submit(self, env_id, current_pos, goal):
        """提交规划请求,返回一个 future(立即返回,不阻塞)。"""
        # 规划"耗时"随机(1~4 tick)—— 模拟 IK/MotionGen 的不定时长
        latency = RNG.randint(1, 4)
        # 25% 概率规划"不可行"(无碰撞自由解)-> 返回 no-op,该技能这一段无进展。
        # 这模拟真实世界里部分规划失败 -> 对应 episode 可能超时 -> 被成功门控丢弃。
        if RNG.random() < 0.25:
            target_step = 0.0
        else:
            target_step = (goal - current_pos) * 0.9   # 朝目标走剩余距离的 90%
        fut = PlannerFuture(target_step, latency)
        self.pending.append(fut)
        return fut

    def step(self):
        """每个物理 tick 推进一次:从 pending 取一个微批开始求解,推进所有在飞 future。

        这一步在主循环之外(off-loop),不阻塞 env 步进。
        关键: 在飞 future 会持续倒计时直到 done —— 慢解在后台跑,主循环照常步进。
        """
        # 1) 按微批窗口把 pending 请求聚合成一次批量求解
        if self.pending:
            batch = []
            while self.pending and len(batch) < self.microbatch_window:
                batch.append(self.pending.popleft())
            self.solve_count += 1            # 一次微批 = 一次批量求解(请求被聚合)
            self.in_flight.extend(batch)
        # 2) 推进所有在飞 future 一个 tick;完成的移出
        for fut in self.in_flight:
            fut.tick()
        self.in_flight = [f for f in self.in_flight if not f.done()]


# =============================================================================
# 3. AtomicSkill 状态机 + AutoCollect driver  (Command→Skill→Planner→Robot→Record)
# =============================================================================
class PerEnvCollectState:
    """每个 env 独立的语义状态(§3.5 agent-level asynchrony 的载体)。

    语义阶段独立推进: SAMPLE -> SUBMIT -> POLL -> EXECUTE -> ...
    不同 env 任意时刻可处于不同阶段。
    """
    def __init__(self):
        self.phase = "SAMPLE"      # 当前 AtomicSkill 阶段
        self.future = None         # 在飞的规划 future
        self.buffer = []           # per-env 录制缓冲(obs/action),成功才落盘
        self.done = False


class AutoCollectDriver:
    """Driver 2: 脚本数据采集 (current)。

    路径: Command -> Skill -> Planner -> Robot -> Record。
    用成功门控决定是否保存轨迹(episode 而非 frame, §3.6)。
    """
    def __init__(self, env: TaskEnv, planner: AsyncPlannerService):
        self.env = env
        self.planner = planner
        self.states = [PerEnvCollectState() for _ in range(env.num_envs)]
        self.saved_trajectories = []   # 仅成功 episode
        self.discarded = 0             # 失败被丢弃的计数

    def run(self, max_ticks=300):
        obs = self.env._get_obs()
        for tick in range(max_ticks):
            actions = {}
            # ---- 每个 env 按自己的语义阶段推进(异步)----
            for e, st in enumerate(self.states):
                if st.done:
                    continue
                pos, goal, dist = obs[e]

                if st.phase == "SAMPLE":
                    # Command: 采样/推进任务指令 -> 进入提交规划
                    st.phase = "SUBMIT"

                elif st.phase == "SUBMIT":
                    # Skill: 提交规划请求,拿 future(非阻塞)
                    st.future = self.planner.submit(e, pos, goal)
                    st.phase = "POLL"
                    actions[e] = 0.0     # 等待中:保持姿态/安全动作

                elif st.phase == "POLL":
                    # 轮询 future:未就绪就保持姿态,就绪则消费
                    if st.future.done():
                        actions[e] = st.future.result()    # Robot: 执行规划动作
                        st.phase = "SAMPLE"                 # 一个技能循环结束,继续下一段
                    else:
                        actions[e] = 0.0                    # 慢规划不冻结其它 env

            # ---- off-loop 异步规划服务推进一次(不阻塞主循环)----
            self.planner.step()

            # ---- 批处理物理 step:所有 env 跨越同一屏障 ----
            obs, reward, terminated, success = self.env.step(actions)

            # ---- Record: per-env 缓冲 + 成功门控落盘 ----
            for e, st in enumerate(self.states):
                if st.done:
                    continue
                st.buffer.append((obs[e], actions.get(e, 0.0)))
                if terminated[e]:
                    st.done = True
                    if success[e]:
                        # 成功且结束 -> 保存为轨迹(对齐 obs/action)
                        self.saved_trajectories.append({
                            "env_id": e, "length": len(st.buffer),
                            "final_dist": abs(obs[e][2]),
                        })
                    else:
                        # 失败/截断 -> 丢弃缓冲(flush/re-roll §3.6)
                        self.discarded += 1

            if all(st.done for st in self.states):
                print(f"  [AutoCollect] 全部 env 在第 {tick} tick 结束")
                break

        return self.saved_trajectories


# =============================================================================
# Driver 1: RL 训练(直接 step,跳过 Skill/Planner 层) §3.4
# =============================================================================
def rl_training_driver(env: TaskEnv, episodes_ticks=60):
    """policy 直接 step TaskEnv,不经过 Collect 层的技能/规划层级。"""
    obs = env.reset()
    total_reward = 0.0
    for _ in range(episodes_ticks):
        # 一个极简"策略":朝目标方向走一小步(代表 RobotManager 动作空间)
        actions = {e: 0.2 * (1 if obs[e][2] > 0 else -1) for e in range(env.num_envs)}
        obs, reward, terminated, success = env.step(actions)
        total_reward += sum(reward.values())
        if all(terminated.values()):
            break
    succ = sum(1 for e in range(env.num_envs) if abs(obs[e][2]) < 0.3)
    return total_reward, succ


# =============================================================================
# Driver 3: Replay(回放预录动作流,planned)§3.4
# =============================================================================
def replay_driver(env: TaskEnv, recorded_actions):
    """用回放动作流 step 同一 MDP —— 走 RobotManager 动作空间。"""
    env.reset()
    for actions in recorded_actions:
        env.step(actions)
    return [abs(o[2]) for o in env._get_obs()]


# =============================================================================
# 主程序:演示同一个 MDP 被三个 driver 复用
# =============================================================================
def main():
    print("=" * 68)
    print("MagicSim 概念 demo:One Runtime, One MDP, Three Drivers")
    print("=" * 68)

    NUM_ENVS = 6

    # ---- One MDP:定义一次 ----
    print(f"\n[共享 MDP] 创建 TaskEnv,num_envs = {NUM_ENVS}")
    print("  同一个任务定义将被三个 driver 复用,batched 物理 step 同步。\n")

    # ---- Driver 1: RL 训练 ----
    print("-" * 68)
    print("Driver 1 · RL 训练 (current):policy 直接 step TaskEnv")
    env1 = TaskEnv(num_envs=NUM_ENVS)
    total_reward, succ = rl_training_driver(env1)
    print(f"  累计奖励 = {total_reward:.1f} | 接近目标的 env 数 = {succ}/{NUM_ENVS}")

    # ---- Driver 2: AutoCollect(核心:异步规划 + 成功门控)----
    print("-" * 68)
    print("Driver 2 · AutoCollect (current):Command→Skill→Planner→Robot→Record")
    env2 = TaskEnv(num_envs=NUM_ENVS)
    planner = AsyncPlannerService(microbatch_window=3)
    collector = AutoCollectDriver(env2, planner)
    trajs = collector.run()
    print(f"  异步规划服务发生 {planner.solve_count} 次微批求解(请求被聚合)")
    print(f"  成功保存轨迹 {len(trajs)} 条 | 丢弃失败 {collector.discarded} 条")
    print("  >>> 体现'episode 而非 frame':只有成功且结束才落盘")
    for t in trajs:
        print(f"      env{t['env_id']}: 长度={t['length']} tick, 末端距离={t['final_dist']:.3f}")

    # ---- Driver 3: Replay ----
    print("-" * 68)
    print("Driver 3 · Replay (planned):回放预录动作流 step 同一 MDP")
    env3 = TaskEnv(num_envs=NUM_ENVS)
    # 造一段简单的回放动作流
    recorded = [{e: 0.3 for e in range(NUM_ENVS)} for _ in range(15)]
    final_dists = replay_driver(env3, recorded)
    print(f"  回放 {len(recorded)} 步后各 env 末端距离: "
          + ", ".join(f"{d:.2f}" for d in final_dists))

    print("\n" + "=" * 68)
    print("要点回顾:")
    print("  1. 三个 driver 复用同一个 TaskEnv(MDP)—— 不是三个仿真器。")
    print("  2. AutoCollect 中规划是 submit-and-poll 的 future,慢解不冻结 batch。")
    print("  3. 物理时间同步(同一 step 屏障),语义时间异步(各 env 阶段不同)。")
    print("  4. 成功门控:只有成功 episode 变成数据,失败缓冲被丢弃。")
    print("=" * 68)


if __name__ == "__main__":
    main()
