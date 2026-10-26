#!/usr/bin/env python3
"""用一维受控系统演示 SuperMimic 的采样、模型拟合和策略反传。

运行：python3 differentiable_control_loop.py --output ../sources/demo-results.json
依赖：PyTorch。只使用 CPU，不下载任何模型或数据。

这里的动力学、网络、奖励和实验数字均为教学构造。保留论文的 k=8、
H=32 和 gamma=0.99，省略机器人接触、历史观测、终止头及学习式 critic。
本程序不复现论文的训练速度、动作覆盖率或真机结果。
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn


K = 8
H = 32
GAMMA = 0.99


class ContextDynamics(nn.Module):
    """上下文改变状态衰减率，目标不改变物理规律。"""

    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([0.70, 0.00, 0.15]))

    def forward(self, state, action, context):
        a, context_slope, b = self.coefficients
        return (a + context_slope * context) * state + b * action


def simulator_step(state, action, context):
    """只在无梯度采样和最终评价中调用这个教学环境。"""
    return (0.84 + 0.08 * context) * state + 0.25 * action


def policy(weights, observation, goal):
    return 3.0 * torch.tanh(weights[0] * observation + weights[1] * goal)


@torch.no_grad()
def collect_windows(weights, batch_size=64):
    context = torch.rand(batch_size, 1) * 2.0 - 1.0
    state = torch.randn(batch_size, 1) * 0.4
    goal = (torch.randint(0, 4, (batch_size, 1)) * 2.0 - 3.0) / 3.0
    states, actions = [state], []
    for _ in range(K):
        observation = state + 0.01 * torch.randn_like(state)
        # 日志必须存实际执行的含噪动作，才能与下一状态对应。
        executed = policy(weights, observation, goal) + 0.10 * torch.randn_like(state)
        state = simulator_step(state, executed, context)
        actions.append(executed)
        states.append(state)
    return torch.stack(states), torch.stack(actions), context, goal


def model_loss(model, window):
    states, actions, context, _ = window
    predicted = states[0]
    loss = predicted.new_zeros(())
    for step in range(K):
        # 每步使用自己的预测，避免把 teacher forcing 误称为多步自回归。
        predicted = model(predicted, actions[step], context)
        loss = loss + (predicted - states[step + 1]).square().mean()
    return loss / K


def imagined_cost(weights, model, initial, goal, context, horizon=H, sensor_noise=False):
    state = initial
    cost = state.new_zeros(())
    for step in range(horizon):
        observation = state
        if sensor_noise:
            observation = observation + 0.01 * torch.randn_like(state)
        action = policy(weights, observation, goal)
        state = model(state, action, context)
        cost = cost + GAMMA**step * (
            (state - goal).square() + 0.01 * action.square()
        ).mean()
    return cost / horizon


@torch.no_grad()
def evaluate(weights):
    # 固定网格，避免优化前后用不同样本制造改善。
    contexts = torch.linspace(-1.0, 1.0, 9)
    goals = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
    starts = torch.tensor([-0.5, 0.0, 0.5])
    context, goal, state = torch.meshgrid(contexts, goals, starts, indexing="ij")
    context, goal, state = [x.reshape(-1, 1) for x in (context, goal, state)]
    squared_errors = []
    for _ in range(100):
        state = simulator_step(state, policy(weights, state, goal), context)
        squared_errors.append((state - goal).square().mean())
    return float(torch.stack(squared_errors).mean().sqrt())


def gradient_checks():
    model = ContextDynamics().double().requires_grad_(False)
    weights = nn.Parameter(torch.tensor([-0.12, 0.21], dtype=torch.float64))
    state = torch.tensor([[0.2], [-0.3]], dtype=torch.float64)
    goal = torch.tensor([[1.0], [-0.6]], dtype=torch.float64)
    context = torch.tensor([[0.4], [-0.7]], dtype=torch.float64)
    before = model.coefficients.detach().clone()
    loss = imagined_cost(weights, model, state, goal, context)
    analytic = torch.autograd.grad(loss, weights)[0]
    finite = torch.zeros_like(weights)
    epsilon = 1e-6
    with torch.no_grad():
        for i in range(weights.numel()):
            offset = torch.zeros_like(weights)
            offset[i] = epsilon
            plus = imagined_cost(weights + offset, model, state, goal, context)
            minus = imagined_cost(weights - offset, model, state, goal, context)
            finite[i] = (plus - minus) / (2.0 * epsilon)
    predicted = model(state, policy(weights, state, goal), context)
    path_gradient = torch.autograd.grad(predicted.square().mean(), weights)[0]
    wrong_objective = predicted.detach().square().mean()
    result = {
        "analytic_gradient": analytic.tolist(),
        "finite_difference_gradient": finite.tolist(),
        "max_absolute_error": float((analytic - finite).abs().max()),
        "frozen_model_weights_unchanged": bool(torch.equal(before, model.coefficients)),
        "frozen_model_parameter_has_no_grad": model.coefficients.grad is None,
        "gradient_through_frozen_model_nonzero": bool(path_gradient.norm() > 0),
        "detached_state_objective_requires_grad": wrong_objective.requires_grad,
    }
    assert result["max_absolute_error"] < 1e-6
    assert result["frozen_model_weights_unchanged"]
    assert result["frozen_model_parameter_has_no_grad"]
    assert result["gradient_through_frozen_model_nonzero"]
    assert not result["detached_state_objective_requires_grad"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=70)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations 必须为正数")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    checks = gradient_checks()
    model = ContextDynamics()
    weights = nn.Parameter(torch.tensor([-0.10, 0.10]))
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    actor_optimizer = torch.optim.Adam([weights], lr=0.04)
    initial_rmse = evaluate(weights)
    replay = []
    trace = []
    for iteration in range(args.iterations):
        replay.append(collect_windows(weights))
        # 保留早期策略数据，同时让新策略及时补充实际访问的状态。
        replay = replay[-32:]
        model.requires_grad_(True)
        for _ in range(10):
            window = replay[int(torch.randint(len(replay), ()).item())]
            model_optimizer.zero_grad(set_to_none=True)
            fit_loss = model_loss(model, window)
            fit_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model_optimizer.step()
        model_optimizer.zero_grad(set_to_none=True)
        # 冻结参数，不能用 no_grad 包裹下面的 imagination。
        model.requires_grad_(False)
        states, _, context, goal = replay[-1]
        before = model.coefficients.detach().clone()
        actor_optimizer.zero_grad(set_to_none=True)
        objective = imagined_cost(weights, model, states[0], goal, context, sensor_noise=True)
        objective.backward()
        nn.utils.clip_grad_norm_([weights], 1.0)
        actor_optimizer.step()
        assert torch.equal(before, model.coefficients)
        if iteration % 10 == 0 or iteration == args.iterations - 1:
            trace.append({"iteration": iteration + 1, "model_loss": float(fit_loss.detach()),
                          "imagined_cost": float(objective.detach()), "real_system_rmse": evaluate(weights)})
    final_rmse = evaluate(weights)
    result = {
        "scope": "一维合成系统教学验证，不是 SuperMimic 论文复现",
        "device": "cpu", "torch_version": torch.__version__, "seed": args.seed,
        "model_fit_window": K, "imagination_horizon": H, "discount": GAMMA,
        "omissions": ["接触", "观测历史", "学习式 critic", "终止头", "机器人及真机"],
        "gradient_checks": checks,
        "training": {"iterations": args.iterations, "true_coefficients": [0.84, 0.08, 0.25],
                     "learned_coefficients": model.coefficients.detach().tolist(),
                     "policy_weights": weights.detach().tolist(),
                     "initial_real_system_rmse": initial_rmse,
                     "final_real_system_rmse": final_rmse,
                     "trace": trace},
        "prediction_error_does_not_bound_jacobian_error": {
            "example": "f_hat(a)=f(a)+epsilon*sin(frequency*a)",
            "synthetic_epsilon": 0.001, "synthetic_frequency": 10000,
            "prediction_error_at_zero": 0.0,
            "derivative_error_at_zero": 10.0,
            "maximum_prediction_error": 0.001},
        "elapsed_cpu_wall_seconds": time.perf_counter() - started,
    }
    assert all(math.isfinite(v) for v in model.coefficients.detach().tolist())
    if args.iterations >= 70:
        assert final_rmse < initial_rmse * 0.75, "教学控制器没有改善，检查实现或版本。"
        assert float(fit_loss.detach()) < 1e-4, "教学动力学没有拟合成功。"
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
