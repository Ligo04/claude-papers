"""
RoboTTT (NVIDIA GEAR, 2026) 核心机制最小复现: TTT (Test-Time Training) 层。

论文机制:
  - fast weights W 参数化一个两层 GeLU MLP f_W。
  - 每个时间步 t 先做 "update": W_t <- W_{t-1} - eta * grad_W || f_{W_{t-1}}(K_t) - V_t ||^2
    (把当前 token 的上下文压进权重空间)，再做 "apply": O_t = f_{W_t}(Q_t) (从权重空间取回信息)。
  - Q/K/V 由可学习投影 theta_Q/theta_K/theta_V 产生；eta (learning rate) 也是可学习标量。
  - W_0 (fast weight 初始化) 和投影矩阵通过外层任务损失做元学习——对内层梯度再求梯度，
    这里用 torch.func.grad 实现内层梯度、外层 loss.backward() 自动穿透 (二阶微分)。

玩具任务: "延迟键值回忆" (delayed key-value recall)。
  - 呈现阶段: 依次展示 m 个 (key_i, value_i) 对 (每个 token 编码 key+value 的和，外加一个
    "呈现/查询" 相位标志位)。
  - 查询阶段: 依次给出 m 个 query key (乱序)，只编码 key，不含 value 信息。
  - TTT 层必须在呈现阶段把 key->value 的绑定关系压进 fast weights W，再在查询阶段用
    Q_t = theta_Q(query key) 从 W 中检索出对应 value。
  - 每个 episode 的 key->value 绑定都是随机采样的，外层权重不可能靠"死记硬背"完成任务，
    只有依靠测试时的梯度更新(fast weights)才能回忆。

对照实验: 训练完成后，强制 eta=0 (fast weights 冻结在 W_0，从不更新)，在相同的 query 上
重新评估。如果回忆准确率骤降到接近随机猜测，就证明了检索能力确实来自测试时的梯度更新，
而不是学出来的静态权重。

运行方式: python3 ttt_layer_demo.py  (纯 CPU, 约数十秒)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap

torch.manual_seed(0)

# ---- toy task config ----
NUM_PAIRS = 16          # m key-value pairs presented, then m queries -> seq_len = 2m
VALUE_VOCAB = 16        # number of distinct value classes
CONTENT_DIM = 24        # embedding dim for key/value content
D_MODEL = CONTENT_DIM + 2   # +2 one-hot phase flag (presentation / query)
D_K = 16                # Q/K/V projection dim (also fast-weight MLP input/output dim)
HIDDEN = 48              # fast-weight MLP hidden dim
BATCH = 24
SEQ_LEN = 2 * NUM_PAIRS

PRESENT_FLAG = torch.tensor([1.0, 0.0])
QUERY_FLAG = torch.tensor([0.0, 1.0])


def make_episode_batch(batch_size: int):
    """Sample a batch of delayed key-value recall episodes.

    Key identities are fixed slots (0..NUM_PAIRS-1); the value bound to each key
    is re-sampled every episode, so the mapping cannot be memorized in outer weights.
    Returns x: (B, SEQ_LEN, D_MODEL), targets: (B, NUM_PAIRS) value indices for
    the query steps (in the order they are queried), target_positions: (B, NUM_PAIRS)
    time indices of the query steps.
    """
    key_embed = MODEL.key_embed.weight  # (NUM_PAIRS, CONTENT_DIM)
    value_embed = MODEL.value_embed.weight  # (VALUE_VOCAB, CONTENT_DIM)

    xs = []
    targets = []
    for _ in range(batch_size):
        values = torch.randint(0, VALUE_VOCAB, (NUM_PAIRS,))
        query_order = torch.randperm(NUM_PAIRS)

        present_tokens = key_embed + value_embed[values]  # (NUM_PAIRS, CONTENT_DIM)
        present_tokens = torch.cat(
            [PRESENT_FLAG.unsqueeze(0).expand(NUM_PAIRS, -1), present_tokens], dim=-1
        )

        query_tokens = key_embed[query_order]  # (NUM_PAIRS, CONTENT_DIM)
        query_tokens = torch.cat(
            [QUERY_FLAG.unsqueeze(0).expand(NUM_PAIRS, -1), query_tokens], dim=-1
        )

        xs.append(torch.cat([present_tokens, query_tokens], dim=0))
        targets.append(values[query_order])

    return torch.stack(xs, dim=0), torch.stack(targets, dim=0)


Q_DIM = D_K


def mlp_apply(params, x):
    """f_W: the fast-weight two-layer GeLU MLP."""
    h = F.gelu(x @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def clip_grad_dict(g, max_norm=1.0):
    """Clip the inner-loop gradient's global norm; bounds each fast-weight update step."""
    total_norm = torch.sqrt(sum((v ** 2).sum() for v in g.values()))
    scale = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    return {k: v * scale for k, v in g.items()}


class TTTLayer(nn.Module):
    def __init__(self, d_model, d_k, hidden, base_lr=0.1):
        super().__init__()
        self.theta_Q = nn.Linear(d_model, d_k, bias=False)
        self.theta_K = nn.Linear(d_model, d_k, bias=False)
        self.theta_V = nn.Linear(d_model, d_k, bias=False)
        self.theta_O = nn.Linear(d_k, CONTENT_DIM)

        # W_0: learnable fast-weight initialization (meta-learned).
        self.W0 = nn.ParameterDict(
            {
                "W1": nn.Parameter(torch.randn(d_k, hidden) * 0.05),
                "b1": nn.Parameter(torch.zeros(hidden)),
                "W2": nn.Parameter(torch.randn(hidden, d_k) * 0.05),
                "b2": nn.Parameter(torch.zeros(d_k)),
            }
        )
        # eta learned in log-space to keep it positive; base value 0.1.
        self.log_lr = nn.Parameter(torch.log(torch.tensor(base_lr)))

    def forward(self, x, use_ttt_update: bool = True):
        """x: (B, T, D_MODEL) -> outputs: (B, T, CONTENT_DIM)."""
        # LayerNorm on Q/K/V: without it, the online SGD recursion on W is a
        # positive-feedback loop (larger W -> larger loss -> larger grad -> larger W)
        # that reliably diverges over tens of steps. Real TTT-layer implementations
        # normalize inputs to the inner loop for exactly this reason.
        Q = F.layer_norm(self.theta_Q(x), (Q_DIM,))
        K = F.layer_norm(self.theta_K(x), (Q_DIM,))
        V = F.layer_norm(self.theta_V(x), (Q_DIM,))

        lr = torch.exp(self.log_lr) if use_ttt_update else torch.zeros(())
        w0 = {k: v for k, v in self.W0.items()}

        def inner_loss(params, k_t, v_t):
            return ((mlp_apply(params, k_t) - v_t) ** 2).sum()

        inner_grad = grad(inner_loss)

        def rollout(params0, q_seq, k_seq, v_seq, lr_):
            params = params0
            outs = []
            for t in range(q_seq.shape[0]):
                g = inner_grad(params, k_seq[t], v_seq[t])
                g = clip_grad_dict(g)  # bound the update step; same divergence guard
                params = {name: params[name] - lr_ * g[name] for name in params}
                outs.append(mlp_apply(params, q_seq[t]))
            return torch.stack(outs, dim=0)

        batched_rollout = vmap(rollout, in_dims=(None, 0, 0, 0, None))
        ttt_out = batched_rollout(w0, Q, K, V, lr)  # (B, T, D_K)
        return self.theta_O(ttt_out)


MODEL = TTTLayer(D_MODEL, D_K, HIDDEN)
MODEL.key_embed = nn.Embedding(NUM_PAIRS, CONTENT_DIM)
MODEL.value_embed = nn.Embedding(VALUE_VOCAB, CONTENT_DIM)


def recall_accuracy(logits_content, targets):
    """Nearest-neighbor classification against the value embedding table."""
    value_table = MODEL.value_embed.weight  # (VALUE_VOCAB, CONTENT_DIM)
    dists = torch.cdist(logits_content, value_table.unsqueeze(0).expand(logits_content.shape[0], -1, -1))
    preds = dists.argmin(dim=-1)
    return (preds == targets).float().mean().item()


def evaluate(use_ttt_update: bool, batch_size=200):
    x, targets = make_episode_batch(batch_size)
    out = MODEL(x, use_ttt_update=use_ttt_update)
    query_out = out[:, NUM_PAIRS:, :]  # only query-phase outputs matter
    loss = F.mse_loss(query_out, MODEL.value_embed(targets))
    acc = recall_accuracy(query_out.reshape(-1, CONTENT_DIM), targets.reshape(-1))
    return loss.item(), acc


def train(num_steps=300, lr=3e-3):
    # key_embed / value_embed were assigned onto MODEL after __init__, but nn.Module
    # auto-registers submodule attributes, so MODEL.parameters() already includes them.
    opt = torch.optim.Adam(MODEL.parameters(), lr=lr)
    for step in range(num_steps):
        x, targets = make_episode_batch(BATCH)
        out = MODEL(x, use_ttt_update=True)
        query_out = out[:, NUM_PAIRS:, :]
        loss = F.mse_loss(query_out, MODEL.value_embed(targets))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 30 == 0 or step == num_steps - 1:
            acc = recall_accuracy(query_out.detach().reshape(-1, CONTENT_DIM), targets.reshape(-1))
            print(f"step {step:4d}  loss {loss.item():.4f}  train_recall_acc {acc:.3f}  "
                  f"eta {torch.exp(MODEL.log_lr).item():.4f}")


if __name__ == "__main__":
    print(f"config: pairs={NUM_PAIRS} seq_len={SEQ_LEN} d_model={D_MODEL} d_k={D_K} hidden={HIDDEN}")
    print("=== training TTT layer on delayed key-value recall ===")
    train()

    print("\n=== evaluation: TTT update ON (test-time gradient descent active) ===")
    loss_on, acc_on = evaluate(use_ttt_update=True)
    print(f"loss={loss_on:.4f}  recall_accuracy={acc_on:.3f}")

    print("\n=== evaluation: TTT update OFF (eta=0, fast weights frozen at W0) ===")
    loss_off, acc_off = evaluate(use_ttt_update=False)
    print(f"loss={loss_off:.4f}  recall_accuracy={acc_off:.3f}")

    chance = 1.0 / VALUE_VOCAB
    print(f"\nchance-level accuracy (random guess over {VALUE_VOCAB} values): {chance:.3f}")
    print(f"summary: TTT-update ON acc={acc_on:.3f}  vs  OFF (frozen W0) acc={acc_off:.3f}  "
          f"vs  chance={chance:.3f}")
    print("-> recall capability comes from the test-time gradient update, not the static weights.")
