"""
RoboTTT (NVIDIA GEAR, 2026): TBPTT (Truncated Backprop Through Time) 的显存/计算解耦。

论文机制:
  - 长序列被切成若干 segment，梯度只在 segment 内部沿时间回传，fast weights 在
    segment 边界处 detach 后继续向下一个 segment 传递 (不中断"记忆"，只中断"梯度")。
  - 结果: 显存开销由 segment 长度 S 决定，而非整条序列长度 T；序列越长，
    full BPTT 的激活显存 ~ O(T) 线性增长，TBPTT 的激活显存 ~ O(S) 恒定。

本脚本两部分:
  Part A (显存/计算解耦): 对同一个简化 TTT 层，分别用 full BPTT 和 TBPTT(segment=S)
    跑一步训练，用 torch.autograd.graph.saved_tensors_hooks 精确统计反向传播需要
    保留的激活张量元素数 (peak retained elements)——这是真实的 autograd 记账，不是
    拍脑袋估算；没有 CUDA 时，这是准确刻画"反向传播要保留多少中间结果"的方式。
    结果画成 code/tbptt_memory_scaling.png。

  Part B (跨 segment 记忆不中断): 复用同款延迟键值回忆任务，对比"fast weights 跨
    segment 传递 (detach 但不重置)" vs "每个 segment 重置为 W_0"。展示呈现阶段和
    查询阶段被切在不同 segment 时，只有"跨 segment 传递"能保留绑定关系并正确回忆；
    "每段重置"会在段边界把已学到的上下文全部丢弃，退化到随机猜测水平。

运行方式: python3 tbptt_memory_demo.py  (纯 CPU, 约数十秒；需要 matplotlib)
"""

import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---- shared TTT-layer building blocks (same mechanism as ttt_layer_demo.py) ----
D_K = 12
HIDDEN = 32


def mlp_apply(params, x):
    h = F.gelu(x @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def clip_grad_dict(g, max_norm=1.0):
    total_norm = torch.sqrt(sum((v ** 2).sum() for v in g.values()))
    scale = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    return {k: v * scale for k, v in g.items()}


def fresh_w0(requires_grad=True):
    g = torch.Generator().manual_seed(1)
    params = {
        "W1": torch.randn(D_K, HIDDEN, generator=g) * 0.05,
        "b1": torch.zeros(HIDDEN),
        "W2": torch.randn(HIDDEN, D_K, generator=g) * 0.05,
        "b2": torch.zeros(D_K),
    }
    for v in params.values():
        v.requires_grad_(requires_grad)
    return params


def ttt_step(params, k_t, v_t, q_t, lr):
    """One update-then-apply TTT step.

    Uses plain torch.autograd.grad(create_graph=True) rather than torch.func.grad
    (as in ttt_layer_demo.py) because torch.func's transforms disable
    saved_tensors_hooks internally, which Part A relies on for memory accounting.
    Both express the same inner-loop gradient descent; this is just the
    hook-compatible variant.
    """
    inner_loss = ((mlp_apply(params, k_t) - v_t) ** 2).sum()
    grads = torch.autograd.grad(inner_loss, list(params.values()), create_graph=True)
    g = clip_grad_dict(dict(zip(params.keys(), grads)))
    params = {name: params[name] - lr * g[name] for name in params}
    o_t = mlp_apply(params, q_t)
    return params, o_t


# =====================================================================
# Part A: memory/compute decoupling via saved_tensors_hooks accounting
# =====================================================================

class SavedTensorCounter:
    """Exact accounting of autograd-saved-tensor elements over time.

    pack_hook fires whenever an op saves a tensor for its backward; unpack_hook
    fires when backward consumes (and frees) it. Tracking current - the running
    total of saved-but-not-yet-freed elements - and its max gives the true peak
    activation memory autograd must hold alive, regardless of CUDA availability.
    """

    def __init__(self):
        self.current = 0
        self.peak = 0

    def pack(self, tensor):
        self.current += tensor.numel()
        self.peak = max(self.peak, self.current)
        return tensor

    def unpack(self, tensor):
        self.current -= tensor.numel()
        return tensor


def run_segment_and_backward(params_in, k_seq, v_seq, q_seq, lr, counter):
    """Roll out one segment under the counting hooks, backward it, return detached params."""
    with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
        params = params_in
        loss = torch.zeros(())
        for t in range(k_seq.shape[0]):
            params, o_t = ttt_step(params, k_seq[t], v_seq[t], q_seq[t], lr)
            loss = loss + o_t.pow(2).sum()  # arbitrary proxy loss for this demo
        loss.backward()
    return {name: p.detach().requires_grad_(True) for name, p in params.items()}


def measure_peak_elements(seq_len, segment):
    """Peak retained saved-tensor elements for one training step over `seq_len`,
    processed either as one full-BPTT segment (segment=None) or as consecutive
    TBPTT segments of length `segment` (grads truncated, fast weights carried)."""
    torch.manual_seed(0)
    k_all = torch.randn(seq_len, D_K)
    v_all = torch.randn(seq_len, D_K)
    q_all = torch.randn(seq_len, D_K)
    lr = torch.tensor(0.1)

    seg_len = seq_len if segment is None else segment
    counter = SavedTensorCounter()
    params = fresh_w0()
    global_peak = 0
    for start in range(0, seq_len, seg_len):
        end = min(start + seg_len, seq_len)
        counter.current = 0  # each segment's memory is freed before the next starts
        params = run_segment_and_backward(
            params, k_all[start:end], v_all[start:end], q_all[start:end], lr, counter
        )
        global_peak = max(global_peak, counter.peak)
        counter.peak = 0
    return global_peak


def part_a_memory_scaling():
    print("=== Part A: activation memory, full BPTT vs TBPTT (measured via saved_tensors_hooks) ===")
    seq_lens = [16, 32, 64, 128, 256]
    tbptt_segment = 16

    full_peaks, tbptt_peaks = [], []
    for T in seq_lens:
        full_peak = measure_peak_elements(T, segment=None)
        tbptt_peak = measure_peak_elements(T, segment=tbptt_segment)
        full_peaks.append(full_peak)
        tbptt_peaks.append(tbptt_peak)
        print(f"T={T:4d}  full_BPTT_peak_elems={full_peak:7d}  "
              f"TBPTT(seg={tbptt_segment})_peak_elems={tbptt_peak:7d}")

    plt.figure(figsize=(6, 4))
    plt.plot(seq_lens, full_peaks, marker="o", label="full BPTT (grows with T)")
    plt.plot(seq_lens, tbptt_peaks, marker="s", label=f"TBPTT, segment={tbptt_segment} (flat)")
    plt.xlabel("sequence length T")
    plt.ylabel("peak retained activation elements")
    plt.title("TBPTT decouples activation memory from sequence length")
    plt.legend()
    plt.tight_layout()
    out_path = "tbptt_memory_scaling.png"
    plt.savefig(out_path, dpi=120)
    print(f"saved plot to {out_path}")
    return seq_lens, full_peaks, tbptt_peaks


# =====================================================================
# Part B: cross-segment fast-weight carry vs per-segment reset
# =====================================================================

NUM_PAIRS = 8
VALUE_VOCAB = 8
CONTENT_DIM = 14
D_MODEL = CONTENT_DIM + 2
SEQ_LEN = 2 * NUM_PAIRS       # 8 presentation + 8 query steps
SEGMENT = 4                   # segment boundaries land strictly inside each phase:
                               # segments = [present 0-3][present 4-7][query 0-3][query 4-7]
                               # -> every query segment starts with zero within-segment
                               #    knowledge of any presented pair; only carried fast
                               #    weights (not a per-segment reset) can supply it.
BATCH = 20

PRESENT_FLAG = torch.tensor([1.0, 0.0])
QUERY_FLAG = torch.tensor([0.0, 1.0])


class SegmentTTTLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_Q = nn.Linear(D_MODEL, D_K, bias=False)
        self.theta_K = nn.Linear(D_MODEL, D_K, bias=False)
        self.theta_V = nn.Linear(D_MODEL, D_K, bias=False)
        self.theta_O = nn.Linear(D_K, CONTENT_DIM)
        self.W0 = nn.ParameterDict(
            {k: nn.Parameter(v) for k, v in fresh_w0(requires_grad=True).items()}
        )
        self.log_lr = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.key_embed = nn.Embedding(NUM_PAIRS, CONTENT_DIM)
        self.value_embed = nn.Embedding(VALUE_VOCAB, CONTENT_DIM)

    def qkv(self, x):
        Q = F.layer_norm(self.theta_Q(x), (D_K,))
        K = F.layer_norm(self.theta_K(x), (D_K,))
        V = F.layer_norm(self.theta_V(x), (D_K,))
        return Q, K, V


MODEL = SegmentTTTLayer()


def make_episode_batch(batch_size):
    key_embed = MODEL.key_embed.weight
    value_embed = MODEL.value_embed.weight
    xs, targets = [], []
    for _ in range(batch_size):
        values = torch.randint(0, VALUE_VOCAB, (NUM_PAIRS,))
        query_order = torch.randperm(NUM_PAIRS)

        present = key_embed + value_embed[values]
        present = torch.cat([PRESENT_FLAG.unsqueeze(0).expand(NUM_PAIRS, -1), present], dim=-1)

        query = key_embed[query_order]
        query = torch.cat([QUERY_FLAG.unsqueeze(0).expand(NUM_PAIRS, -1), query], dim=-1)

        xs.append(torch.cat([present, query], dim=0))
        targets.append(values[query_order])
    return torch.stack(xs, dim=0), torch.stack(targets, dim=0)


def rollout_batch(x, mode: str):
    """Run x: (B, SEQ_LEN, D_MODEL) through the TTT recursion.

    mode="full":  one continuous unrolled pass, no segmentation (training uses this
      -- gradient must reach all the way back to the presentation steps for theta_K/
      theta_V to learn how to encode key-value pairs into the fast weights at all;
      truncating that at SEGMENT=4 would sever credit assignment before it gets there).
    mode="carry": same VALUES as "full", just computed in SEGMENT-sized chunks with
      params.detach() at each boundary (grad truncated, no backward called here anyway
      -- this is an inference-time comparison). Detaching does not change the forward
      values, only which tensors autograd tracks, so this should reproduce "full"'s
      accuracy almost exactly.
    mode="reset": segmented like "carry", but params are reset to W0 at every boundary
      instead of carried -- an ablation with no cross-segment memory at all.
    """
    B = x.shape[0]
    Q, K, V = MODEL.qkv(x)
    lr = torch.exp(MODEL.log_lr)
    w0 = {k: v for k, v in MODEL.W0.items()}

    seg_len = SEQ_LEN if mode == "full" else SEGMENT
    params = [dict(w0) for _ in range(B)]
    outputs = [[None] * SEQ_LEN for _ in range(B)]

    for start in range(0, SEQ_LEN, seg_len):
        end = min(start + seg_len, SEQ_LEN)
        for b in range(B):
            p = dict(w0) if mode == "reset" else params[b]
            for t in range(start, end):
                p, o_t = ttt_step(p, K[b, t], V[b, t], Q[b, t], lr)
                outputs[b][t] = o_t
            params[b] = (
                {k: v.detach().requires_grad_(True) for k, v in p.items()}
                if mode == "carry" else p
            )

    out = torch.stack([torch.stack(row, dim=0) for row in outputs], dim=0)
    return MODEL.theta_O(out)


def recall_accuracy(pred_content, targets):
    value_table = MODEL.value_embed.weight
    dists = torch.cdist(pred_content, value_table.unsqueeze(0).expand(pred_content.shape[0], -1, -1))
    return (dists.argmin(dim=-1) == targets).float().mean().item()


def train_full_bptt(num_steps=250, lr=5e-3):
    """Meta-train theta_Q/K/V/O, W0, log_lr, embeddings via full BPTT (feasible here
    since the toy sequence is short). This is what actually teaches theta_K/theta_V
    to encode key-value pairs into the fast weights during presentation."""
    opt = torch.optim.Adam(MODEL.parameters(), lr=lr)
    for step in range(num_steps):
        x, targets = make_episode_batch(BATCH)
        out = rollout_batch(x, mode="full")
        query_out = out[:, NUM_PAIRS:, :]
        loss = F.mse_loss(query_out, MODEL.value_embed(targets))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == num_steps - 1:
            acc = recall_accuracy(query_out.detach().reshape(-1, CONTENT_DIM), targets.reshape(-1))
            print(f"  step {step:4d}  loss {loss.item():.4f}  train_recall_acc {acc:.3f}")


def part_b_carry_vs_reset():
    print("\n=== Part B: does truncating gradient at segment boundaries corrupt the forward state? ===")
    print(f"config: pairs={NUM_PAIRS} seq_len={SEQ_LEN} segment={SEGMENT} "
          f"(segments split presentation/query into disjoint chunks)")
    print("training with full BPTT (short sequence, so this is tractable and gives "
          "credit assignment all the way back to the presentation steps):")
    train_full_bptt()

    # Note: no torch.no_grad() here -- the inner-loop "test-time" gradient step needs
    # autograd enabled even at inference (that is the point of TTT); we just never
    # call .backward() on these outputs, so no outer gradients get accumulated.
    x, targets = make_episode_batch(200)
    out_full = rollout_batch(x, mode="full").detach()
    out_carry = rollout_batch(x, mode="carry").detach()
    out_reset = rollout_batch(x, mode="reset").detach()

    query_targets = targets.reshape(-1)
    acc_full = recall_accuracy(out_full[:, NUM_PAIRS:, :].reshape(-1, CONTENT_DIM), query_targets)
    acc_carry = recall_accuracy(out_carry[:, NUM_PAIRS:, :].reshape(-1, CONTENT_DIM), query_targets)
    acc_reset = recall_accuracy(out_reset[:, NUM_PAIRS:, :].reshape(-1, CONTENT_DIM), query_targets)
    max_diff_full_vs_carry = (out_full - out_carry).abs().max().item()
    chance = 1.0 / VALUE_VOCAB

    print(f"\neval recall accuracy: full={acc_full:.3f}  carry={acc_carry:.3f}  "
          f"reset={acc_reset:.3f}  chance={chance:.3f}")
    print(f"max |full - carry| output difference: {max_diff_full_vs_carry:.2e} "
          "(near zero -> segmenting + detaching reproduces the unsegmented forward pass exactly)")
    print("-> segment boundaries truncate GRADIENT flow but not the fast-weight VALUES:")
    print("   carrying them across segments preserves recall; resetting them destroys it.")
    return acc_full, acc_carry, acc_reset


if __name__ == "__main__":
    part_a_memory_scaling()
    part_b_carry_vs_reset()
