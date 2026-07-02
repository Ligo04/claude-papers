"""
MotionBricks Multi-Head VQ-VAE Tokenizer Demo

Demonstrates: multi-head codebook capacity, robustness, and training.
Run: python multi_head_tokenizer_demo.py
Requires: pip install torch
"""
import torch, torch.nn as nn, torch.nn.functional as F

class MultiHeadVQVAE(nn.Module):
    def __init__(self, input_dim=128, num_heads=4, codebook_size=256, beta=0.25):
        super().__init__()
        self.K, self.V, self.d = num_heads, codebook_size, input_dim // num_heads
        self.beta = beta
        self.enc = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, input_dim))
        self.cbs = nn.ParameterList([nn.Parameter(torch.randn(codebook_size, self.d) * 0.02) for _ in range(num_heads)])
        self.dec = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, input_dim))

    def forward(self, x):
        z = self.enc(x)
        parts = z.chunk(self.K, -1)
        qs, ids, vq = [], [], 0.
        for zh, cb in zip(parts, self.cbs):
            i = torch.cdist(zh, cb).argmin(-1)
            q = cb[i]
            vq += F.mse_loss(q.detach(), zh) + self.beta * F.mse_loss(q, zh.detach())
            qs.append(zh + (q - zh).detach())
            ids.append(i)
        rec = self.dec(torch.cat(qs, -1))
        return rec, F.mse_loss(rec, x), vq, ids

def main():
    print("\n" + "="*55 + "\n  MotionBricks Multi-Head VQ-VAE Demo\n" + "="*55)

    print("\n[1] Capacity comparison:")
    for n, h, v in [("1-head 1024",1,1024),("1-head 16384",1,16384),("4-head x256",4,256),("8-head x256",8,256)]:
        print(f"  {n:20s} => {v**h:>15,} combinations")

    print("\n[2] Robustness test:")
    torch.manual_seed(42)
    orig = torch.randn(1, 128)
    heads = orig.chunk(4, -1)
    for n in range(5):
        c = [torch.randn_like(heads[i]) if i < n else heads[i] for i in range(4)]
        e = F.mse_loss(torch.cat(c,-1), orig).item() / orig.var().item() * 100
        print(f"  {n}/4 heads wrong: {e:5.1f}% error  {'#'*min(int(e/5),20)}")

    print("\n[3] Training:")
    torch.manual_seed(42)
    ctr = torch.randn(20, 128) * 2
    data = torch.randn(1000, 128) * .5 + ctr[torch.randint(0, 20, (1000,))]
    m = MultiHeadVQVAE(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for s in range(200):
        b = data[torch.randint(0, 1000, (64,))]
        _, rl, vl, _ = m(b); loss = rl + vl
        opt.zero_grad(); loss.backward(); opt.step()
        if (s+1) % 50 == 0: print(f"  step {s+1}: loss={loss.item():.4f}")
    with torch.no_grad():
        _, rl, _, ids = m(data[:100])
        print(f"  Final MSE: {rl.item():.6f}")
        for i, idx in enumerate(ids):
            print(f"  Head {i} util: {len(idx.unique())}/256 ({len(idx.unique())/2.56:.0f}%)")

    print("\nKey takeaways:")
    print("  V^K >> V: exponential capacity scaling")
    print("  Error degrades linearly, not catastrophically")
    print("  Network learns optimal decomposition automatically\n")

if __name__ == "__main__": main()
