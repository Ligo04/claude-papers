#!/usr/bin/env python3
"""检查训练期接触监督的信息流；依赖 PyTorch，合成输入，无 PPO 训练。"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ProprioceptiveActor(nn.Module):
    def __init__(self):
        super().__init__()
        # 论文给出 30 帧、33 维输入和 32 维 latent。
        # 卷积层、池化与激活的以下选择仅为教学，论文没有给全这些配置。
        self.encoder = nn.Sequential(
            nn.Conv1d(33, 64, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
        )
        self.action_head = nn.Sequential(
            nn.Linear(33 + 32, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 7),
        )
        self.contact_decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64 * 3),
        )

    def encode(self, history):
        return self.encoder(history.transpose(1, 2))

    def forward(self, history):
        # 推理入口只接收可部署观测，接触真值与 decoder 输出都不在这里。
        latent = self.encode(history)
        action = self.action_head(torch.cat((history[:, -1], latent), dim=-1))
        return action


def gradient_norm(module):
    return sum(
        float(parameter.grad.detach().square().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    ) ** 0.5


def main():
    torch.manual_seed(23849)
    torch.set_num_threads(1)
    model = ProprioceptiveActor()
    history = torch.randn(4, 30, 33)
    labels = torch.randint(0, 3, (4, 64))
    with torch.no_grad():
        before = model(history)
    latent = model.encode(history)
    logits = model.contact_decoder(latent).reshape(4, 64, 3)
    # 类权重来自附录 A；这里的 loss reduction 是教学选择。
    loss = F.cross_entropy(
        logits.reshape(-1, 3),
        labels.reshape(-1),
        weight=torch.tensor([0.25, 3.0, 2.0]),
    )
    loss.backward()
    encoder_gradient = gradient_norm(model.encoder)
    decoder_gradient = gradient_norm(model.contact_decoder)
    actor_head_gradient = gradient_norm(model.action_head)
    assert encoder_gradient > 0 and decoder_gradient > 0
    assert actor_head_gradient == 0

    # 删除 decoder 后前向结果不变，验证部署的观测边界。
    del model.contact_decoder
    with torch.no_grad():
        after = model(history)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert after.shape == (4, 7)

    # 附录 A 的动作语义：目标相对 measured q，而非上一次 target 累加。
    measured_q = history[:, -1, :7]
    deployed_scale = 0.04
    target = measured_q + deployed_scale * after
    assert torch.isfinite(target).all()
    report = {
        "性质": "未训练网络、合成输入、教学信息流检查",
        "设备": "CPU",
        "PyTorch版本": str(torch.__version__),
        "输入形状": list(history.shape),
        "输出形状": list(after.shape),
        "辅助损失对编码器有梯度": encoder_gradient > 0,
        "辅助损失对decoder有梯度": decoder_gradient > 0,
        "辅助损失对actor动作头无梯度": actor_head_gradient == 0,
        "删除decoder后动作逐值不变": True,
        "已运行原论文训练": False,
        "原论文成功率复现": False,
    }
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "information-flow-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
