#!/usr/bin/env python3
"""Teaching proxy for π0.5's hierarchical VLA recipe.

This is deliberately small and deterministic. It is not an official π0.5
implementation or a robot simulator. It shows two ideas from the paper:
(1) a semantic subtask is predicted before a low-level action chunk, and
(2) discrete-token cross-entropy and flow-matching MSE can be combined.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class ObjectState:
    name: str
    location: str


class SemanticPolicy:
    """Toy high-level policy: choose the next unfinished semantic subtask."""

    def __init__(self, target_location: str) -> None:
        self.target_location = target_location

    def predict(self, objects: list[ObjectState]) -> str | None:
        for obj in objects:
            if obj.location == "counter":
                return f"pick up {obj.name}"
            if obj.location == "gripper":
                return f"put {obj.name} in {self.target_location}"
        return None


class ActionExpert:
    """Toy continuous action expert producing a short 2-D action chunk."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def denoise(self, subtask: str, steps: int = 10) -> list[tuple[float, float]]:
        # Start from a noisy action and move it toward a subtask-specific target.
        target = (1.0, 0.0) if subtask.startswith("pick") else (0.0, 1.0)
        action = [self.rng.gauss(0.0, 0.6), self.rng.gauss(0.0, 0.6)]
        chunk = []
        for _ in range(steps):
            action[0] += (target[0] - action[0]) / (steps + 1)
            action[1] += (target[1] - action[1]) / (steps + 1)
            chunk.append((round(action[0], 3), round(action[1], 3)))
        return chunk


def cross_entropy(target: int, logits: list[float]) -> float:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    log_z = m + math.log(sum(exps))
    return -logits[target] + log_z


def flow_matching_mse(clean: list[float], noise: list[float], prediction: list[float], tau: float) -> float:
    # The paper trains against the vector field noise - action.
    target = [n - a for n, a in zip(noise, clean)]
    interpolated = [tau * a + (1.0 - tau) * n for a, n in zip(clean, noise)]
    del interpolated  # Included to make the correspondence with the paper explicit.
    return sum((p - t) ** 2 for p, t in zip(prediction, target)) / len(target)


def run_episode() -> None:
    objects = [ObjectState("plate", "counter"), ObjectState("cup", "counter")]
    high_level = SemanticPolicy("sink")
    expert = ActionExpert()

    print("Toy π0.5 episode: command = 'put the dishes in the sink'")
    for step in range(6):
        subtask = high_level.predict(objects)
        if subtask is None:
            print("done")
            break
        chunk = expert.denoise(subtask)
        print(f"step {step + 1}: {subtask:22s} action_chunk[0]={chunk[0]} action_chunk[-1]={chunk[-1]}")
        parts = subtask.split()
        name = parts[1] if parts[0] == "put" else parts[-1]
        for obj in objects:
            if obj.name == name:
                obj.location = "gripper" if obj.location == "counter" else "sink"
                break

    ce = cross_entropy(target=1, logits=[0.2, 1.4, -0.1])
    fm = flow_matching_mse(
        clean=[0.8, -0.2], noise=[-0.4, 0.5], prediction=[-1.0, 0.65], tau=0.4
    )
    total = ce + 10.0 * fm
    print(f"toy loss: CE={ce:.4f}, flow_MSE={fm:.4f}, CE+10*flow_MSE={total:.4f}")


if __name__ == "__main__":
    run_episode()
