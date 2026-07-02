# ReActor Code Demos

This directory contains educational code demonstrations of key concepts from the ReActor paper.

## Available Demos

### 1. Bilevel Optimization (`bilevel_optimization.py`)

Demonstrates the core bilevel optimization framework:
- Upper level: adapts retargeting parameters
- Lower level: trains tracking policy
- Simplified gradient estimate

**Run:**
```bash
python bilevel_optimization.py
```

**Requirements:**
- numpy
- matplotlib (optional, for visualization)

### 2. Sparse Parameterization (`sparse_parameterization.py`)

Shows why constant per-joint offsets generalize better than per-frame offsets:
- Compares sparse vs dense parameterization
- Evaluates generalization to new motions
- Demonstrates parameter efficiency

**Run:**
```bash
python sparse_parameterization.py
```

**Requirements:**
- numpy

## Installation

```bash
pip install numpy matplotlib
```

## Notes

These are simplified educational implementations to help understand the concepts. For production use, refer to the paper's official implementation (if available) or implement using proper physics simulation frameworks like Isaac Gym.

## Learning Path

1. Start with `sparse_parameterization.py` to understand the parameterization choice
2. Then run `bilevel_optimization.py` to see the full optimization loop
3. Read the paper alongside the code to connect theory with implementation

## Key Takeaways

- **Bilevel optimization** jointly optimizes reference motion and tracking policy
- **Sparse parameterization** uses constant offsets for better generalization
- **Simplified gradient** avoids expensive inverse Hessian computation
- **Physics-in-loop** ensures dynamically feasible motions
