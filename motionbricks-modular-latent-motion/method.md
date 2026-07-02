# MotionBricks 方法详解

## 系统架构总览

```
用户输入 (手柄/键盘/游戏事件)
         │
         ▼
┌─────────────────────────┐
│   Stage 0: Smart        │  生成目标关键帧 T
│   Primitives            │  (locomotion / object)
└──────────┬──────────────┘
           │ 关键帧约束 T = {T₁, T₂, T₃}
           ▼
┌─────────────────────────┐
│   Stage 1: Root Module  │  预测帧数 T₂ + 根轨迹 {r_g}
│   (50M params)          │
└──────────┬──────────────┘
           │ T₂, {r_g}, T
           ▼
┌─────────────────────────┐
│   Stage 2: Pose Module  │  生成多头 pose token {z_q}
│   (150M params)         │  masked token modeling
└──────────┬──────────────┘
           │ {z_q}, {r_g}, T
           ▼
┌─────────────────────────┐
│   Stage 3: Decoder      │  解码连续运动
│   (23.5M params)        │  {r, p, q, v, c}
└─────────────────────────┘
```

## 运动表示

每帧 t 的状态元组：(r_g, r_l, p, q, v, c)

| 符号 | 含义 | 维度 |
|------|------|------|
| r_g | 全局根值（投影位置 + 航向 cos/sin） | 4 |
| r_l | 局部根值（投影速度 + 角速度） | 4 |
| p | 关节位置（不含根） | J×3 |
| q | 关节旋转 | J×? |
| v | 关节速度 | J×3 |
| c | 接触标签 | 4（脚） |

关键设计：
- **不做 heading canonicalization**：支持爬行等非标准朝向
- 用**随机旋转增强**代替 heading 对齐
- 关节位置 p 在全局坐标系表示

## 组件详解

### 1. 多头结构化 Tokenizer (Section 4)

**编码器**：
```
输入: {p^t, q^t}_{t=1}^T  (不含根!)
  │
  ├── 1D Conv (下采样 2×)
  ├── 1D Conv (下采样 2×)  → 总下采样 4×
  │
  ▼
连续 latent: z_e^t ∈ R^D, t = 1..T/4
  │
  ├── 沿特征维分割为 K 份: z_{e,1}^t, ..., z_{e,K}^t
  │
  ▼
独立量化: z_{q,k}^t = argmin_{e_k ∈ E_k} ||z_{e,k}^t - e_k||²
```

**为什么多头优于单头？**
- 单头 codebook 大 → collapse（大部分 code 未使用）
- 多头：K 个小 codebook（每个 128-256），组合容量 256^K
- 网络自动学习分配：无需手动定义"头1=上半身, 头2=下半身"
- 单 token 错误只损失 1/K 信息 → 鲁棒性

**训练**：标准 VQ-VAE loss + running-mean codebook update
```
L = Σ (||sg(z_{e,k}) - e_k||² + β||z_{e,k} - sg(e_k)||²)
```

**解码器**：对称上采样 (2× → 2×)，skip connection 注入根轨迹和关键帧约束

### 2. 约束系统 (Section 3)

三类约束集：
- **T₁** (local root): 局部根速度/角速度
- **T₂** (global root): 全局位置/朝向
- **T₃** (pose): 关节位置/旋转

```
Context keyframes:  索引 0, 1, 2, 3  (当前状态，始终可用)
Target keyframes:   索引 4, 5, 6, 7  (由 smart primitive 提供，可变)
```

缺失约束 → 可学习 mask embedding 替代（训练时随机采样 0-10 个）

### 3. Root Module (Section 5.1)

两步设计：

**Step 1: 隐状态初始化 + 时序预测**
```
输入:
  - 16 个可学习 frame-slot embedding {h₁^t}
  - 全局关键帧编码 f(T₁, T₂, T₃)
  - 帧数编码 g(T₁) 或可学习 mask

Transformer encoder → 预测帧间隔分布 → T₂
```

**Step 2: 条件根轨迹估计**
```
输入: frame-slot embedding + T₂ 约束
Temporal transformer → 全局根轨迹 {r_g}_{t=1}^{T₂}
```

公式化：
```
{h₂}, T₂ = F₁({h₁}; g(T₁); f(T₁, T₂, T₃))
{r_g} = F₂({h₂}; g(T₂); f(T₁, T₂, T₃))
```

### 4. Pose Module (Section 5.2)

最大的 transformer (150M params)：

```
输入:
  (1) local/global root embedding
  (2) 局部 pose 关键帧约束（来自 T₃）
  (3) pose token embedding（训练: cosine schedule 混合 GT 和 mask;
                            推理: 全部初始化为 mask）

Transformer → 每个 head 独立 linear + softmax → token 分布 p(z_{q,k})
```

采用 **masked token modeling**：单次前向传播即可生成（不需迭代去噪），但支持多步迭代以提升质量。

### 5. Smart Locomotion (Section 6.1)

```
用户命令 (v, θ)
      │
      ▼
┌──────────────────────┐
│ Naive: r_{g,1}       │  线性外推 1s
│ = 当前位置 + v·t     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Spring: r_{g,2}      │  临界阻尼弹簧平滑
│ r(t) = e^{-γt}(...)  │  γ 为阻尼系数
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 风格关键帧放置       │  从动作库选取风格 pose
│ 放到 r_{g,2} 上      │  支持 2D blendspace 插值
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Root Module: r_{g,3}  │  神经网络细化
│ 处理死区 + 风格约束   │  自动调整不合理轨迹
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Decoder: r_{g,4}      │  全身协调细化
│ 脚步 + 速度协调       │  最终运动输出
└──────────────────────┘
```

**临界阻尼弹簧**（Eq. 6）：
```
r(t) = e^{-γt} ((r₀ - r_{g,1}) + (v₀ + γ(r₀ - r_{g,1}))t) + r_{g,1}
```
t=1.0s 时产生平滑目标 r_{g,2}。

### 6. Smart Object (Section 6.2)

两个核心组件：

**意图关键帧 (Intent Keyframes)**：
- 定义交互行为的关键姿态（如攀爬：接触姿势 + 站立姿势）
- 硬约束 (τ=0): 必须精确达到（手部接触）
- 软引导 (τ>0): 允许提前 τ 帧过渡（准备姿势）
- 支持多个关键帧集 → 动作多样性
- 旋转标志 ω：是否绕交互枢轴旋转

**交互绑定 (Interaction Binding)**：
- **检测**: 物体碰撞体积作为触发区域
- **放置**: socket 系统定义物体附着点和释放位置
- **锚定**: 物体世界变换作为关键帧旋转枢轴（同一关键帧适配不同位置/角度的物体）

## 算法 1: 推理循环

```python
# 初始化
B = MotionBuffer()      # 运动缓冲区
S = CharacterState()    # 角色状态

while running:
    C = read_user_commands()
    
    if C.changed or len(B) < threshold:
        # Stage 0: Smart Primitives → 关键帧约束
        T = smart_primitive(S, C)  # {T₁, T₂, T₃}
        
        # Stage 1: Root Module → 时序 + 轨迹
        T2, r_g = root_module(T)
        
        # Stage 2: Pose Module → 离散 token
        z_q = pose_module(r_g, T, T2)
        
        # Stage 3: Decoder → 连续运动
        r, p, q, v, c = decoder(z_q, r_g, T)
        
        B.update(r, p, q, v, c)
    
    frame = B.pop_next()
    S.update(frame)
    render(frame)
```

Replan 频率：推荐每 3-9 帧 + 命令变化时立即 replan。

## 训练细节

| 组件 | 架构 | 参数量 | 训练时间 |
|------|------|--------|---------|
| Tokenizer | U-Net (1D Conv) | 23.5M | 7 天 |
| Root Module | Transformer (12层, 512d) | 50M | 3 天 |
| Pose Module | Transformer (16层, 1024d) | 150M | 7 天 |

- 硬件: 4×8 H100 (32 GPU)，16 GPU 质量相当
- Batch size: 256/GPU
- 优化器: Adam, lr=5×10⁻⁵, 10k warmup, cosine → 2×10⁻⁶
- 运动段长度: 随机采样 T=12~64 帧（增量 4）
- 推理: ONNX → TensorRT, RTX 5090 → 2ms / 15,000 FPS

## 实现陷阱

| 陷阱 | 说明 | 对策 |
|------|------|------|
| Codebook collapse | 大 codebook 中多数 code 未用 | 多头分割 + running-mean update |
| Heading canonicalization | 爬行等动作无明确朝向 | 不做 heading 对齐，用随机旋转增强 |
| 根轨迹训练/推理差异 | 训练用 GT root，推理用预测 | Decoder 对 root 来源不敏感（解耦设计） |
| 关键帧过约束 | 物体交互需密集关键帧 | 硬/软约束混合 + drop-frame 属性 |
| Replan 频率过高 | 离散 latent 难捕捉帧间微变化 | 默认 3-9 帧间隔 + 命令变化触发 |
| Retarget 误差累积 | 训练骨架≠部署骨架 | UE5 内置 retarget + 数据规模弥补 |

## 超参数敏感性

| 参数 | 推荐值 | 影响 |
|------|--------|------|
| Codebook heads K | 5-20 | K↑ 容量↑ 但单 token 信息↓ |
| Tokens per head | 128-256 | 过大 → collapse，过小 → 容量不足 |
| 运动段长度 T | 12-64 帧 | T↑ 上下文更丰富但计算更贵 |
| Replan 间隔 τ | 3-9 帧 | τ↓ 响应快但 jitter 增加 |
| 弹簧阻尼 γ | ~2.0 | γ↑ 响应快但过冲，γ↓ 平滑但迟钝 |
| 关键帧数量 | 0-10 | 过多 → 过约束，过少 → 欠约束 |
