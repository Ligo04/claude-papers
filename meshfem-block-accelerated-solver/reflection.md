# 延伸思考

## 如果继续扩展

- 将块 supernodal 路径扩展到 GPU，比较小块 kernel、稀疏不规则访问和因子同步的实际收益。
- 对接触活动集做增量消去树更新，而不是在硬过期时整体重建。
- 结合自动微分，让新元素能量自动生成结构化局部 Hessian，并保留块边界。
- 为 `d=2/3`、混合材料和高阶元素分别设计 kernel，避免“一套块大小”覆盖所有场景。
- 建立统一 profiling schema，把模式、装配、symbolic、numeric、solve 和 globalization 的时间写入同一日志。

## 脆弱假设

- 局部 Hessian 足够准确且模式大部分时间稳定；
- 经过投影或 shift 后系统适合实数 Cholesky；
- 线程局部矩阵的复制成本不会吞掉并行收益；
- 接触活动集变化率与缓存预算匹配；
- 仿真/几何工作负载具有足够多的元素，能摊薄 symbolic 和调度开销。

## 可能失败的场景

- `d` 很大或块大小高度不均匀，专用小块 kernel 失去优势；
- 模式每次迭代都剧烈变化，LRU 只能不断淘汰；
- Hessian 长时间强烈不定，Cholesky 前的修正成为主导成本；
- 问题规模太小，线程和块调度开销大于计算；
- 需要复数或真正 indefinite 的直接解法，而当前 BlockCatamari 路径只覆盖实数 supernodal Cholesky。
