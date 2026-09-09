# 延伸与改进

- 用 CUB radix sort、scan 和缓存分配器替代频繁的 Thrust 临时分配。
- 为 two-pass 查询真正实现 count/scan/fill，并与单遍预分配方案做端到端 A/B。
- 在 `StacklessLite` 自查询中加入严格的唯一对规则，避免依赖验证阶段集合去重。
- 用更适合 warp coherence 的 BFS-threaded 或 query-aware 排序代替纯 DFS pre-order。
- 为 `N=0/1`、场景某轴长度为零、结果容量溢出和极深树添加明确契约与测试。
- 检查 refit 的跨线程内存顺序：发布 child bound 应先发生，再发布到达标志。
- 把构建临时数组与稳定拓扑数据分离，使 refit 和重复查询的常驻内存更清晰。
