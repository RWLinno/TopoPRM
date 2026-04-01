# DAG Verifiability Upgrade Plan (Draft for Confirmation)

## 背景与问题
当前 DAG 节点的 `local_verdict` 默认为 `unverifiable`，定义见 `src/dag/node.py`，构建逻辑在 `src/data/build_dag.py`。这并不意味着 TopoPRM 不能用于 RLVR：
- 现有 Topo/Continuity 奖励本身是程序可计算的（结构与引用关系可验证）。
- 但步骤级正确性信号缺失，导致“可验证粒度”偏粗，难以进一步提升奖励质量。

## 目标
在不破坏现有训练链路的前提下，提升 `local_verdict` 的可判定率，将 `unverifiable` 比例从“几乎全量”下降到“可控比例”，并将其作为可选辅助信号接入奖励。

## 最小可行方案（推荐）

### Phase A: 增强本地可验证器（低风险）
在 `src/data/build_dag.py` 增加轻量规则校验：
1. **Arithmetic Check**：可解析算术等式（如 `2+3=5`）直接判 `correct/incorrect`。
2. **Substitution Consistency**：对“代入”步骤检查左右式在上一步变量赋值下是否一致。
3. **Equation Transform Check (limited)**：单步等价变形（移项/同除）做保守验证。

未命中规则时保持 `unverifiable`，确保零回归风险。

### Phase B: DAG 序列一致性校验（中风险）
新增 `consistency_check`：若某节点被后续节点引用但内部表达自相矛盾，则降权或标记 `incorrect`。

### Phase C: 奖励轻接入（可开关）
在 `src/reward/topo_reward.py` 增加可选项：
- `verdict_bonus = gamma * correct_ratio - lambda * incorrect_ratio`
- 默认关闭（`gamma=lambda=0`），通过配置开启，便于 ablation。

## 数据结构建议（向后兼容）
保持 `local_verdict` 三值不变：`correct | incorrect | unverifiable`。
新增可选字段：
- `verdict_source`: `symbolic_check | consistency_check | none`
- `verdict_confidence`: `[0,1]`

旧数据无此字段时默认回退，不影响现有流程。

## 代码改动面
- `src/data/build_dag.py`：新增 step-level verifier 与字段写入。
- `src/dag/node.py`：扩展 Node 可选字段（保持反序列化兼容）。
- `src/reward/topo_reward.py`：新增可选 verdict bonus。
- `tests/`：补充 verifier 单测与回归测试。

## 验收标准
1. `parsed.dag.jsonl` 中 `local_verdict != unverifiable` 占比显著提升。
2. 在小样本上 verifier 精度可人工 spot-check。
3. 开关关闭时，训练结果与现有版本数值一致（回归）。
4. 开关开启时，能在结构指标上观察到增益趋势。
