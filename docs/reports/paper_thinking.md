# 对论文写作的思考（Reviewer 风险视角）

## 核心策略

1. 把 claim 收紧到“可辩护”范围。
2. 始终区分结构可验证与语义正确性。
3. 蒸馏必须与 PRM 并列，不再是附录式叙事。

## 高风险表述（应避免）

- logical soundness guarantee
- proof verification
- step correctness verification
- true DAG / ground-truth dependency graph（无标注支撑时）

## 推荐表达

- verifiable structural/process rewards
- deterministic and auditable reward computation
- dependency consistency and traceability
- extracted dependency DAG as structural proxy

## 实验叙事建议

- 主指标：accuracy + efficiency
- 过程指标：拓扑与连续性指标（明确其 proxy 属性）
- 蒸馏指标：teacher-student retention + compression ratio
- 局限性必须显式写：extractor fidelity、mode collapse 风险
