# Reverse-KL Reasoning Distillation

## 1. 目标

将 teacher 中“graph-enriched reasoning behavior”迁移到更小 student，形成更紧凑的 chain-like reasoning。

---

## 2. 两阶段思路

1. Teacher 阶段：通过稀疏但可验证奖励（outcome + deterministic process rewards）训练
2. Student 阶段：通过 process-aware 过滤后，用 reverse-KL 蒸馏

---

## 3. 数学定义

- teacher policy: `π_T(y|x)`
- student policy: `π_S(y|x)`

目标函数：

`L_RKL = E_x [ KL( π_S(.|x) || π_T(.|x) ) ]`

reverse KL 的 mode-seeking 性质使 student 倾向聚焦 teacher 高质量模式，而非平均化所有可能输出。

---

## 4. 为什么需要 process-aware filtering

若不过滤 teacher 样本，student 会学习到噪声轨迹和冗余推理。

过滤标准可包括：

- 最终答案有效性
- topology score 阈值
- continuity score 阈值
- 长度/步骤上限

---

## 5. 工程实现

- 过滤：`src/distill/teacher_trace_filter.py`
- 损失：`src/distill/reverse_kl_loss.py`
- 指标：`src/distill/chain_compression_metrics.py`
- 训练骨架：`src/distill/student_train.py`

---

## 6. 评估建议

- Teacher/Student accuracy retention
- 推理长度压缩率
- 结构指标保真（acyclicity / dependency depth proxy）
