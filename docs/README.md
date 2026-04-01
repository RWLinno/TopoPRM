# TopoPRM 文档导航（2026 重构版）

本目录已按“研究叙事 + 工程实现 + 对外传播”重组，核心围绕两条主贡献：

1. **Deterministic Verifiable Process Reward Model（可验证过程奖励）**
2. **Reverse-KL Reasoning Distillation（反向 KL 推理蒸馏）**

> 术语约定：本文档中的“可验证”指**奖励计算可复现、可审计、可程序化检查**，不等同于语义完备证明验证。

---

## 1. 快速入口

- 训练与复现实操：`docs/training_pipeline.md`
- 奖励方法设计：`docs/reward_design.md`
- 蒸馏方法设计：`docs/distillation_reverse_kl.md`
- 进展与执行状态：`docs/progress.md`
- 阶段总结：`docs/work_summary.md`

---

## 2. 对外宣传与可视化

- 宣传总览页面：`docs/publicity/demo_page.html`
- 框架可视化：`docs/publicity/framework_visualization.html`
- 技术路线可视化：`docs/publicity/technical_roadmap.html`
- 说明文档：`docs/publicity/README.md`

---

## 3. 工作报告与研究沉淀

- 报告总览：`docs/reports/README.md`
- Proposal（重构版）：`docs/reports/proposal_reframed.md`
- 更新日志（结构化）：`docs/reports/update_log.md`
- 难点与解决：`docs/reports/challenges_and_solutions.md`
- 论文思考：`docs/reports/paper_thinking.md`

---

## 4. 测试与验证

- 测试说明：`docs/testing.md`
- 测试代码目录：`tests/`
  - 新增：
    - `tests/test_reverse_kl_loss.py`
    - `tests/test_distill_filter.py`
    - `tests/test_prm_model.py`

---

## 5. 历史文档

`dag_schema.md`、`dag_pipeline.md`、`ms_swift_custom_reward.md` 等历史文档保留，用于追溯实现细节与迭代背景。
