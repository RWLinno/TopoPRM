# Tutorials

本目录包含 TopoPRM 项目的可视化工具和复现指南。

## 文件说明

| 文件 | 用途 |
|------|------|
| `training_curve.py` | 绘制训练曲线（loss、reward、pass@1） |
| `efficiency_barplot.py` | 生成效率对比柱状图 |
| `render_dag_cases.py` | DAG 案例可视化（从 rollout 或 eval 结果渲染） |
| `dag_coverage_audit.py` | DAG 覆盖率审计（检查 DAG 提取质量） |
| `reproduce_experiments.md` | 完整实验复现指南 |

## 快速使用

```bash
# 绘制训练曲线
python3 tutorials/training_curve.py

# 生成效率对比图
python3 tutorials/efficiency_barplot.py

# 渲染 DAG 案例
python3 tutorials/render_dag_cases.py --from-rollout output/eval/

# DAG 覆盖率审计
python3 tutorials/dag_coverage_audit.py
```
