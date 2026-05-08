# 核心技术验证与测试说明

## 测试目标

围绕两条主贡献提供最小可复现验证：

1. deterministic PRM（结构奖励链路）
2. reverse-KL distillation（损失与过滤逻辑）
3. 宣传报告数据生成链路（eval metrics -> publicity summary）

## 关键测试文件

- 历史测试：
  - `tests/test_build_dag.py`
  - `tests/test_graph.py`
  - `tests/test_rewards.py`
- 新增测试：
  - `tests/test_reverse_kl_loss.py`
  - `tests/test_distill_filter.py`
  - `tests/test_prm_model.py`
  - `tests/test_publicity_report.py`

## 运行方式

```bash
cd /mnt/users/rwl/topoprm
pytest -q tests/test_reverse_kl_loss.py tests/test_distill_filter.py tests/test_prm_model.py tests/test_publicity_report.py
```

如需全量：

```bash
pytest -q tests
```

## 生成演示数据

```bash
python3 scripts/generate_publicity_report.py --eval_dir output/eval --output_dir docs/publicity/data
```

然后打开 `docs/publicity/index.html`。
