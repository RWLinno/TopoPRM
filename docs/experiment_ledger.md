# 实验台账（开始-结束-命令-日志）

> 自动生成时间：2026-03-24 13:32:02
> 结束时间规则：运行中记为 `RUNNING`；已结束使用日志最后时间戳，若缺失则用日志修改时间。

| 开始时间 | 结束时间 | 启动命令 | 结果日志路径 |
|---|---|---|---|
| 2026-03-23 17:37:20 | 2026-03-23 17:37:20 | `bash scripts/run_eval_light_private.sh (light-200 private eval)` | `/mnt/users/rwl/topoprm/output/eval/baseline_qwen3_32b_instruct_light200.nohup.log` |
| 2026-03-23 17:37:57 | 2026-03-23 17:37:57 | `bash scripts/run_eval_light_private.sh (light-200 private eval)` | `/mnt/users/rwl/topoprm/output/eval/grpo_no_topo_light200.nohup.log` |
| 2026-03-23 17:49:17 | 2026-03-23 18:20:10 | `bash scripts/run_eval_light_private.sh (light-200 private eval)` | `/mnt/users/rwl/topoprm/output/eval/grpo_no_topo_light200.topoprm.nohup.log` |
| 2026-03-23 17:51:49 | 2026-03-23 17:57:53 | `bash scripts/run_eval_light_private.sh (light-200 private eval)` | `/mnt/users/rwl/topoprm/output/eval/baseline_7b8b_light200.topoprm.nohup.log` |
| 2026-03-23 18:01:31 | 2026-03-23 18:07:56 | `bash scripts/run_eval_light_private.sh (light-200 private eval)` | `/mnt/users/rwl/topoprm/output/eval/baseline_7b8b_light200_resume.topoprm.nohup.log` |
| 2026-03-23 18:13:53 | 2026-03-23 18:27:50 | `bash scripts/run_eval_light_private.sh (light-200 private eval)` | `/mnt/users/rwl/topoprm/output/eval/baseline_llama31_8b_light200_resume.topoprm.nohup.log` |
| 2026-03-23 18:34:50 | 2026-03-23 18:34:50 | `TBD(待补: 从终端历史恢复精确命令)` | `/mnt/users/rwl/topoprm/logs/benchmark_light_topoprm.log` |
| 2026-03-23 20:11:56 | RUNNING | `python scripts/run_benchmark_light.py` | `/mnt/users/rwl/topoprm/logs/benchmark_light_topoprm_gpu2.log` |
| 2026-03-23 20:13:18 | 2026-03-23 21:17:41 | `bash scripts/run_eval_light_private.sh (light-200 private eval)` | `/mnt/users/rwl/topoprm/output/eval/grpo_no_topo_light200.gpu3.topoprm.nohup.log` |
| 2026-03-23 20:23:32 | 2026-03-23 20:55:44 | `bash scripts/run_eval_light_private.sh (light-200 private eval)` | `/mnt/users/rwl/topoprm/output/eval/baseline_qwen3_32b_light200.gpu5.topoprm.nohup.log` |
| 2026-03-23 20:23:52 | 2026-03-23 21:27:36 | `bash scripts/run_eval_light_private.sh (light-200 private eval)` | `/mnt/users/rwl/topoprm/output/eval/sft_light200.gpu4.topoprm.nohup.log` |
| 2026-03-23 20:30:26 | 2026-03-23 21:43:28 | `swift eval Qwen3-32B(+outcome-only) gsm8k --eval_limit 200` | `/mnt/users/rwl/topoprm/logs/benchmark_grpo_outcome_gsm8k_gpu7.log` |
| 2026-03-23 20:30:28 | 2026-03-23 21:41:47 | `swift eval Qwen3-32B(+TopoPRM) gsm8k --eval_limit 200` | `/mnt/users/rwl/topoprm/logs/benchmark_grpo_main_gsm8k_gpu6.log` |
| 2026-03-23 20:37:03 | 2026-03-23 20:51:07 | `bash scripts/run_eval_light_private.sh (light-200 private eval)` | `/mnt/users/rwl/topoprm/output/eval/baseline_qwen25_7b_light200_gpu7.topoprm.nohup.log` |
| 2026-03-23 21:04:28 | 2026-03-23 21:21:00 | `swift eval Qwen2.5-7B gsm8k --eval_limit 200` | `/mnt/users/rwl/topoprm/logs/benchmark_baseline_qwen25_gsm8k_gpu5.log` |
| 2026-03-23 21:04:28 | 2026-03-23 21:14:07 | `swift eval Llama-3.1-8B gsm8k --eval_limit 200` | `/mnt/users/rwl/topoprm/logs/benchmark_baseline_llama31_gsm8k_gpu7.log` |
| 2026-03-23 21:14:11 | 2026-03-23 22:32:42 | `swift eval Qwen2.5-7B math_500 --eval_limit 200` | `/mnt/users/rwl/topoprm/logs/benchmark_baseline_qwen25_math500_gpu7.log` |
| 2026-03-23 21:14:11 | 2026-03-23 21:36:09 | `swift eval Llama-3.1-8B math_500 --eval_limit 200` | `/mnt/users/rwl/topoprm/logs/benchmark_baseline_llama31_math500_gpu5.log` |
| 2026-03-24 00:04:33 | 2026-03-24 02:28:27 | `swift eval Llama-3.1-8B (gsm8k -> math_500, ports 8127/8128)` | `/mnt/users/rwl/topoprm/logs/benchmark_llama31_gpu7_retry2.log` |

| 2026-03-24 00:12:11 | RUNNING | `bash scripts/run_eval_light_private.sh Qwen/Qwen3-32B output/grpo_no_continuity/.../checkpoint-450 grpo_no_continuity_light200` | `/mnt/users/rwl/topoprm/output/eval/grpo_no_continuity_light200.gpu3.topoprm.nohup.log` |
| 2026-03-24 00:19:44 | RUNNING | `bash scripts/run_grpo.sh grpo_clipped` | `/mnt/users/rwl/topoprm/output/grpo_clipped_topoprm_retry3_20260324_001943.log` |
| 2026-03-24 00:35:02 | RUNNING | `swift eval Qwen/Qwen3-32B + SFT (gsm8k -> math_500, ports 8134/8137)` | `/mnt/users/rwl/topoprm/logs/benchmark_sft_gpu4.log` |
| 2026-03-24 00:35:03 | RUNNING | `swift eval Qwen/Qwen3-32B + TopoPRM math_500 (port 8135)` | `/mnt/users/rwl/topoprm/logs/benchmark_grpo_main_math500_gpu5.log` |
| 2026-03-24 00:35:03 | RUNNING | `swift eval Qwen/Qwen3-32B + outcome-only math_500 (port 8136)` | `/mnt/users/rwl/topoprm/logs/benchmark_grpo_outcome_math500_gpu6.log` |
| 2026-03-24 00:01:38 | FAILED 2026-03-24 00:16:19 | `bash scripts/run_grpo.sh grpo_mulgate` | `/mnt/users/rwl/topoprm/output/grpo_mulgate_topoprm_20260324_000137.log (已删除)` |
| 2026-03-24 00:12:04 | FAILED 2026-03-24 00:24:10 | `bash scripts/run_grpo.sh grpo_scae` | `/mnt/users/rwl/topoprm/output/grpo_scae_topoprm_20260324_001202.log (已删除)` |
| 2026-03-24 01:23:41 | 2026-03-24 02:41:08 | `swift eval Llama-3.1-8B-Instruct math_500 (port 8133)` | `/mnt/users/rwl/topoprm/logs/benchmark_llama31_math500_gpu3_retry3.log` |
| 2026-03-24 08:48:53 | RUNNING | `bash scripts/run_grpo.sh grpo_confgate` | `/mnt/users/rwl/topoprm/output/grpo_confgate_topoprm_retry2_20260324_084852.log` |
| 2026-03-24 08:48:54 | RUNNING | `bash scripts/run_eval_light_private.sh Qwen/Qwen3-32B output/grpo_clipped/v2-20260324-002031/checkpoint-318 grpo_clipped_light200` | `/mnt/users/rwl/topoprm/output/eval/grpo_clipped_light200.gpu3.topoprm.nohup.log` |
| 2026-03-24 11:29:14 | RUNNING | `swift eval Qwen/Qwen3-32B + grpo_clipped checkpoint-318 gsm8k (port 8140)` | `/mnt/users/rwl/topoprm/logs/benchmark_grpo_clipped_gsm8k_gpu0.log` |
| 2026-03-24 11:29:15 | RUNNING | `swift eval Qwen/Qwen3-32B + grpo_clipped checkpoint-318 math_500 (port 8141)` | `/mnt/users/rwl/topoprm/logs/benchmark_grpo_clipped_math500_gpu1.log` |
| 2026-03-24 11:29:16 | RUNNING | `bash scripts/run_eval_light_private.sh Qwen/Qwen3-32B output/grpo_clipped/v2-20260324-002031/checkpoint-318 grpo_clipped_light200` | `/mnt/users/rwl/topoprm/output/eval/grpo_clipped_light200.gpu3.topoprm.nohup.log` |
| 2026-03-24 13:30:18 | RUNNING | `swift eval Qwen/Qwen3-32B + grpo_no_topo checkpoint-637 gsm8k (port 8142)` | `/mnt/users/rwl/topoprm/logs/benchmark_grpo_no_topo_gsm8k_gpu0.log` |
| 2026-03-24 13:30:20 | RUNNING | `swift eval Qwen/Qwen3-32B + grpo_no_topo checkpoint-637 math_500 (port 8143)` | `/mnt/users/rwl/topoprm/logs/benchmark_grpo_no_topo_math500_gpu3.log` |
| 2026-03-24 16:46:51 | FAILED 2026-03-24 16:48:10 | `swift eval Qwen/Qwen3-32B + grpo_no_topo checkpoint-637 math_500 (port 8150)` | `/mnt/users/rwl/topoprm/logs/benchmark_grpo_no_topo_math500_gpu0_retry.log` |
| 2026-03-24 16:48:10 | FAILED 2026-03-25 13:04:52 | `swift eval Qwen/Qwen3-32B + grpo_no_topo checkpoint-637 math_500 (port 8151, eval_num_proc=1, max_new_tokens=1536)` | `/mnt/users/rwl/topoprm/logs/benchmark_grpo_no_topo_math500_gpu0_retry2.log` |
| 2026-03-25 13:04:52 | BLOCKED | `GPU0 diagnostics: residual VRAM owner pid=2136374 (not found in ps), attempted kill/reset unsupported` | `/mnt/users/rwl/topoprm/logs/benchmark_grpo_no_topo_math500_gpu0_retry2.log` |
