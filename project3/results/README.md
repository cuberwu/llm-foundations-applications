# 手搓最小 LLM——使用 CPU 训练

| 目录或文件 | 用途 |
|---|---|
| `summary.json` | 九次训练的参数量、初始与最终 loss、耗时等汇总 |
| `figures/` | 学习率、层数、宽度、上下文、位置编码和采样对照图 |
| `sampling/generated_all.txt` | 五组正式采样与重复惩罚对比的完整生成文本 |
| `sampling/sampling.json` | 每组温度、top-k、种子和文本重复指标 |
| `training/` | 下表中的九组训练，结果分别保存 |
| `logs/run.log` | 实验实际运行日志 |
| `logs/thread_benchmark.json` | CPU 线程速度对比 |
| `logs/self_check.txt` | 模型自检结果 |
| `logs/qualitative_review.json` | 对生成文本的描述性评定 |

| training 下的目录 | 对应实验 |
|---|---|
| `baseline` | 基线：2000 步、128 维、2 层 |
| `exp1_lr_high` / `exp1_lr_low` | 学习率 0.01 / 0.0001 |
| `exp2_layer1` / `exp2_layer4` | Transformer 层数 1 / 4 |
| `exp3_emb64` / `exp3_emb256` | 嵌入维度 64 / 256，头数 2 / 8 |
| `exp4_context32` | 上下文长度 32 |
| `exp5_no_pos` | 去掉可学习位置编码 |

每组训练中，`generated.txt` 是可直接阅读的生成文本，`loss_curve.png` 是损失曲线，`metrics.json` 是配置和指标，`losses.csv` 是逐步记录，`generated.json` 是结构化生成记录。为减小提交体积，仅基线保留 `model.pt` 及第 1000 步模型和生成结果；其他模型可按对应配置重训生成。
