# Project 8 自制语言模型知识蒸馏

在 `project8/code` 执行，全部使用 CPU，无需下载语料或模型：

```powershell
python distill_lm.py --mode quick
python distill_lm.py --mode teacher --retrain
python distill_lm.py --mode all
python distill_lm.py --mode ablate
python distill_lm.py --mode scan
python verify_experiment.py
python make_figures.py
```

默认教师 200 步、学生 120 步；消融 16 组，各 80 步。教师规模扫描使用 96、160、256 三种宽度。训练与评估均固定种子，学生初始化及批次顺序有 SHA-256 校验。

```text
project8/
├── code/                 模型、实验、自检和绘图脚本
├── docs/                 实验指南与空白模板
├── report/               最终实验报告
└── results/              各组实测 JSON
    ├── checkpoints/      正式模型权重与教师缓存
    ├── figures/          训练曲线与消融热力图
    ├── logs/             运行日志与检查结果
    └── quick/            自检指标
```

实际环境为 Python 3.10.19、PyTorch 2.11.0.dev20260127+cu128、matplotlib 3.10.8，4 个 CPU 线程，未调用 GPU。自检权重可重新生成，不纳入提交。
