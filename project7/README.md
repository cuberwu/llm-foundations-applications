# 作业七 白模转写实

报告：`report/实验作业七_实验报告.docx`。图像、完整命令、原始日志和计时记录位于 `output/`。

在 `project7` 目录执行，需 Python 与 Pillow：

```powershell
python -m pip install -r requirements.txt
powershell -File scripts/setup.ps1
python scripts/run_pipeline.py --name A --steps 4
python scripts/run_pipeline.py --name B --steps 8
python scripts/run_pipeline.py --name C --steps 4 --control-strength 0.5
```

流水线自动创建白模并导出控制图。默认纯 CPU、16 线程、512×512、CFG 1.2、种子 42、euler_a、discrete。场景文件为 `assets/demo_white_model.blend`。

| 运行 | 步数 | 控制强度 | 采样 s | VAE s | 总耗时 s |
|---|---:|---:|---:|---:|---:|
| A | 4 | 0.9 | 39.89 | 16.05 | 65.72 |
| B | 8 | 0.9 | 79.21 | 13.87 | 109.14 |
| C | 4 | 0.5 | 40.92 | 14.91 | 70.48 |

本机为 i9-13900HX / 32 GB / Windows 11。总耗时包含 Blender 子进程和推理，不含下载与转换。三张控制参考图同时导出，共享渲染耗时，详见各组 `*_render.json`；汇总见 `output/summary.csv`。

重新生成报告：

```powershell
python scripts/build_report.py
```

工具和权重来源：[Blender 4.5](https://download.blender.org/release/Blender4.5/)、[stable-diffusion.cpp 465adf3](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1086-465adf3)、[SD-Turbo Q4_0](https://huggingface.co/samoudi/sd-turbo-gguf)、[SD2.1 depth ControlNet](https://huggingface.co/thibaud/controlnet-sd21)。首次准备时校验原始权重的 SHA256，再将 ControlNet 转为 FP16 GGUF。
