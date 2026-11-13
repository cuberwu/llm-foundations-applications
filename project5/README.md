# 实验五 QLoRA 与 DPO

报告在 `report/实验作业五_实验报告.docx`；实现为 `code/hw5_dpo.py`。

```powershell
pip install -r code/requirements.txt
python code/hw5_dpo.py --mode all
python code/hw5_dpo.py --mode dpo --extra
python code/verify_experiment.py
python code/make_figures.py
python code/hw5_dpo.py --dpo_check=1.2,0.8,0.9,0.6,0.1
```

运行目录为 `project5`。也可分别使用 `--mode make_data/infer/sft/dpo/eval`；SFT默认60步、DPO默认40步，均可用`--steps`覆盖。`--extra`将12组偏好对扩为17组，DPO始终从保存的SFT适配器开始。

本次实际使用Python 3.10.19、torch 2.11.0.dev20260127+cu128、RTX 5070 Ti Laptop 12GB、bitsandbytes 0.50.2；NF4双重量化和fp32计算，种子42，预处理最长边512。其余版本见 `results/environment.json`。请先安装适配设备的CUDA torch；本地模型缓存不入库，无缓存时按固定模型版本从Hugging Face加载。

`results/`保存实测逐步日志、适配器、图与逐题结果；`comparison.csv`同时保留修正判分与指南原子串判分。修正规则识别中英文并避免否定子串误判，不能识别的回答不计对。基线、SFT、DPO、17对DPO依次为6、11、9、9题正确（均为15题）。训练与评测复用指南固定题集，不代表独立测试泛化。

`run.log`与`extra_run.log`为训练时的原始输出；修正英文数字与无关否定句判分后，最终逐题评测以`eval.log`、`comparison.csv`和`summary.json`为准。`verification.log`记录公式、梯度与参考模型检查。

已修正指南框架中的手算loss符号、步数循环、保存路径、DPO重复嵌套适配器及评测否定词问题。验证脚本核对答案掩码、全部DPO日志公式、梯度符号、参考模型一致性及新增5题输出。报告生成使用Codex bundled Python运行 `code/make_report.py`，不修改原样板。
