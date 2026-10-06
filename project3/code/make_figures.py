"""绘制训练与采样对照图，读取已经保存的实验结果。"""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results"
FIG=RESULTS/"figures"
FIG.mkdir(parents=True,exist_ok=True)
sampling=json.loads((RESULTS/"sampling/sampling.json").read_text(encoding="utf-8"))
def losses(run_id):
    with (RESULTS / "training" / run_id / "losses.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [float(r["loss"]) for r in rows]


def plot_compare(name, runs, labels):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for run_id, label in zip(runs, labels):
        ys = losses(run_id)[:1000]
        ax.plot(range(1, len(ys)+1), ys, linewidth=0.9, alpha=0.85, label=label)
    ax.set(xlabel="Iteration", ylabel="Training cross-entropy", title=name)
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=180); plt.close(fig)
    return path


charts = [plot_compare("exp1_learning_rate", ["baseline", "exp1_lr_high", "exp1_lr_low"], ["lr=0.001", "lr=0.01", "lr=0.0001"]),
          plot_compare("exp2_layers", ["baseline", "exp2_layer1", "exp2_layer4"], ["L=2", "L=1", "L=4"]),
          plot_compare("exp3_width", ["baseline", "exp3_emb64", "exp3_emb256"], ["C=128, H=4", "C=64, H=2", "C=256, H=8"]),
          plot_compare("exp4_context", ["baseline", "exp4_context32"], ["T=128", "T=32"]),
          plot_compare("exp5_position", ["baseline", "exp5_no_pos"], ["learned positions", "no positions"])]


def text_panel(path, blocks, width=1500):
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 26)
    titlefont = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 28)
    lines = []
    for title, text in blocks:
        lines.append((title, True))
        for source_line in text.splitlines():
            current = ""
            for ch in source_line:
                if font.getlength(current+ch) > width - 90:
                    lines.append((current, False)); current = ch
                else:
                    current += ch
            lines.append((current, False))
        lines.append(("", False))
    im = Image.new("RGB", (width, 55+len(lines)*40), "white")
    draw = ImageDraw.Draw(im)
    for i, (line, title) in enumerate(lines):
        draw.text((40, 25+i*40), line, fill="#202a35", font=titlefont if title else font)
    im.save(path)


text_panel(FIG / "sampling_originals.png", [
    (f"组{i+1}  temperature={g['temperature']}  top-k={g['top_k'] or '不限'}  seed=101",
     g["samples"][0]["text"][:60]) for i,g in enumerate(sampling[:5])])
text_panel(FIG / "parameter_output.png", [
    ("实际运行输出摘录（源文件 results/logs/run.log）", "\n".join(
        (RESULTS / "logs/run.log").read_text(encoding="utf-8-sig").splitlines()[:4]))])
fig, ax = plt.subplots(figsize=(7, 3.2))
groups = sampling[:5]
xs = list(range(5))
ax.bar([x-0.18 for x in xs], [g["repeat3"] for g in groups], width=.36, label="Repeated trigrams")
ax.bar([x+0.18 for x in xs], [g["corpus_match10"] for g in groups], width=.36, label="Corpus-matched 10-grams")
ax.set_xticks(xs, ["1.0 / 20", "0.5 / 20", "1.5 / 20", "1.0 / 5", "1.0 / all"])
ax.set(xlabel="Temperature / top-k", ylabel="Ratio", ylim=(0,1))
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "sampling_metrics.png", dpi=180); plt.close(fig)

