"""由实测日志生成报告所需的参数输出图与训练曲线。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parents[1]/'results'
def read(n): return json.loads((R/n).read_text(encoding='utf-8'))

fig, ax = plt.subplots(1, 3, figsize=(10.3, 2.8), layout='constrained')
sft = read('sft_training.json')['logs']
ax[0].plot([r['step'] for r in sft], [r['loss'] for r in sft], color='#234f80', linewidth=1)
ax[0].set_title('SFT loss')
for tag, label, color in [('dpo', '12 pairs', '#234f80'), ('dpo17', '17 pairs', '#b45d28')]:
    logs = read(tag+'_training.json')['logs']
    for a, key in [(ax[1], 'loss'), (ax[2], 'margin')]:
        a.plot([r['step'] for r in logs], [r[key] for r in logs], label=label, color=color, linewidth=1)
ax[1].set_title('DPO loss'); ax[2].set_title('DPO margin')
ax[2].axhline(0, color='gray', linestyle='--', linewidth=.6)
for a in ax:
    a.set_xlabel('Step'); a.grid(alpha=.2); a.spines[['top', 'right']].set_visible(False)
ax[2].legend(fontsize=8)
fig.savefig(R/'training_curves.png', dpi=220)
plt.close(fig)

im = Image.new('RGB', (1420, 144), 'white')
draw = ImageDraw.Draw(im)
font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 25)
for i, line in enumerate((R/'parameters.txt').read_text(encoding='utf-8').splitlines()):
    draw.text((12, 20+52*i), line, fill='black', font=font)
im.save(R/'parameters.png')
