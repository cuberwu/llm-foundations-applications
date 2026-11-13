import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path(__file__).resolve().parents[1] / 'results'
figs = out / 'figures'
figs.mkdir(exist_ok=True)
r = json.loads((out / 'results_all.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(6.4, 2.8), layout='constrained')
for key, label in [('teacher', 'Teacher'), ('scratch', 'Student scratch'), ('kd', 'Student KD')]:
    curve = r[key]['curve']
    ax.plot([v['step'] for v in curve], [v['ce'] for v in curve], label=label, lw=1.4)
ax.set(xlabel='Training step', ylabel='Hard-label cross entropy')
ax.grid(alpha=.2)
ax.legend()
fig.savefig(figs / 'curves.png', dpi=220)
plt.close(fig)

ab = json.loads((out / 'results_ablate.json').read_text(encoding='utf-8'))['ablation']
matrix = np.array([[next(v['val_loss'] for v in ab if v['T'] == t and v['alpha'] == a)
                    for a in (.3, .5, .7, .9)] for t in (1, 2, 4, 8)])
fig, ax = plt.subplots(figsize=(6.4, 2.9), layout='constrained')
im = ax.imshow(matrix, cmap='YlGnBu_r', aspect='auto')
ax.set_xticks(range(4), ['0.3', '0.5', '0.7', '0.9'])
ax.set_yticks(range(4), ['1', '2', '4', '8'])
ax.set(xlabel='alpha', ylabel='Temperature T')
best = np.unravel_index(matrix.argmin(), matrix.shape)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{matrix[i,j]:.4f}' + (' *' if (i,j) == best else ''),
                ha='center', va='center', color='white' if matrix[i,j] < 1.73 else 'black')
fig.colorbar(im, ax=ax, label='Validation loss (lower is better)')
fig.savefig(figs / 'ablation_heatmap.png', dpi=220)
plt.close(fig)
print('Saved curves.png and ablation_heatmap.png')
