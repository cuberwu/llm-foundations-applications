"""核对影响实验结论的损失实现、因果掩码和公平对照。"""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from distill_lm import GPT, kd_loss, make_data

torch.set_num_threads(4)
torch.manual_seed(21)
student = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
teacher = torch.randn_like(student)
T = 4
p = F.softmax(teacher / T, -1)
q = F.softmax(student / T, -1)
manual = (p * (p.log() - q.log())).sum(-1).mean() * T ** 2
actual = kd_loss(student, teacher, T)
assert torch.allclose(actual, manual, atol=1e-12)
gradient = torch.autograd.grad(actual, student)[0]
assert torch.allclose(gradient, T * (q - p) / 6, atol=1e-12)
assert torch.allclose(kd_loss(student, teacher, T), kd_loss(student.repeat(1, 2, 1), teacher.repeat(1, 2, 1), T))
print('PASS: KL direction, token normalization, T^2 gradient scaling')

chars, tr, va = make_data()
assert (len(chars), len(tr), len(va)) == (322, 49457, 5357)
model = GPT(len(chars), 64).eval()
x = torch.randint(322, (2, 20))
y = x.clone()
y[:, 10:] = torch.randint(322, (2, 10))
with torch.no_grad():
    assert torch.equal(model(x)[:, :10], model(y)[:, :10])
assert all(p.device.type == 'cpu' for p in model.parameters())
for c in (64, 96, 160, 256):
    exact = 2 * 322 * c + 96 * c + 24 * c ** 2 + 28 * c
    assert sum(p.numel() for p in GPT(322, c).parameters()) == exact
print('PASS: corpus counts, causal mask, CPU device, exact parameter formula')

out = Path(__file__).resolve().parents[1] / 'results'
r = json.loads((out / 'results_all.json').read_text(encoding='utf-8'))
for key in ('init_sha256', 'batches_sha256'):
    assert r['scratch'][key] == r['kd'][key]
assert r['teacher']['params'] / r['kd']['params'] == 5
ab = json.loads((out / 'results_ablate.json').read_text(encoding='utf-8'))['ablation']
assert len(ab) == 16 and len({(row['T'], row['alpha']) for row in ab}) == 16
assert len({row['init_sha256'] for row in ab}) == 1
assert len({row['batches_sha256'] for row in ab}) == 1
scan = json.loads((out / 'results_scan.json').read_text(encoding='utf-8'))['scan']
for row in scan:
    for key in ('init_sha256', 'batches_sha256'):
        assert row['student'][key] == r['kd'][key]
assert scan[1]['student']['val_loss'] == r['kd']['val_loss']
print('PASS: main comparison, 16-cell ablation fairness, teacher scan fairness and deterministic rerun')
