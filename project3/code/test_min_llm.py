"""直接运行：python test_min_llm.py。检查任务最关键的三项约束。"""
import torch
from min_llm import CharTokenizer, MiniGPT, POEMS, get_batch

torch.set_num_threads(1)
torch.manual_seed(42)
tok = CharTokenizer("\n".join(POEMS))
assert tok.vocab_size == 699
assert tok.decode(tok.encode("床前明月光")) == "床前明月光"
model = MiniGPT(tok.vocab_size)
assert sum(p.numel() for p in model.parameters()) == 502656
assert model.head.weight is model.tok_emb.weight
x, y = get_batch(torch.arange(200), 32, 4, torch.Generator().manual_seed(42))
assert torch.equal(x[:, 1:], y[:, :-1]) and torch.equal(x[:, -1] + 1, y[:, -1])
# 只修改未来字符，前四个位置的预测必须保持不变。
a = torch.tensor([tok.encode("床前明月光")])
b = a.clone(); b[0, 4] = tok.stoi["春"]
with torch.no_grad():
    assert torch.allclose(model(a)[0][:, :4], model(b)[0][:, :4], atol=1e-6)
_, loss = model(a, a)
loss.backward()
assert torch.isfinite(loss) and model.tok_emb.weight.grad is not None
generated = model.generate(a, max_new_tokens=3, top_k=1)
assert generated.shape == (1, 8)
assert model.generate(a, max_new_tokens=3, top_k=0, repetition_penalty=1.2).shape == (1, 8)
try:
    tok.encode("🙂")
    raise AssertionError("未知字符未报错")
except KeyError:
    pass
print("PASS: tokenizer, shifted labels, 502656 parameters, tied weights, causal mask, backward, sampling")
