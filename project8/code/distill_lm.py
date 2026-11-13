"""作业八：按指南运行字符级 GPT 的 CPU 知识蒸馏。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SUBJ = ["小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人", "少年", "故人", "归客", "山僧", "孤客", "游子"]
PLACE = ["江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡", "溪桥", "松间", "篱下", "窗前", "陌上", "舟中"]
VERB = ["望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看", "静听", "又逢", "恰遇", "空忆"]
OBJ = ["明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟", "白雪", "寒江", "疏星", "暮云", "秋草", "远帆"]
MOOD = ["心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声", "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别"]
POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
]


def build_corpus(n_lines=3000, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append("%s%s%s%s，%s。" % (s, p, v, o, m))
        elif i % 3 == 1:
            out.append("%s在%s%s了一夜%s，%s。" % (s, p, o, rng.choice(["看", "听", "守", "忆"]), m))
        else:
            out.append("%s%s道：“%s%s。”%s。" % (s, v, o, rng.choice(["正好", "可怜", "可惜", "依旧"]), m))
    out.append("\n")
    return "".join(out + POEMS)


def make_data():
    train, val = build_corpus(), build_corpus(300, 999)
    chars = sorted(set(train))
    stoi = {c: i for i, c in enumerate(chars)}
    assert set(val) <= set(chars)
    return chars, torch.tensor([stoi[c] for c in train]), torch.tensor([stoi[c] for c in val])


class Attention(nn.Module):
    def __init__(self, c, heads=4, block=96):
        super().__init__()
        self.heads = heads
        self.c_attn, self.c_proj = nn.Linear(c, 3 * c), nn.Linear(c, c)
        self.register_buffer("mask", torch.tril(torch.ones(block, block)).view(1, 1, block, block))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = [z.view(b, t, self.heads, c // self.heads).transpose(1, 2)
                   for z in self.c_attn(x).split(c, dim=2)]
        a = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        a = F.softmax(a.masked_fill(self.mask[:, :, :t, :t] == 0, -float("inf")), dim=-1)
        return self.c_proj((a @ v).transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.attn = Attention(c)
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, c):
        super().__init__()
        self.tok_emb, self.pos_emb = nn.Embedding(vocab, c), nn.Embedding(96, c)
        self.blocks = nn.Sequential(Block(c), Block(c))
        self.ln_f, self.head = nn.LayerNorm(c), nn.Linear(c, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb(torch.arange(x.shape[1]))
        return self.head(self.ln_f(self.blocks(h)))


def new_model(vocab, c, seed):
    torch.manual_seed(seed)
    return GPT(vocab, c)


def batch(data, bs, gen):
    ix = torch.randint(len(data) - 97, (bs,), generator=gen)
    return (torch.stack([data[i:i + 96] for i in ix]),
            torch.stack([data[i + 1:i + 97] for i in ix]))


def ce(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def kd_loss(s, t, temperature):
    v = s.shape[-1]
    return F.kl_div(F.log_softmax(s.reshape(-1, v) / temperature, -1),
                    F.softmax(t.reshape(-1, v) / temperature, -1),
                    reduction="batchmean") * temperature ** 2


def model_hash(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def train(model, data, steps, args, teacher=None, T=4, alpha=0.5, cache=None, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed + 1)
    hist, data_hash = [], hashlib.sha256()
    initial_hash = model_hash(model)
    if teacher is not None:
        teacher.eval()
    model.train()
    start = time.perf_counter()
    for it in range(steps):
        if cache is None:
            x, y = batch(data, args.batch_size, gen)
            with torch.no_grad():
                tl = teacher(x) if teacher is not None else None
        else:
            x, y, tl = cache[it % len(cache)]
        data_hash.update(x.numpy().tobytes())
        data_hash.update(y.numpy().tobytes())
        logits = model(x)
        hard = ce(logits, y)
        soft = kd_loss(logits, tl, T) if tl is not None else hard.new_zeros(())
        loss = (1 - alpha) * hard + alpha * soft if tl is not None else hard
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append({"step": it + 1, "ce": hard.item(), "kd_scaled": soft.item(), "loss": loss.item()})
        if it == 0 or (it + 1) % 50 == 0 or it + 1 == steps:
            print(f"[{tag}] {it+1}/{steps} loss={loss.item():.4f} ce={hard.item():.4f}", flush=True)
    return {"steps": steps, "time_s": time.perf_counter() - start, "curve": hist,
            "init_sha256": initial_hash, "batches_sha256": data_hash.hexdigest(),
            "final_train_ce": hist[-1]["ce"], "final_train_loss": hist[-1]["loss"]}


@torch.no_grad()
def evaluate(model, data, bs, teacher=None):
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    losses = [ce(model(x), y).item() for x, y in (batch(data, bs, gen) for _ in range(20))]
    result = {"params": sum(p.numel() for p in model.parameters()), "val_loss": sum(losses) / len(losses)}
    result["ppl"] = math.exp(result["val_loss"])
    if teacher is not None:
        teacher.eval()
        gen, hit, total = torch.Generator().manual_seed(4321), 0, 0
        for _ in range(10):
            x, _ = batch(data, bs, gen)
            a, b = model(x).argmax(-1), teacher(x).argmax(-1)
            hit += (a == b).sum().item()
            total += b.numel()
        result["agree"] = hit / total
    return result


@torch.no_grad()
def generate(model, chars):
    model.eval()
    torch.manual_seed(2026)
    ids = torch.tensor([[chars.index("月")]])
    for _ in range(60):
        logits = model(ids[:, -96:])[:, -1, :] / 0.9
        top, _ = logits.topk(20)
        logits[logits < top[:, [-1]]] = -float("inf")
        ids = torch.cat((ids, torch.multinomial(F.softmax(logits, -1), 1)), 1)
    return "".join(chars[i] for i in ids[0].tolist())


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_teacher(args, chars, tr, va, c=160):
    checkpoint_dir = OUT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"teacher_c{c}.pt"
    stats = checkpoint_dir / f"teacher_c{c}.json"
    m = new_model(len(chars), c, args.seed)
    if path.exists() and stats.exists() and not args.retrain:
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        result = json.loads(stats.read_text(encoding="utf-8"))
        assert result["steps"] == args.teacher_iters
    else:
        result = train(m, tr, args.teacher_iters, args, tag=f"teacher{c}")
        result.update(evaluate(m, va, args.batch_size))
        result["sample"] = generate(m, chars)
        result["width"] = c
        torch.save(m.state_dict(), path)
        save_json(f"checkpoints/{stats.name}", result)
    m.zero_grad(set_to_none=True)
    m.requires_grad_(False)
    print(f"[teacher{c}] val={result['val_loss']:.4f} ppl={result['ppl']:.4f}", flush=True)
    return m, result


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "teacher", "all", "ablate", "scan"], default="all")
    p.add_argument("--teacher-iters", type=int, default=200)
    p.add_argument("--student-iters", type=int, default=120)
    p.add_argument("--abl-iters", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--T", type=float, default=4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--retrain", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.mode == "quick":
        OUT = OUT / "quick"
        args.teacher_iters, args.student_iters, args.batch_size = 30, 20, 8
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    chars, tr, va = make_data()
    metadata = {"args": vars(args), "device": "cpu", "python": sys.version.split()[0],
                "torch": torch.__version__, "platform": platform.platform(), "threads": torch.get_num_threads(),
                "vocab": len(chars), "train_chars": len(tr), "val_chars": len(va),
                "train_sha256": hashlib.sha256(tr.numpy().tobytes()).hexdigest(),
                "val_sha256": hashlib.sha256(va.numpy().tobytes()).hexdigest()}
    print(metadata, flush=True)
    teacher, teacher_result = get_teacher(args, chars, tr, va)
    result = {"metadata": metadata, "teacher": teacher_result}
    if args.mode in ("quick", "all"):
        for tag, tm in (("scratch", None), ("kd", teacher)):
            m = new_model(len(chars), 64, args.seed + 7)
            row = train(m, tr, args.student_iters, args, tm, args.T, args.alpha, tag=tag)
            row.update(evaluate(m, va, args.batch_size, teacher))
            row["sample"] = generate(m, chars)
            result[tag] = row
            torch.save(m.state_dict(), OUT / "checkpoints" / f"student_{tag}.pt")
            print(f"[{tag}] val={row['val_loss']:.4f} ppl={row['ppl']:.4f} agree={row['agree']:.4f}", flush=True)
        assert result["scratch"]["init_sha256"] == result["kd"]["init_sha256"]
        assert result["scratch"]["batches_sha256"] == result["kd"]["batches_sha256"]
        assert all(p.grad is None for p in teacher.parameters())
    elif args.mode == "ablate":
        gen = torch.Generator().manual_seed(2024)
        teacher.eval()
        with torch.no_grad():
            cache = [(x, y, teacher(x)) for x, y in (batch(tr, args.batch_size, gen) for _ in range(32))]
        result["ablation"] = []
        for t in (1, 2, 4, 8):
            for a in (0.3, 0.5, 0.7, 0.9):
                m = new_model(len(chars), 64, args.seed + 7)
                row = train(m, tr, args.abl_iters, args, T=t, alpha=a, cache=cache, tag=f"T{t}_a{a}")
                row.update(evaluate(m, va, args.batch_size))
                row.update(T=t, alpha=a)
                result["ablation"].append(row)
                print(f"[ablation] T={t} alpha={a} val={row['val_loss']:.4f}", flush=True)
                save_json("results_ablate.json", result)
    elif args.mode == "scan":
        result["scan"] = []
        for c in (96, 160, 256):
            tm, rowt = (teacher, teacher_result) if c == 160 else get_teacher(args, chars, tr, va, c)
            m = new_model(len(chars), 64, args.seed + 7)
            row = train(m, tr, args.student_iters, args, tm, args.T, args.alpha, tag=f"kd_c{c}")
            row.update(evaluate(m, va, args.batch_size, tm))
            result["scan"].append({"width": c, "teacher": rowt, "student": row})
            save_json("results_scan.json", result)
    result["wall_time_s"] = time.perf_counter() - start
    save_json(f"results_{args.mode}.json", result)
    print(f"[done] {args.mode}: {result['wall_time_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
