"""指南同构的字符级 GPT：CPU 训练、保存检查点和独立采样。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]

POEMS = [
    '床前明月光，疑是地上霜。举头望明月，低头思故乡。',
    '春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。',
    '白日依山尽，黄河入海流。欲穷千里目，更上一层楼。',
    '锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。',
    '离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。',
    '远芳侵古道，晴翠接荒城。又送王孙去，萋萋满别情。',
    '千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。',
    '鹅，鹅，鹅，曲项向天歌。白毛浮绿水，红掌拨清波。',
    '两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。',
    '朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。',
    '故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。',
    '李白乘舟将欲行，忽闻岸上踏歌声。桃花潭水深千尺，不及汪伦送我情。',
    '日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。',
    '天门中断楚江开，碧水东流至此回。两岸青山相对出，孤帆一片日边来。',
    '月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。',
    '清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。',
    '独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。',
    '烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。',
    '折戟沉沙铁未销，自将磨洗认前朝。东风不与周郎便，铜雀春深锁二乔。',
    '巴山楚水凄凉地，二十三年弃置身。怀旧空吟闻笛赋，到乡翻似烂柯人。',
    '沉舟侧畔千帆过，病树前头万木春。今日听君歌一曲，暂凭杯酒长精神。',
    '朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。',
    '湖光秋月两相和，潭面无风镜未磨。遥望洞庭山水翠，白银盘里一青螺。',
    '杨柳青青江水平，闻郎江上踏歌声。东边日出西边雨，道是无晴却有晴。',
    '自古逢秋悲寂寥，我言秋日胜春朝。晴空一鹤排云上，便引诗情到碧霄。',
    '前不见古人，后不见来者。念天地之悠悠，独怆然而涕下。',
    '葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。',
    '黄河远上白云间，一片孤城万仞山。羌笛何须怨杨柳，春风不度玉门关。',
    '秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。',
    '月黑雁飞高，单于夜遁逃。欲将轻骑逐，大雪满弓刀。',
    '好雨知时节，当春乃发生。随风潜入夜，润物细无声。',
    '晓看红湿处，花重锦官城。野径云俱黑，江船火独明。',
    '国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。',
    '烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。',
    '细草微风岸，危樯独夜舟。星垂平野阔，月涌大江流。',
    '风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。',
    '花近高楼伤客心，万方多难此登临。锦江春色来天地，玉垒浮云变古今。',
    '独怜幽草涧边生，上有黄鹂深树鸣。春潮带雨晚来急，野渡无人舟自横。',
    '天街小雨润如酥，草色遥看近却无。最是一年春好处，绝胜烟柳满皇都。',
    '昔人已乘黄鹤去，此地空余黄鹤楼。黄鹤一去不复返，白云千载空悠悠。',
    '晴川历历汉阳树，芳草萋萋鹦鹉洲。日暮乡关何处是，烟波江上使人愁。',
    '客舍青青柳色新，渭城朝雨浥轻尘。劝君更尽一杯酒，西出阳关无故人。',
    '寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。',
    '山光忽西落，池月渐东上。散发乘夕凉，开轩卧闲敞。',
    '荷笠带斜阳，青山独归远。苍苍竹林寺，杳杳钟声晚。',
    '空山不见人，但闻人语响。返景入深林，复照青苔上。',
    '人闲桂花落，夜静春山空。月出惊山鸟，时鸣春涧中。',
    '红豆生南国，春来发几枝。愿君多采撷，此物最相思。',
    '独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。',
    '君自故乡来，应知故乡事。来日绮窗前，寒梅著花未。',
    '山中相送罢，日暮掩柴扉。春草明年绿，王孙归不归。',
    '花间一壶酒，独酌无相亲。举杯邀明月，对影成三人。',
    '小时不识月，呼作白玉盘。又疑瑶台镜，飞在青云端。',
    '长安一片月，万户捣衣声。秋风吹不尽，总是玉关情。',
    '弃我去者，昨日之日不可留。乱我心者，今日之日多烦忧。',
    '抽刀断水水更流，举杯消愁愁更愁。人生在世不称意，明朝散发弄扁舟。',
    '千里黄云白日曛，北风吹雁雪纷纷。莫愁前路无知己，天下谁人不识君。',
    '慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。谁言寸草心，报得三春晖。',
    '山重水复疑无路，柳暗花明又一村。莫笑农家腊酒浑，丰年留客足鸡豚。',
    '纸上得来终觉浅，绝知此事要躬行。古人学问无遗力，少壮功夫老始成。',
    '死去元知万事空，但悲不见九州同。王师北定中原日，家祭无忘告乃翁。',
    '小荷才露尖尖角，早有蜻蜓立上头。泉眼无声惜细流，树阴照水爱晴柔。',
    '接天莲叶无穷碧，映日荷花别样红。毕竟西湖六月中，风光不与四时同。',
    '胜日寻芳泗水滨，无边光景一时新。等闲识得东风面，万紫千红总是春。',
    '半亩方塘一鉴开，天光云影共徘徊。问渠那得清如许，为有源头活水来。',
    '郁孤台下清江水，中间多少行人泪。西北望长安，可怜无数山。',
    '人生自古谁无死，留取丹心照汗青。辛苦遭逢起一经，干戈寥落四周星。',
    '咬定青山不放松，立根原在破岩中。千磨万击还坚劲，任尔东西南北风。',
    '千锤万凿出深山，烈火焚烧若等闲。粉骨碎身浑不怕，要留清白在人间。',
    '浩荡离愁白日斜，吟鞭东指即天涯。落红不是无情物，化作春泥更护花。',
    '九州生气恃风雷，万马齐喑究可哀。我劝天公重抖擞，不拘一格降人才。',
    '力微任重久神疲，再竭衰庸定不支。苟利国家生死以，岂因祸福避趋之。',
]


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        # 未知字符应明确报错，避免静默删除提示词。
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd 必须能被 n_head 整除")
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = [a.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for a in (q, k, v)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.n_head)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=2,
                 block_size=128, use_pos=True):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd) if use_pos else None
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size)
                                     for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        if not 0 < T <= self.block_size:
            raise ValueError("输入长度应在 1 到 block_size 之间")
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=20,
                 repetition_penalty=1.0):
        if temperature <= 0 or top_k < 0 or repetition_penalty < 1:
            raise ValueError("temperature > 0，top_k >= 0，重复惩罚 >= 1")
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :].clone()
            if repetition_penalty > 1:
                # ponytail: 对窗口内出现过的字符统一惩罚；需要细粒度控制时改为 n-gram。
                for row in range(idx.size(0)):
                    seen = idx[row, -self.block_size:].unique()
                    values = logits[row, seen]
                    logits[row, seen] = torch.where(values > 0,
                        values / repetition_penalty, values * repetition_penalty)
            logits = logits / temperature
            if top_k:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def get_batch(data, block_size, batch_size, generator):
    if len(data) <= block_size:
        raise ValueError("语料长度必须超过上下文长度")
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    offsets = starts[:, None] + torch.arange(block_size + 1)
    windows = data[offsets]
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


BASE = dict(n_embd=128, n_head=4, n_layer=2, block_size=128, use_pos=True)


def sample(model, tokenizer, prompt, seed, **kwargs):
    torch.manual_seed(seed)
    out = model.generate(torch.tensor([tokenizer.encode(prompt)]), **kwargs)
    return tokenizer.decode(out[0].tolist())


def train(run_id="baseline", iters=2000, lr=1e-3, seed=42, threads=2,
          batch_size=32, output_root=ROOT / "results/training", **config):
    config = BASE | config
    if iters < 1 or batch_size < 1:
        raise ValueError("iters 与 batch_size 必须为正数")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    text = "\n".join(POEMS)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    model = MiniGPT(tok.vocab_size, **config)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)
    out_dir = Path(output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{run_id}: V={tok.vocab_size}, params={n_params:,}, config={config}", flush=True)
    losses, elapsed = [], []
    t0 = time.perf_counter()
    for step in range(1, iters + 1):
        model.train()
        xb, yb = get_batch(data, config["block_size"], batch_size, rng)
        _, loss = model(xb, yb)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{run_id} 在第 {step} 步出现非有限 loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        elapsed.append(time.perf_counter() - t0)
        if step == 1 or step % 100 == 0:
            print(f"{run_id} step={step}/{iters} loss={loss.item():.4f} "
                  f"speed={step / elapsed[-1]:.2f} it/s", flush=True)
        if run_id == "baseline" and step == 1000:
            torch.save(dict(state=model.state_dict(), config=config, chars=tok.chars),
                       out_dir / "model_step1000.pt")
    duration = elapsed[-1]
    checkpoint = dict(state=model.state_dict(), config=config, chars=tok.chars)
    torch.save(checkpoint, out_dir / "model.pt")
    metrics = dict(run_id=run_id, config=config, iters=iters, lr=lr, seed=seed,
        batch_size=batch_size, threads=threads, corpus_chars=len(text),
        corpus_sha256=hashlib.sha256(text.encode()).hexdigest(), vocab=tok.vocab_size,
        parameters=n_params, initial_loss=losses[0], final_loss=losses[-1],
        mean_last100=sum(losses[-100:]) / len(losses[-100:]),
        duration_seconds=duration, torch_version=torch.__version__,
        python_version=platform.python_version(), os=platform.platform())
    if iters >= 1000:
        metrics.update(loss_step1000=losses[999], time_step1000=elapsed[999],
                       mean_steps901_1000=sum(losses[900:1000]) / 100)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False,
                                                   indent=2), encoding="utf-8")
    with (out_dir / "losses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "elapsed_seconds"])
        writer.writerows(zip(range(1, iters + 1), losses, elapsed))
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, iters + 1), losses, linewidth=0.8)
    plt.xlabel("Iteration"); plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{run_id}: L={config['n_layer']}, C={config['n_embd']}, "
              f"T={config['block_size']}, lr={lr:g}")
    plt.tight_layout(); plt.savefig(out_dir / "loss_curve.png", dpi=180); plt.close()
    samples = [dict(prompt=p, seed=s, text=sample(model, tok, p, s))
               for p in ("春", "月") for s in (101, 102)]
    (out_dir / "generated.json").write_text(json.dumps(samples, ensure_ascii=False,
                                                      indent=2), encoding="utf-8")
    (out_dir / "generated.txt").write_text("\n\n".join(
        f"prompt={s['prompt']} seed={s['seed']}\n{s['text']}" for s in samples), encoding="utf-8")
    print(f"DONE {run_id} seconds={duration:.1f} final={losses[-1]:.4f}", flush=True)
    return metrics


def load_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    tok = CharTokenizer("".join(saved["chars"]))
    model = MiniGPT(tok.vocab_size, **saved["config"])
    model.load_state_dict(saved["state"])
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_id", default="baseline")
    for name in ("n_embd", "n_head", "n_layer", "block_size"):
        ap.add_argument(f"--{name}", type=int, default=BASE[name])
    ap.add_argument("--no_pos", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--prompt", default="春")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=20, help="0 表示不限制")
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    args = vars(ap.parse_args())
    checkpoint = args.pop("checkpoint")
    sampling = {k: args.pop(k) for k in ("prompt", "temperature", "top_k", "repetition_penalty")}
    args["use_pos"] = not args.pop("no_pos")
    if checkpoint:
        torch.set_num_threads(args["threads"])
        model, tok = load_model(checkpoint)
        print(sample(model, tok, sampling.pop("prompt"), args["seed"], **sampling))
    else:
        train(**args)


if __name__ == "__main__":
    main()
