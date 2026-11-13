"""实验五：固定几何题集上的 NF4 QLoRA、SFT 与 DPO。"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
IMGS = RESULTS / 'hw5_imgs'
MODEL_ID = 'HuggingFaceTB/SmolVLM-256M-Instruct'
REVISION = '7e3e67edbbed1bf9888184d9df282b700a323964'
MODEL = ROOT / '.cache/model'
SEED = 42
COLORS = {'红': (220, 50, 50), '蓝': (50, 80, 220), '绿': (40, 170, 70), '黄': (230, 190, 40)}
IMG_SPEC = {
    'img01': [('红', '圆', 2), ('蓝', '方', 1)], 'img02': [('绿', '圆', 3)],
    'img03': [('蓝', '圆', 1), ('黄', '方', 2)], 'img04': [('红', '方', 3)],
    'img05': [('绿', '方', 2), ('红', '圆', 1)], 'img06': [('黄', '圆', 4)],
    'img07': [('蓝', '方', 2), ('绿', '圆', 1)], 'img08': [('红', '圆', 4)],
    'img09': [('绿', '方', 1), ('黄', '圆', 2)], 'img10': [('蓝', '圆', 3)],
    'img11': [('黄', '方', 1)], 'img12': [('红', '方', 1), ('蓝', '方', 1)],
    'img13': [('绿', '圆', 2), ('黄', '圆', 1)],
    'img14': [('蓝', '方', 3), ('红', '方', 1)], 'img15': [('黄', '圆', 2), ('绿', '圆', 2)],
}
QUESTIONS = [
    dict(id='P01', type='count', img='img01', q='图中有几个红色圆形？', gold='2'),
    dict(id='P02', type='count', img='img02', q='图中有几个绿色圆形？', gold='3'),
    dict(id='P03', type='count', img='img03', q='图中有几个黄色方形？', gold='2'),
    dict(id='P04', type='count', img='img06', q='图中有几个黄色圆形？', gold='4'),
    dict(id='P05', type='count', img='img10', q='图中有几个蓝色圆形？', gold='3'),
    dict(id='P06', type='attr', img='img01', q='图中蓝色的图形是什么形状？', gold='方形'),
    dict(id='P07', type='attr', img='img05', q='图中红色的图形是什么形状？', gold='圆形'),
    dict(id='P08', type='attr', img='img09', q='图中绿色的图形是什么形状？', gold='方形'),
    dict(id='P09', type='attr', img='img13', q='图中黄色的图形是什么形状？', gold='圆形'),
    dict(id='P10', type='attr', img='img14', q='图中蓝色的图形是什么形状？', gold='方形'),
    dict(id='P11', type='exist', img='img11', q='图里有没有黄色方形？', gold='有'),
    dict(id='P12', type='exist', img='img12', q='图里有没有红色圆形？', gold='没有'),
    dict(id='P13', type='exist', img='img07', q='图里有没有绿色圆形？', gold='有'),
    dict(id='P14', type='exist', img='img08', q='图里有没有绿色方形？', gold='没有'),
    dict(id='P15', type='exist', img='img15', q='图里有没有蓝色图形？', gold='没有'),
]
SFT_DATA = [dict(img=q['img'], prompt=q['q'], answer=q['gold'] + ('个。' if q['type'] == 'count' else '。')) for q in QUESTIONS]
PRE_PAIRS = [
    ('P01', '2 个。', '3 个。图中有三个红色圆形整齐排列。'),
    ('P02', '3 个。', '2 个。'), ('P03', '2 个。', '没有看到黄色图形。'),
    ('P04', '4 个。', '3 个。'), ('P05', '3 个。', '图中有三个蓝色方形。'),
    ('P06', '方形。', '圆形。'), ('P07', '圆形。', '方形，而且有两个红色方形。'),
    ('P08', '方形。', '图中没有绿色图形。'), ('P11', '有。', '没有。'),
    ('P12', '没有。', '有，图左上角有一个红色圆形。'),
    ('P14', '没有。', '有，一共两个绿色方形。'),
    ('P15', '没有。图中只有黄色和绿色的圆形。', '有，两个蓝色圆形。'),
]
PRE_PAIRS = [dict(id=f'B{i+1:02d}', img=QUESTIONS[int(pid[1:])-1]['img'], prompt=QUESTIONS[int(pid[1:])-1]['q'], chosen=w, rejected=l) for i, (pid, w, l) in enumerate(PRE_PAIRS)]
EXTRA_PAIRS = [
    dict(id='N01', img='img04', prompt='图中有几个红色方形？', chosen='3个。', rejected='2个。', failure='少数一个', type='count', gold='3'),
    dict(id='N02', img='img07', prompt='图中蓝色的图形是什么形状？', chosen='方形。', rejected='圆形。', failure='颜色与形状绑定错误', type='attr', gold='方形'),
    dict(id='N03', img='img13', prompt='图中有几个绿色圆形？', chosen='2个。', rejected='3个，三个都是绿色圆形。', failure='将黄色圆形也计入', type='count', gold='2'),
    dict(id='N04', img='img14', prompt='图里有没有红色圆形？', chosen='没有。', rejected='有，右上角有一个红色圆形。', failure='不存在的物体及位置幻觉', type='exist', gold='没有'),
    dict(id='N05', img='img15', prompt='图中黄色的图形是什么形状？', chosen='圆形。', rejected='圆形。图中还有两只蓝色小鸟。', failure='答案正确但附加虚构内容', type='attr', gold='圆形'),
]


def save(name, value):
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def make_data():
    IMGS.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(SEED)
    boxes = {}
    for key, specs in IMG_SPEC.items():
        im = Image.new('RGB', (256, 256), 'white')
        draw = ImageDraw.Draw(im)
        boxes[key] = []
        for color, shape, count in specs:
            for _ in range(count):
                x, y, s = rnd.randint(10, 200), rnd.randint(10, 200), rnd.randint(30, 50)
                box = [x, y, x+s, y+s]
                (draw.ellipse if shape == '圆' else draw.rectangle)(box, outline=COLORS[color], width=4)
                boxes[key].append(dict(color=color, shape=shape, box=box))
        im.save(IMGS / (key+'.png'))
    save('questions.json', QUESTIONS)
    save('sft_data.json', SFT_DATA)
    save('preference_pairs.json', dict(original=PRE_PAIRS, extra=EXTRA_PAIRS))
    save('image_specs.json', boxes)
    print('[data] 15 images, 15 questions, 12+5 preference pairs; seed=42', flush=True)


def image(img_id):
    return Image.open(IMGS / (img_id+'.png')).convert('RGB')


def judge(q, pred):
    """统一的保守匹配：处理英文答案、否定词；不以“有”匹配“没有”。"""
    p = pred.lower().replace(' ', '')
    if q['type'] == 'exist':
        text = pred.lower().strip()
        color = next((v for k, v in {'红': 'red', '蓝': 'blue', '绿': 'green', '黄': 'yellow'}.items() if k in q['q']), '')
        no = bool(re.search(r'没有|不存在|未见', text)) or bool(re.match(r'no\b', text)) or (bool(re.search(r'\b(no|not)\b', text)) and color in text)
        yes = ('有' in p and not no) or bool(re.match(r'yes\b', text))
        return (no and not yes) if q['gold'] == '没有' else (yes and not no)
    if q['type'] == 'attr':
        square = bool(re.search(r'方形|正方|矩形|square|rectangle', p))
        circle = bool(re.search(r'圆形|圆圈|circle|circular', p))
        return (square and not circle) if q['gold'] == '方形' else (circle and not square)
    nums = set(re.findall(r'(?<!\d)\d+(?!\d)', p))
    nums.update(str(i) for i, n in enumerate('零一二三四五六七八九') if re.search(n+r'(?:个|只)', p))
    nums.update(str(i) for i, n in enumerate(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']) if re.search(r'\b'+n+r'\b', pred.lower()))
    return nums == {q['gold']}


def guide_judge(q, pred):
    return q['gold'] in pred.replace(' ', '')


def source():
    if (MODEL / 'config.json').exists():
        return str(MODEL), {}
    return MODEL_ID, dict(revision=REVISION)


def load_model(adapter=None, quantized=True):
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    src, kw = source()
    processor = AutoProcessor.from_pretrained(src, **kw)
    processor.image_processor.size = {'longest_edge': 512}
    cfg = dict(torch_dtype=torch.float32, attn_implementation='sdpa', **kw)
    if quantized:
        if not torch.cuda.is_available():
            raise RuntimeError('NF4 路径需要 CUDA；可用 --fp32 明确切换为 LoRA 演示。')
        cfg.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float32, bnb_4bit_use_double_quant=True))
    model = AutoModelForVision2Seq.from_pretrained(src, **cfg)
    if not quantized:
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
    if adapter:
        from peft import PeftModel, prepare_model_for_kbit_training
        if quantized:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(model, RESULTS / adapter, is_trainable=True)
    return processor, model


def prompt(processor, text):
    return processor.apply_chat_template([dict(role='user', content=[dict(type='image'), dict(type='text', text=text)])], add_generation_prompt=True)


def encode(processor, img, question, answer=None):
    pre = prompt(processor, question)
    prefix = processor(text=[pre], images=[img], return_tensors='pt')
    if answer is None:
        return prefix
    full = processor(text=[pre+answer], images=[img], return_tensors='pt')
    n = prefix['input_ids'].shape[1]
    assert torch.equal(full['input_ids'][:, :n], prefix['input_ids']), 'prompt token boundary changed'
    labels = full['input_ids'].clone()
    labels[:, :n] = -100
    assert (labels[:, 1:] != -100).sum() > 0
    return full, labels


@torch.no_grad()
def evaluate(processor, model, tag):
    model.eval()
    start = time.perf_counter()
    records = []
    for q in QUESTIONS:
        inp = encode(processor, image(q['img']), q['q']).to(model.device)
        out = model.generate(**inp, max_new_tokens=24, do_sample=False, use_cache=True)
        pred = processor.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        records.append(dict(**q, pred=pred, ok=judge(q, pred), guide_ok=guide_judge(q, pred)))
        print(f"[{tag}:{q['id']}] {pred!r} gold={q['gold']} ok={records[-1]['ok']}", flush=True)
    def metric(field):
        m = {tp: sum(r[field] for r in records if r['type'] == tp) for tp in ('count', 'attr', 'exist')}
        m['total'] = sum(m.values())
        m['accuracy'] = m['total']/15
        return m
    result = dict(tag=tag, metrics=metric('ok'), guide_metrics=metric('guide_ok'), seconds=time.perf_counter()-start, records=records)
    save(tag+'_eval.json', result)
    print(f'[{tag}] {result["metrics"]}', flush=True)
    return result


def seq_logprob(model, encoded):
    full, labels = encoded
    full, labels = full.to(model.device), labels.to(model.device)
    logits = model(**full, use_cache=False).logits[:, :-1].float()
    target = labels[:, 1:]
    mask = target.ne(-100)
    token_logp = F.log_softmax(logits, -1).gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return token_logp[mask].sum()


def train_sft(processor, model, steps):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if getattr(model, 'is_loaded_in_4bit', False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
    trainable, total = model.get_nb_trainable_parameters()
    model.print_trainable_parameters()
    assert all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
    params = dict(trainable=trainable, total=total, percent=100*trainable/total, quantized_modules=sum(m.__class__.__name__ == 'Linear4bit' for m in model.modules()), r=8, alpha=16, dropout=.05)
    save('parameters.json', params)
    (RESULTS/'parameters.txt').write_text(f'trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.4f}\nNF4 Linear4bit modules: {params["quantized_modules"]}\n', encoding='utf-8')
    cached = [encode(processor, image(s['img']), s['prompt'], s['answer']) for s in SFT_DATA]
    save('mask_check.json', [dict(id=q['id'], prompt_tokens=int((e[1] == -100).sum()), answer_tokens=int((e[1] != -100).sum()), prompt_prefix_equal=True) for q, e in zip(QUESTIONS, cached)])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    start = time.perf_counter()
    logs = []
    rnd, order = random.Random(SEED), []
    for step in range(steps):
        if step % len(cached) == 0:
            order = list(range(len(cached))); rnd.shuffle(order)
        idx = order[step % len(order)]
        full, labels = cached[idx]
        loss = model(**full.to(model.device), labels=labels.to(model.device), use_cache=False).loss
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        logs.append(dict(step=step+1, id=QUESTIONS[idx]['id'], loss=loss.item()))
        print(f'[sft] step={step+1}/{steps} loss={loss.item():.5f}', flush=True)
    model.save_pretrained(RESULTS/'sft_adapter')
    save('sft_training.json', dict(steps=steps, seconds=time.perf_counter()-start, lr=1e-4, logs=logs))
    return model


def train_dpo(processor, model, steps, beta, extra=False):
    tag = 'dpo17' if extra else 'dpo'
    pairs = PRE_PAIRS + EXTRA_PAIRS if extra else PRE_PAIRS
    # eval 仅关闭 dropout；梯度仍开启，使 policy 与冻结 reference 的比较没有随机噪声。
    model.eval()
    cached = [(encode(processor, image(p['img']), p['prompt'], p[k])) for p in pairs for k in ('chosen', 'rejected')]
    start = time.perf_counter()
    with torch.no_grad(), model.disable_adapter():
        refs = [(seq_logprob(model, cached[2*i]).item(), seq_logprob(model, cached[2*i+1]).item()) for i in range(len(pairs))]
    save(tag+'_reference.json', [dict(id=p['id'], chosen=r[0], rejected=r[1]) for p, r in zip(pairs, refs)])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    logs = []
    rnd, order = random.Random(SEED), []
    for step in range(steps):
        if step % len(pairs) == 0:
            order = list(range(len(pairs))); rnd.shuffle(order)
        idx = order[step % len(order)]
        pw = seq_logprob(model, cached[2*idx])
        pl = seq_logprob(model, cached[2*idx+1])
        rw, rl = refs[idx]
        margin = beta*((pw-rw)-(pl-rl))
        loss = -F.logsigmoid(margin)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        log = dict(step=step+1, id=pairs[idx]['id'], pol_w=pw.item(), ref_w=rw, pol_l=pl.item(), ref_l=rl, margin=margin.item(), loss=loss.item())
        logs.append(log)
        print(f'[{tag}] step={step+1}/{steps} loss={loss.item():.5f} margin={margin.item():.5f}', flush=True)
    model.save_pretrained(RESULTS/(tag+'_adapter'))
    save(tag+'_training.json', dict(steps=steps, pairs=len(pairs), seconds=time.perf_counter()-start, lr=5e-5, beta=beta, reference='frozen base with disabled adapter', dropout='disabled', logs=logs))


def dpo_check(values):
    pw, rw, pl, rl, beta = map(float, values.split(','))
    margin = beta*((pw-rw)-(pl-rl))
    # stable -log sigmoid；指南示例漏了负号，不能照抄。
    loss = max(0., -margin)+math.log1p(math.exp(-abs(margin)))
    result = dict(inputs=values, margin=margin, loss=loss)
    print(json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['make_data', 'infer', 'sft', 'dpo', 'eval', 'all'], default='infer')
    ap.add_argument('--steps', type=int)
    ap.add_argument('--beta', type=float, default=.1)
    ap.add_argument('--extra', action='store_true')
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--dpo_check')
    args = ap.parse_args()
    if args.dpo_check:
        dpo_check(args.dpo_check); return
    torch.manual_seed(SEED); random.seed(SEED); torch.set_num_threads(4)
    make_data()
    if args.mode == 'make_data': return
    RESULTS.mkdir(exist_ok=True)
    import platform, importlib.metadata as md
    save('environment.json', dict(python=platform.python_version(), packages={k: md.version(k) for k in ['torch', 'transformers', 'peft', 'accelerate', 'Pillow', 'bitsandbytes']}, device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU', model_revision=REVISION, model_id=MODEL_ID, precision='fp32 LoRA' if args.fp32 else 'NF4 double quantization + fp32 compute', seed=SEED, image_longest_edge=512, max_new_tokens=24, quantized=not args.fp32))
    quant = not args.fp32
    processor, model = load_model('sft_adapter' if args.mode == 'dpo' else None, quant)
    if args.mode in ('infer', 'all'): evaluate(processor, model, 'baseline')
    if args.mode in ('sft', 'all'):
        model = train_sft(processor, model, args.steps or 60)
        evaluate(processor, model, 'sft')
    if args.mode in ('dpo', 'all'):
        train_dpo(processor, model, args.steps or 40, args.beta, args.extra)
        evaluate(processor, model, 'dpo17' if args.extra else 'dpo')
    if args.mode == 'eval':
        from peft import PeftModel
        evaluate(processor, model, 'baseline')
        model = PeftModel.from_pretrained(model, RESULTS/'sft_adapter', adapter_name='sft')
        evaluate(processor, model, 'sft')
        for name in ('dpo', 'dpo17'):
            if (RESULTS/(name+'_adapter')).exists():
                model.load_adapter(RESULTS/(name+'_adapter'), adapter_name=name)
                model.set_adapter(name)
                evaluate(processor, model, name)
    if torch.cuda.is_available():
        save('memory_'+args.mode+('_extra' if args.extra else '')+'.json', dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__ == '__main__':
    main()
