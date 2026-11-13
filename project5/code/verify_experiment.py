"""验证实际日志、评分边界、DPO 符号与新增偏好对。"""
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import hw5_dpo as hw

R = hw.RESULTS
def read(name): return json.loads((R/name).read_text(encoding='utf-8'))


def main():
    checks = [hw.dpo_check(s) for s in ['1.2,0.8,0.9,0.6,0.1', '-1,-2,-3,-2,0.1', '-3,-2,-1,-2,0.1', '1.5,1.0,0.2,0.6,0.1']]
    hw.save('dpo_checks.json', checks)
    assert abs(checks[0]['loss'] - .6881596805) < 1e-8
    assert checks[1]['loss'] < math.log(2) < checks[2]['loss']
    assert math.isfinite(hw.dpo_check('-10000,0,10000,0,1')['loss'])
    pw, pl = torch.tensor(-1., requires_grad=True), torch.tensor(-2., requires_grad=True)
    (-F.logsigmoid(.1*(pw-pl))).backward()
    assert pw.grad.item() < 0 < pl.grad.item()
    q = dict(type='exist', q='图里有没有蓝色图形？', gold='有')
    assert not hw.judge(q, '没有。')
    q['gold'] = '没有'
    assert hw.judge(q, 'No.') and not hw.judge(q, 'The image does not contain any text.')
    assert hw.judge(dict(type='count', gold='3'), 'There are three circles.')
    assert not hw.judge(dict(type='count', gold='3'), '1. 3.')
    summary, allrows = {}, []
    for tag in ['baseline', 'sft', 'dpo', 'dpo17']:
        p = R/(tag+'_eval.json')
        if not p.exists(): continue
        d = read(p.name)
        for row in d['records']:
            row['ok'] = hw.judge(row, row['pred'])
            allrows.append(dict(stage=tag, **row))
        m = {tp: sum(r['ok'] for r in d['records'] if r['type'] == tp) for tp in ['count', 'attr', 'exist']}
        m.update(total=sum(m.values()))
        m['accuracy'] = m['total']/15
        d['metrics'] = m; hw.save(p.name, d); summary[tag] = m
        assert len(d['records']) == 15
    with (R/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['stage', 'id', 'type', 'img', 'q', 'gold', 'pred', 'ok', 'guide_ok']); w.writeheader(); w.writerows(allrows)
    for tag, steps in [('sft', 60), ('dpo', 40), ('dpo17', 40)]:
        d = read(tag+'_training.json')
        assert len(d['logs']) == steps == d['steps']
        assert all(math.isfinite(x['loss']) and x['loss'] >= 0 for x in d['logs'])
        if tag != 'sft':
            for x in d['logs']:
                m = d['beta']*((x['pol_w']-x['ref_w'])-(x['pol_l']-x['ref_l']))
                assert abs(m-x['margin']) < 2e-5
                expected = max(0., -m)+math.log1p(math.exp(-abs(m)))
                assert abs(x['loss']-expected) < 2e-5
    hw.save('summary.json', summary)
    torch.manual_seed(hw.SEED); torch.set_num_threads(4)
    processor, model = hw.load_model()
    from peft import PeftModel
    outputs = {}
    reference_scores = []
    for tag in ['sft', 'dpo', 'dpo17']:
        if tag == 'sft': model = PeftModel.from_pretrained(model, R/'sft_adapter', adapter_name='sft')
        else: model.load_adapter(R/(tag+'_adapter'), adapter_name=tag)
        model.set_adapter(tag); model.eval()
        records = []
        for p in hw.EXTRA_PAIRS:
            inp = hw.encode(processor, hw.image(p['img']), p['prompt']).to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=24, do_sample=False)
            pred = processor.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            q = dict(type=p['type'], gold=p['gold'], q=p['prompt'])
            records.append(dict(id=p['id'], pred=pred, ok=hw.judge(q, pred)))
        outputs[tag] = records
        e = hw.encode(processor, hw.image('img01'), hw.PRE_PAIRS[0]['prompt'], hw.PRE_PAIRS[0]['chosen'])
        with torch.no_grad(), model.disable_adapter():
            reference_scores.append(hw.seq_logprob(model, e).item())
    assert max(reference_scores)-min(reference_scores) < 1e-6
    hw.save('extra_eval.json', outputs)
    hw.save('verification.json', dict(status='passed', steps=[60,40,40], dpo_log_formula=True, margin_gradient_sign=True, judge_negative_and_english=True, reference_logps=reference_scores, prompt_masks=15, model_sha256=hashlib.file_digest((hw.MODEL/'model.safetensors').open('rb'), 'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256((hw.MODEL/'model.safetensors').read_bytes()).hexdigest()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
