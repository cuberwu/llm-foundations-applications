"""完整实验，已有完整结果时断点继续；所有训练配置使用同一语料和种子。"""
import json
import torch
from min_llm import BASE, ROOT, POEMS, load_model, sample, train

RESULTS = ROOT / "results"
TRAINING = RESULTS / "training"

RUNS = [
    ("baseline", 2000, 1e-3, {}),
    ("exp1_lr_high", 1000, 1e-2, {}),
    ("exp1_lr_low", 1000, 1e-4, {}),
    ("exp2_layer1", 1000, 1e-3, {"n_layer": 1}),
    ("exp2_layer4", 1000, 1e-3, {"n_layer": 4}),
    ("exp3_emb64", 1000, 1e-3, {"n_embd": 64, "n_head": 2}),
    ("exp3_emb256", 1000, 1e-3, {"n_embd": 256, "n_head": 8}),
    ("exp4_context32", 1000, 1e-3, {"block_size": 32}),
    ("exp5_no_pos", 1000, 1e-3, {"use_pos": False}),
]


def text_metrics(text):
    content = text[1:]  # 不计提示词。
    grams = [content[i:i+3] for i in range(len(content)-2)]
    corpus = "\n".join(POEMS)
    copied = sum(content[i:i+10] in corpus for i in range(len(content)-9))
    return dict(distinct3=len(set(grams)) / len(grams),
                repeat3=1 - len(set(grams)) / len(grams),
                corpus_match10=copied / (len(content)-9),
                punctuation=sum(ch in "，。！？；" for ch in content))


def sampling_experiments(threads):
    torch.set_num_threads(threads)
    model, tok = load_model(TRAINING / "baseline/model.pt")
    groups = [("s1_base", 1.0, 20, 1.0), ("s2_cold", 0.5, 20, 1.0),
              ("s3_hot", 1.5, 20, 1.0), ("s4_top5", 1.0, 5, 1.0),
              ("s5_unlimited", 1.0, 0, 1.0),
              ("advanced_before", 0.5, 5, 1.0),
              ("advanced_after", 0.5, 5, 1.2)]
    results = []
    for group, temperature, top_k, penalty in groups:
        samples = []
        for seed in (101, 102, 103):
            text = sample(model, tok, "春", seed, temperature=temperature,
                          top_k=top_k, repetition_penalty=penalty)
            samples.append(dict(seed=seed, text=text, **text_metrics(text)))
        row = dict(group=group, temperature=temperature, top_k=top_k,
                   repetition_penalty=penalty, samples=samples)
        for metric in ("distinct3", "repeat3", "corpus_match10", "punctuation"):
            row[metric] = sum(s[metric] for s in samples) / len(samples)
        results.append(row)
        print(f"SAMPLE {group} repeat3={row['repeat3']:.3f} "
              f"corpus_match10={row['corpus_match10']:.3f}", flush=True)
    out = RESULTS / "sampling"; out.mkdir(exist_ok=True)
    (out / "sampling.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "generated_all.txt").write_text("\n\n".join(
        f"{g['group']} temp={g['temperature']} top_k={g['top_k']} penalty={g['repetition_penalty']} "
        f"seed={s['seed']}\n{s['text']}" for g in results for s in g["samples"]), encoding="utf-8")
    model1000, _ = load_model(TRAINING / "baseline/model_step1000.pt")
    samples = [dict(prompt=p, seed=s, text=sample(model1000, tok, p, s))
               for p in ("春", "月") for s in (101, 102)]
    (TRAINING / "baseline/generated_step1000.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    benchmark = json.loads((RESULTS / "logs/thread_benchmark.json").read_text())
    threads = min(benchmark, key=lambda r: r["seconds_per_step"])["threads"]
    print(f"Selected CPU threads: {threads}", flush=True)
    all_results = []
    for run_id, iters, lr, config in RUNS:
        out = TRAINING / run_id
        if (out / "metrics.json").exists() and (out / "generated.json").exists():
            result = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            assert result["config"] == BASE | config and result["iters"] == iters and result["lr"] == lr
            assert len((out / "losses.csv").read_text().splitlines()) == iters + 1
            print(f"RESUME: {run_id} already complete", flush=True)
        else:
            result = train(run_id, iters, lr, threads=threads, **config)
        all_results.append(result)
        (RESULTS / "summary.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    sampling_experiments(threads)
    print("ALL EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
