"""Minimal CPU pipeline. Run from project7; defaults match the lab guide."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROMPT = 'photorealistic render, studio soft lighting, PBR material, preserve original 3d geometry, sharp'
NEGATIVE = 'deformed geometry, perspective changed'


def run(command, log):
    start = time.perf_counter()
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([str(v) for v in command], cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT)
    text = log.read_text(encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')
    return time.perf_counter() - start, text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='A')
    p.add_argument('--res', type=int, default=512)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--cfg', type=float, default=1.2)
    p.add_argument('--control-strength', type=float, default=0.9)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--control-image', type=Path)
    p.add_argument('--prompt', default=PROMPT)
    p.add_argument('--negative', default=NEGATIVE)
    args = p.parse_args()
    start = time.perf_counter()
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    blender_seconds = 0
    if args.control_image:
        depth = args.control_image.resolve()
    else:
        blender = os.environ.get('BLENDER_EXE') or str(next((ROOT / 'vendor/blender').rglob('blender.exe')))
        scene = ROOT / 'assets/demo_white_model.blend'
        if not scene.exists():
            run([blender, '-b', '-P', ROOT / 'blender/make_demo_scene.py'], out / 'scene.log')
        blender_seconds, _ = run([
            blender, '-b', scene, '-P', ROOT / 'blender/export_control_maps.py', '--',
            '--outdir', out, '--name', args.name, '--res', args.res,
        ], out / f'{args.name}_blender.log')
        depth = out / f'{args.name}_depth_0001.png'
    with Image.open(depth) as image:
        if image.mode != 'RGB':
            depth = out / f'{args.name}_depth_rgb.png'
            image.convert('RGB').save(depth)
    command = [
        ROOT / 'vendor/sdcpp-ext/sd-cli.exe', '--backend', 'cpu',
        '-m', ROOT / 'models/sd_turbo-q4_0.gguf',
        '--control-net', ROOT / 'models/control_depth_sd21-fp16.gguf',
        '--control-image', depth, '--control-strength', args.control_strength,
        '-p', args.prompt, '-n', args.negative, '-W', args.res, '-H', args.res,
        '--steps', args.steps, '--cfg-scale', args.cfg, '--sampling-method', 'euler_a',
        '--scheduler', 'discrete', '-s', args.seed, '-t', args.threads,
        '-o', out / f'{args.name}_photoreal.png',
    ]
    sd_seconds, log = run(command, out / f'{args.name}_sd.log')
    def timing(pattern):
        match = re.search(pattern, log, re.IGNORECASE)
        return float(match.group(1)) if match else None
    record = vars(args).copy()
    record.update(control_image=str(depth), command=[str(c) for c in command],
                  blender_process_seconds=blender_seconds, sd_process_seconds=sd_seconds,
                  total_seconds=time.perf_counter()-start,
                  sampling_seconds=timing(r'sampling completed.*?([\d.]+)\s*s'),
                  vae_seconds=timing(r'decod(?:e|ing).*?completed.*?([\d.]+)\s*s'))
    (out / f'{args.name}_metrics.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
