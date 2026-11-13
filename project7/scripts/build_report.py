"""Fill the supplied lab template with the measured outputs."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
template = next((ROOT / 'docs').glob('*.docx'))
doc = Document(template)
paras = list(doc.paragraphs)
tables = list(doc.tables)
metrics = [json.loads((ROOT / f'output/{n}_metrics.json').read_text()) for n in 'ABC']
renders = [json.loads((ROOT / f'output/{n}_render.json').read_text()) for n in 'ABC']
a, b, c = metrics


def text(p, value, size=10.5, bold=False):
    p.clear()
    run = p.add_run(value)
    run.font.name = 'Arial'
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '微软雅黑')
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    p.paragraph_format.line_spacing = Pt(12 if size <= 9 else 14)
    p.paragraph_format.space_after = Pt(2)
    return p


def cell(t, row, col, value, size=10.5):
    target = tables[t].cell(row, col)
    text(target.paragraphs[0], value, size)
    return target


def trim(t, count):
    table = tables[t]
    for row in list(table.rows)[count:]:
        table._tbl.remove(row._tr)


def answer(t, value):
    trim(t, 1)
    cell(t, 0, 0, value)


def remove(p):
    p._element.getparent().remove(p._element)


# Retain the source's type, page geometry, table borders and footer.
paras[0].style = doc.styles['Title']
for idx in [4, 9, 10, 12, 17, 32]:
    remove(paras[idx])
for t in tables:
    for row in t.rows:
        row.height = None
        props = row._tr.get_or_add_trPr()
        props.append(OxmlElement('w:cantSplit'))
        for target in row.cells:
            for p in target.paragraphs:
                p.paragraph_format.keep_with_next = False
                p.paragraph_format.keep_together = True
for idx in [8, 18, 25, 34]:
    paras[idx].paragraph_format.page_break_before = True

cell(0, 0, 1, '大模型基础与应用探索')
cell(0, 0, 3, '2026 年 11 月 11 日')
cell(0, 2, 3, '本地计算机')
for r, value in enumerate([
    '1. 使用 Blender 导出白模、近亮远暗的深度图和法线图。',
    '2. 在纯 CPU 上运行 Q4_0 扩散模型与 SD2.1 深度 ControlNet。',
    '3. 比较采样步数和控制强度，记录耗时并分析几何与画面瑕疵。',
]):
    cell(1, r, 0, value)
env = [
    'Intel Core i9-13900HX，24 核 32 线程；约 32 GB 内存；实验用 16 线程',
    'Windows 11 64 位',
    'Blender 4.5.13 LTS，Cycles CPU，64 samples',
    'stable-diffusion.cpp，commit 465adf3，Windows CPU 预编译版',
    'sd_turbo-q4_0.gguf，1.72 GiB，Q4_0 混合精度',
    'control_depth_sd21-fp16.gguf，694.74 MiB，SD2.1 深度控制',
    'Python 3.12.14、Pillow 13.0.0；AI 协作工具 Codex',
]
for r, value in enumerate(env, 1):
    cell(2, r, 1, value)
principles = [
    '深度控制：将相机视锥内顶点距离的 2% 与 98% 分位数设为 Mist 起止范围，反相得到近亮远暗的 RGB 深度图；本次范围为 7.39～9.88。',
    '法线检查：把法线 n 按 n×0.5+0.5 映射到 RGB，仅检查表面朝向，不作为本次生成条件。',
    '模型量化：Q4_0 降低权重存储和带宽需求，部分张量仍保留浮点精度。日志中参数内存为 2457.52 MiB，VRAM 为 0；这不等于进程峰值内存。',
    '推理链路：CLIP 文本编码 → 带深度 ControlNet 的 UNet 迭代采样 → VAE 解码。CFG 固定为 1.2，采样器为 euler_a，调度器为 discrete。',
]
for r, value in enumerate(principles):
    cell(3, r, 0, value)
steps = [
    ('准备环境', 'powershell -File\nscripts/setup.ps1', 'vendor/、models/'),
    ('创建白模并导出三图\n运行 A', 'python scripts/run_pipeline.py\n--name A --steps 4', 'A_clay.png\nA_depth_0001.png\nA_normal_0001.png\nA_photoreal.png'),
    ('运行 B', 'python scripts/run_pipeline.py\n--name B --steps 8', 'B_photoreal.png\nB_metrics.json'),
    ('运行 C', 'python scripts/run_pipeline.py\n--name C --steps 4\n--control-strength 0.5', 'C_photoreal.png\nC_metrics.json'),
]
trim(4, 5)
for r, values in enumerate(steps, 1):
    for col, value in enumerate(values, 1):
        cell(4, r, col, value, 9 if col > 1 else 10.5)
for r, m in enumerate(metrics, 1):
    values = [m['name'], '512²', str(m['steps']), '1.2', str(m['control_strength']), 'euler_a', '42']
    for col, value in enumerate(values):
        cell(5, r, col, value, 10)
answer(6, a['prompt'])
answer(7, a['negative'])
# An existing blank after parameters carries the common command settings.
text(paras[14], '在 project7 下执行。默认 512²、CFG 1.2、控制强度 0.9、种子 42、16 线程；流水线自动调用 Blender，记录完整命令与日志。', 9)
trim(8, 4)
for r, m in enumerate(metrics, 1):
    values = [m['name'], f"{m['sampling_seconds']:.2f}", f"{m['vae_seconds']:.2f}",
              f"{m['total_seconds']:.2f}", f"CPU；{m['steps']} 步；强度 {m['control_strength']}"]
    for col, value in enumerate(values):
        cell(8, r, col, value, 10)
render_line = '；'.join(f"{m['name']}：{r['render_seconds']:.2f} s" for m, r in zip(metrics, renders))
text(paras[19], f'各组 clay、depth、normal 在同一次渲染中导出，共享耗时：{render_line}。总耗时包含 Blender 子进程、模型加载、采样与解码，不含下载和模型转换。', 9)


def figures(t, title, files, caption):
    cell(t, 0, 0, title)
    target = tables[t].cell(1, 0)
    p = target.paragraphs[0]
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for filename in files:
        p.add_run().add_picture(str(ROOT / 'output' / filename), width=Inches(1.90))
        if filename != files[-1]:
            p.add_run(' ')
    text(target.add_paragraph(), caption, 9).alignment = WD_ALIGN_PARAGRAPH.CENTER


figures(9, '白模与控制图', ['A_clay.png', 'A_depth_0001.png', 'A_normal_0001.png'],
        '左：clay 白模；中：depth 深度；右：normal 法线。均为 512×512。')
figures(10, '三组实际生成结果', [f'{n}_photoreal.png' for n in 'ABC'],
        '左：A 组 4 步 / 0.9；中：B 组 8 步 / 0.9；右：C 组 4 步 / 0.5。')
trim(11, 2)
cell(11, 0, 0, '轮廓 遮挡与透视')
cell(11, 1, 0, '__GEOMETRY__')
trim(12, 2)
cell(12, 0, 0, '画面瑕疵与成因')
cell(12, 1, 0, '__DEFECTS__')
problems = [
    ('指南所列运行文件缺失', '目录中只有指南与模板', '根据附录补全脚本，下载便携工具与权重，校验 SHA256'),
    ('需匹配控制模型架构', 'SD-Turbo 使用 SD2.x 文本特征维度', '采用 SD2.1 depth ControlNet，并转为 FP16 GGUF'),
    ('前方立方体深度偏白', '近端截断与 8 bit 映射压缩了局部层次', '后续可调整近端范围和构图，保留更多深度层次'),
]
for r, values in enumerate(problems, 1):
    for col, value in enumerate(values):
        cell(13, r, col, value, 10)
answer(14, '__CONCLUSION__')
answers = [
    f'Turbo 经少步蒸馏，较高 CFG 会把预测推离训练分布，容易过饱和或变形。本次均取 1.2；4 步增到 8 步后采样耗时由 {a["sampling_seconds"]:.2f} s 变为 {b["sampling_seconds"]:.2f} s。CFG 与步数没有固定换算关系，增加步数不保证改善结构。',
    '动态范围决定物体深度能否被区分。本次采用 7.39～9.88，前景立方体部分区域仍接近纯白。若远端不变、起点变小，范围扩大，反相深度整体变暗、局部对比减弱，但近端截断会减少；若 Mist depth 保持不变，整个区间前移，远处更易变黑。',
    '__CONTROL__',
    f'主要瓶颈是 UNet 与 ControlNet 的计算及内存访问。A 组采样占总耗时 {a["sampling_seconds"]/a["total_seconds"]*100:.1f}%，VAE 占 {a["vae_seconds"]/a["total_seconds"]*100:.1f}%。可用 T≈T固定+k×W×H×步数+T_VAE(W,H) 粗估；768² 是 512² 的 2.25 倍像素，但实际还受缓存、线程与温度影响。',
    '预乘 alpha 是先将 RGB 与透明度相乘，再同步缩放颜色和 alpha，最后合成。这样可避免白底透明像素的颜色在插值时渗入头发和脸部边缘；直接缩放未预乘图像容易产生白边。本次未开展可选的人像融合。',
    '可用 IP-Adapter FaceID 或 InstantID 注入身份特征，并结合姿态控制，需额外模型与算力；也可用多角度照片训练人物 LoRA 或 DreamBooth，身份更稳定但需数据和训练。局部换脸成本较低，侧脸、遮挡和光照仍需修复。固定种子本身不能保证跨姿态身份一致。',
]
for t, value in zip(range(15, 21), answers):
    answer(t, value)

# Written after inspecting all three generated images and the control maps.
analysis = {
    '__GEOMETRY__': '三组都保留了左后球体、前方主体和右侧圆柱的大致位置，球体被前景物遮挡的关系基本一致，但并未严格还原白模。A 组立方体变成前圆后方的异形块，后方圆环仅剩小块暗色结构；B 组前景更接近杯状物，圆环变为细圈；C 组前景继续圆化，圆环消失，背景还出现额外物体。深度约束能保持整体布局，不能保证棱边、孔洞和物体类别。前景深度局部接近纯白，加上圆环遮挡较多，细部几何约束不足，模型先验会补出不同结构。',
    '__DEFECTS__': 'A 组材质偏棕、球体偏暗；B 组出现明显水平细纹，立方体的杯状变形未随步数增加而修复；C 组材质自由度更大，却新增背景灯状结构。深度图缺少材质和棱面朝向信息，通用提示词也未明确四个几何体，因此模型会自行解释表面。可优先改善圆环可见性与深度层次，再补充明确的物体描述；增加步数不能替代控制信息。',
    '__CONCLUSION__': f'我依据指南确定实验目标，Codex 协助补全脚本、运行实验和整理数据。已完成三图导出与三组纯 CPU 推理。4 步和 8 步的采样耗时比为 {b["sampling_seconds"]/a["sampling_seconds"]:.2f}，总耗时比为 {b["total_seconds"]/a["total_seconds"]:.2f}，固定开销使总耗时没有翻倍。当前 0.9 控制强度更适合保持布局，但局部几何仍有明显失败；应先改善控制图和构图，再调整生成参数。',
    '__CONTROL__': 'A 与 C 均为 4 步、种子 42。强度 0.9 时圆环仍有残留，背景结构较简单；0.5 时圆环消失并新增背景物体，说明几何约束减弱。两组采样为 39.89 s 和 40.92 s，降低强度不等于减少 ControlNet 计算。本次是文生图，没有初始图；若采用 img2img，需同时调节初图去噪强度，去噪越低通常越能保留初图。',
}
for t in [11, 12, 14, 17]:
    for row in tables[t].rows:
        for target in row.cells:
            for p in target.paragraphs:
                if p.text in analysis:
                    text(p, analysis[p.text])
for t in [9, 10]:
    for row in tables[t].rows:
        for target in row.cells:
            for p in target.paragraphs:
                p.paragraph_format.keep_with_next = True
    tables[t].rows[-1].cells[0].paragraphs[-1].paragraph_format.keep_with_next = False

output = ROOT / 'report/实验作业七_实验报告.docx'
temp = ROOT / 'tmp/report_working.docx'
output.parent.mkdir(exist_ok=True)
temp.parent.mkdir(exist_ok=True)
doc.save(temp)
# Preserve all unchanged source package parts byte-for-byte.
editable = {'word/document.xml', 'word/_rels/document.xml.rels', '[Content_Types].xml'}
with ZipFile(template) as base, ZipFile(temp) as built, ZipFile(output, 'w', ZIP_DEFLATED) as final:
    for info in base.infolist():
        final.writestr(info, built.read(info.filename) if info.filename in editable else base.read(info.filename))
    for name in built.namelist():
        if name not in base.namelist():
            final.writestr(name, built.read(name))
print(output)
