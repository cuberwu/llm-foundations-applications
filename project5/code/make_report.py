"""使用 bundled Python；保留样板的页面、样式和表格构件。"""
import copy
import io
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.table import Table

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'results'
REF = next((ROOT/'docs').glob('*.docx'))
OUT = ROOT/'report/实验作业五_实验报告.docx'
SOURCE_HASH = '25700737da7fe8d41d6729736916540275d69a2649f6176353ea3315727a4992'
assert sha256(REF.read_bytes()).hexdigest() == SOURCE_HASH
def read(n): return json.loads((R/n).read_text(encoding='utf-8'))

env, params, summary = read('environment.json'), read('parameters.json'), read('summary.json')
stages = {k: read(k+'_eval.json') for k in summary}
training = {k: read(k+'_training.json') for k in ['sft', 'dpo', 'dpo17']}
pairs, checks, extra = read('preference_pairs.json'), read('dpo_checks.json'), read('extra_eval.json')
d = Document(REF)
table_source = copy.deepcopy(d.tables[0]._tbl)
for child in list(d._element.body):
    if child.tag != qn('w:sectPr'): d._element.body.remove(child)
WIDTH = (d.sections[0].page_width-d.sections[0].left_margin-d.sections[0].right_margin)/914400


def font(run, size=10.5, bold=False):
    run.font.name = 'Arial'; run.font.size = Pt(size); run.font.bold = bold
    run.font.color.rgb = RGBColor(0,0,0)
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')


def para(text='', style='Normal', size=10.5, bold=False, after=5, before=0):
    p = d.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(after); p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.line_spacing = Pt(size+3.5)
    p.paragraph_format.keep_with_next = False
    grid=OxmlElement('w:snapToGrid'); grid.set(qn('w:val'),'0'); p._p.get_or_add_pPr().append(grid)
    font(p.add_run(text), size, bold)
    return p


def heading(text, page=False, sub=False):
    p = para(text, 'Heading 2' if sub else 'Heading 1', 12 if sub else 14, True, 5, 7)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.page_break_before = page
    return p


def caption(text):
    p = para(text, size=9.5, bold=True, after=4, before=5)
    p.paragraph_format.keep_with_next = True


def table(headers, rows, fractions, size=9.5):
    el = copy.deepcopy(table_source)
    # Clone source properties, but generate the requested grid and cells.
    for tr in list(el.findall(qn('w:tr'))): el.remove(tr)
    grid = el.find(qn('w:tblGrid'))
    for c in list(grid): grid.remove(c)
    for f in fractions:
        c=OxmlElement('w:gridCol'); c.set(qn('w:w'), str(round(WIDTH*f*1440))); grid.append(c)
    d._element.body.insert(len(d._element.body)-1, el)
    t=Table(el, d._body); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    pr=el.find(qn('w:tblPr'))
    w=pr.find(qn('w:tblW')); w.set(qn('w:w'),str(round(WIDTH*1440))); w.set(qn('w:type'),'dxa')
    for ri, values in enumerate([headers]+rows):
        row=t.add_row()
        rp=row._tr.get_or_add_trPr(); rp.append(OxmlElement('w:cantSplit'))
        if ri == 0: rp.append(OxmlElement('w:tblHeader'))
        for i, value in enumerate(values):
            c=row.cells[i]; c.width=Inches(WIDTH*fractions[i]); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p=c.paragraphs[0]; p.paragraph_format.space_after=Pt(2); p.paragraph_format.space_before=Pt(2)
            p.paragraph_format.line_spacing=Pt(size+2.5); p.paragraph_format.keep_with_next=False
            grid=OxmlElement('w:snapToGrid'); grid.set(qn('w:val'),'0'); p._p.get_or_add_pPr().append(grid)
            font(p.add_run(str(value)),size,ri==0)
            cp=c._tc.get_or_add_tcPr()
            borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'A6A6A6');borders.append(e)
            cp.append(borders)
            sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'D9E2F3' if ri==0 else 'FFFFFF');cp.append(sh)
            mar=OxmlElement('w:tcMar')
            for edge,v in [('top','35'),('bottom','35'),('left','65'),('right','65')]:
                e=OxmlElement('w:'+edge);e.set(qn('w:w'),v);e.set(qn('w:type'),'dxa');mar.append(e)
            cp.append(mar)
    return t


def picture(name, width=WIDTH):
    p=para(after=3)
    p.paragraph_format.line_spacing=1.0
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(R/name),width=Inches(width))


def answer(tag,pid): return next(r['pred'] for r in stages[tag]['records'] if r['id']==pid)


p=para('SmolVLM 256M QLoRA与DPO实验报告','Title',16,True,after=7)
p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p._p.get_or_add_pPr().append(OxmlElement('w:pBdr'))
para('姓名：                    学号：                    日期：2026年11月11日',size=10,after=7)

heading('一 实验概述')
para('按指南完成15道几何图形问答的基线推理、60步QLoRA SFT、40步DPO，并加入5组偏好对重新训练。SFT正确率由40.0%升至73.3%，DPO后为60.0%；偏好损失下降与答题正确率并不一致。')
table(['项目','实际配置'],[
    ['硬件','i9-13900HX；RTX 5070 Ti Laptop 12GB；Windows'],
    ['软件','Python 3.10.19；torch 2.11.0.dev20260127+cu128；transformers 4.57.6；peft 0.18.1；bitsandbytes 0.50.2'],
    ['模型与数据','SmolVLM-256M-Instruct；版本7e3e67ed；15张256×256合成图；随机种子42'],
],[.18,.82])
para(f"实测耗时：基线15题 {stages['baseline']['seconds']:.1f}s；SFT 60步 {training['sft']['seconds']:.1f}s；DPO 40步 {training['dpo']['seconds']:.1f}s；新增偏好对DPO 40步 {training['dpo17']['seconds']:.1f}s。训练耗时含适配器保存，DPO含参考分数预计算，不含模型加载与评测。",size=9.5,before=4)

heading('二 QLoRA与SFT实现')
para('基座以4bit NF4和双重量化加载，计算使用fp32，439个Linear4bit模块实际生效。q/k/v/o投影接入LoRA，r=8、α=16、dropout=0.05；冻结基座，仅更新适配器，使用AdamW。',size=10)
caption('图 1 可训练参数统计输出')
picture('parameters.png')
para('可训练参数1,363,968／总参数257,848,896，占0.52898%。LoRA用ΔW=BA限制更新的秩，少量参数仍可改变注意力投影和输出行为。',size=10)
para('SFT使用15条标准答案，batch=1、lr=1e-4，循环打乱数据直至恰好60步。labels对图像与问题前缀置-100，先断言完整输入的前缀token与单独prompt一致，再仅对答案计算交叉熵。15条掩码检查均通过。',size=10)

heading('三 对比实验设计')
para('三阶段使用同一NF4基座、同一15题、同一图像预处理与贪心解码，最多生成24个token；图像最长边统一设512，避免默认放大到2048带来的开销。DPO的12对与17对实验均从同一SFT适配器开始，β=0.1、lr=5e-5、40步。',size=10)
para('判分统一识别中英文数字、圆形／方形与Yes／No；否定词优先，数字或形状互相矛盾时不计对。指南的纯中文子串匹配会漏掉英文，并将“没有”误匹配为“有”；原口径另存于JSON与CSV。训练与评测复用同一题集，结果仅说明固定题集上的行为变化。',size=10)

heading('四 结果与数据记录',page=True)
caption('表 1 三阶段与新增偏好对的正确率')
names={'baseline':'基线','sft':'SFT 60步','dpo':'SFT+DPO 12对','dpo17':'SFT+DPO 17对'}
table(['阶段','count /5','attr /5','exist /5','总计 /15','正确率'],[
    [names[k],v['count'],v['attr'],v['exist'],v['total'],f"{v['accuracy']:.1%}"] for k,v in summary.items()
],[.30,.13,.13,.13,.15,.16])
caption('表 2 内置15题的逐题判定')
table(['题号','类型','标准答案','基线','SFT','DPO 12对','DPO 17对'],[
    [r['id'],r['type'],r['gold']]+['对' if stages[k]['records'][i]['ok'] else '错' for k in summary]
    for i,r in enumerate(stages['baseline']['records'])
],[.10,.14,.16,.12,.12,.18,.18],size=9)
caption('表 3 三题在三阶段的完整输出')
table(['题号','基线','SFT','DPO 12对'],[
    [pid,answer('baseline',pid),answer('sft',pid),answer('dpo',pid)] for pid in ['P01','P06','P12']
],[.09,.49,.20,.22],size=9)
para('P01数红圆，P06问蓝色图形的形状，P12问是否有红圆。全部逐题原文见results/comparison.csv；基线按指南原中文匹配为0/15，按统一中英文规则为6/15。',size=9,after=4,before=4)

heading('五 结果分析',page=True)
para('SFT的count由1/5升至4/5，提升最明显；attr由3/5升至4/5，exist由2/5升至3/5。输出从英文泛化描述转为短答案，但P04仍把4个黄圆数成2个，P14、P15仍对不存在的目标答“有”。格式学习并不等于视觉事实全部掌握。')
para('DPO使count达到5/5，却令attr降至3/5、exist降至1/5，总计由11/15降至9/15。12对实验前8步与末8步的平均loss为0.508→0.161，margin为0.578→2.331；这只证明相对偏好目标更容易满足，不能替代任务评测。')
para('实际B10（P12）中，chosen的log概率由-1.591降至-5.035，rejected由-34.559降至-61.960，margin仍从0.506升到2.902。错误回答下降得更多，足以降低DPO损失，但正确答案的绝对概率也下降，最终生成“号称。”。这与偏好优化退化一致；单次小样本实验不能把原因仅归结为β。')

heading('六 失败案例与环境踩坑记录')
caption('表 4 真实失败与忠实度问题')
table(['题号与阶段','输出原文','核对结果'],[
    ['P02 基线',answer('baseline','P02'),'实际3个绿圆，回答2个，计数错误。'],
    ['P04 基线',answer('baseline','P04'),'实际4个黄圆，回答2个，计数错误。'],
    ['P11 基线',answer('baseline','P11'),'实际为黄方形图，却声称网页截图。'],
    ['P12 基线',answer('baseline','P12'),'不存在红圆，却答Yes并转述蓝方形。'],
    ['P06 基线',answer('baseline','P06'),'答出square，但虚构方形嵌套关系。'],
    ['P10 基线',answer('baseline','P10'),'答出square，但说总计3个；实际3蓝方+1红方。'],
],[.17,.48,.35],size=9)
para('前四例是答案错误；后两例是目标形状答对但补充描述失真。DPO后P07把红圆答成“Square.”，P12从SFT的“没有。”退化为“号称。”，分别属于形状幻觉与答题退化。',size=9.5,before=4)
para('环境处理：为project5建立独立环境并复用已缓存模型；安装依赖时避免覆盖已有CUDA torch，验证cuda可用。参考代码中有限epoch导致步数不足、提前return漏存权重的问题均改为按steps循环并统一保存；DPO不再嵌套PeftModel。',size=9.5)

heading('七 AI协作记录')
para('我提出按指南完成实验、控制实现规模并保留验证依据；AI据此整理单文件训练脚本、执行实验和填写报告。协作核对重点是labels掩码、disable_adapter参考路径、DPO符号和否定句判分。手算、梯度方向检查及逐题输出共同用于复核，报告中的曲线和数值均来自本次日志。',size=10)
para('本次协作工具为Codex；对应指南中Builder负责骨架与重复代码、Chat辅助解释和调试的分工。',size=9.5)

heading('八 思考题',page=True)
for title, text in [
    ('题 1 SFT与诚实拒绝','SFT会提高P12“没有”的概率，也会通过softmax间接压低其他词；但它没有显式比较“没有”与特定幻觉回答“有，左上角有红圆”。DPO增加这种对照。不过本次P12从SFT正确变为DPO答非所问，说明负例并不保证更诚实。'),
    ('题 2 手算与margin含义','m=0.1×[(1.5−1.0)−(0.2−0.6)]=0.09；L=ln(1+exp(−0.09))=0.649159，CLI复核一致。m<0表示策略相对参考模型的chosen/rejected对数优势变差，不等于chosen的绝对概率一定低于rejected。'),
    ('题 3 为什么无需奖励模型','KL正则目标的最优解满足π*(y|x)∝πref(y|x)exp(r/β)，故r=βlog(π*/πref)+βlogZ(x)。同一问题下的奖励相减时logZ抵消，代入Bradley-Terry概率即可直接优化策略的二元偏好损失，无需另训奖励网络。[1]'),
    ('题 4 量化基座与训练适配器','NF4主要节省冻结基座的存储；低比特离散权重不宜直接按普通浮点参数更新，全参训练还需梯度和优化器状态。LoRA保留小型浮点增量来更新。本次仅训练0.529%；r=16时适配器参数约翻倍为2,727,936，占比约1.052%，接近2倍。[2]'),
    ('题 5 自动构造偏好对','多次采样加规则打分可减少人工成本，并覆盖模型真实错误；风险是规则误判、同源模型偏差和对评分规则投机。应验证图中事实、加入困难负例并保留独立题集；不能仅因DPO loss下降就继续加步数。'),
    ('题 6 最易写反的位置及验证','最容易写反chosen/rejected差或忘记损失前的负号。本次检查提高pol_w应降低loss、提高pol_l应增加loss，反向梯度符号也通过。15条样本检查prompt前缀与答案掩码；关闭不同适配器后同一参考log概率均为-26.086069，验证参考未随策略漂移。'),
]:
    heading(title,sub=True); para(text,size=10,after=5)

heading('九 DPO偏好对齐实验',page=True)
heading('9.1 偏好数据与新增样例',sub=True)
para('保留指南12组chosen/rejected，覆盖数错、形状幻觉和对不存在的对象硬答“有”；新增以下5组，全部图像与答案由场景规格核对。',size=10)
caption('表 5 新增偏好对')
table(['编号与问题','chosen','rejected','失败模式'],[
    [p['id']+' '+p['prompt'],p['chosen'],p['rejected'],p['failure']] for p in pairs['extra']
],[.31,.12,.35,.22],size=9)
heading('9.2 DPO实现与手算复核',sub=True)
para('m=β[(logπθ(w)−logπref(w))−(logπθ(l)−logπref(l))]；L=−logσ(m)=ln(1+exp(−m))。序列log概率对答案token求和并作因果位移。参考取关闭适配器的冻结基座，以no_grad预计算；DPO关闭dropout但保留梯度。',size=10)
caption('表 6 手算与CLI复核结果')
table(['代入 pw rw pl rl β','margin计算','loss'],[
    ['1.2, 0.8, 0.9, 0.6, 0.1','0.1×(0.4−0.3)=0.01',f"{checks[0]['loss']:.6f}"],
    ['−1, −2, −3, −2, 0.1','0.1×(1−(−1))=0.2',f"{checks[1]['loss']:.6f}"],
    ['−3, −2, −1, −2, 0.1','0.1×(−1−1)=−0.2',f"{checks[2]['loss']:.6f}"],
],[.43,.38,.19],size=9)
para('pw/rw/pl/rl依次为policy/reference的chosen/rejected序列log概率。每行以L=ln(1+exp(−m))计算；四组CLI输出见dpo_check_cli.log。指南正数仅作代数示例，真实log概率非正；m=0.01的精确loss是0.688160，非样板的0.693。',size=9,before=4)
caption('图 2 实测SFT loss及DPO loss与margin曲线')
picture('training_curves.png')
para('曲线保留逐步值；不同步抽到的偏好对不同，波动不代表同一样本恶化。12对与17对的末8步平均loss分别为0.161、0.166。',size=9)

heading('9.3 忠实度评分与新增数据对照',page=True,sub=True)
para('按1至5分量表逐条复核P01、P06、P12：5分为完全符合图像，4分为核心正确但表述轻微含混，3分为部分事实正确，2分为明显错误，1分为无关或主要事实错误。评分由AI辅助复核，未采集独立多人评价。',size=10)
caption('表 7 三阶段忠实度评分')
scores=[['P01',1,5,5,'基线对象错误；SFT与DPO均数对2个红圆。'],['P06',3,2,5,'基线形状对但有嵌套幻觉；SFT“图形”未回答形状；DPO正确。'],['P12',2,5,1,'基线错误肯定；SFT正确否定；DPO“号称”无关。']]
table(['题号','基线','SFT','DPO','评分依据'],scores,[.10,.10,.10,.10,.60],size=9)
para('均分为2.00→4.00→3.67；DPO相对SFT为1胜、1平、1负，胜率1/3，平局计半为50%。仅3题，不能据此宣称总体偏好提升。',size=9.5,before=4)
caption('表 8 新增5题在训练前后的输出')
table(['题号','SFT','DPO 12对','DPO 17对'],[
    [p['id']]+[extra[k][i]['pred'] for k in ['sft','dpo','dpo17']] for i,p in enumerate(pairs['extra'])
],[.10,.36,.27,.27],size=9)
para('新增5题正确数为3/5→2/5→2/5；加入17对训练后，N02仍答“1.”、N04仍答“一。”，未解决形状绑定和存在性判断。N05虽未生成负例中的小鸟，却把圆形改成方形，不能视为纠错成功。内置15题仍为9/15，新增数据没有带来可测提升。',size=10,before=4)
para('复现：python code/hw5_dpo.py --mode all；python code/hw5_dpo.py --mode dpo --extra；python code/verify_experiment.py。日志、适配器、逐题CSV和图像保存在results/。β与秩扫描为指南选做，本次未扩展。',size=9.5)
heading('参考资料',sub=True)
para('[1] Rafailov等，Direct Preference Optimization，式4至7。https://arxiv.org/abs/2305.18290',size=9)
para('[2] Dettmers等，QLoRA。https://arxiv.org/abs/2305.14314',size=9)
para('[3] 作业五实验指南与实验报告样板；bitsandbytes安装说明 https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation',size=9)

# python-docx仅用于编辑正文；重封装时保留所有无关模板part的原始字节。
buffer=io.BytesIO(); d.save(buffer)
changed={'word/document.xml','word/_rels/document.xml.rels','[Content_Types].xml'}
OUT.parent.mkdir(parents=True,exist_ok=True)
with ZipFile(REF) as original, ZipFile(io.BytesIO(buffer.getvalue())) as built, ZipFile(OUT,'w') as out:
    for info in original.infolist():
        out.writestr(info, built.read(info.filename) if info.filename in changed else original.read(info.filename))
    for info in built.infolist():
        if info.filename not in original.namelist(): out.writestr(info,built.read(info.filename))
with ZipFile(REF) as original, ZipFile(OUT) as final:
    for name in original.namelist():
        if name not in changed: assert original.read(name)==final.read(name),name
assert sha256(REF.read_bytes()).hexdigest()==SOURCE_HASH
final=Document(OUT)
assert len(final.sections)==1 and final.sections[0].page_width==d.sections[0].page_width
alltext='\n'.join(p.text for p in final.paragraphs)+'\n'+'\n'.join(c.text for t in final.tables for row in t.rows for c in row.cells)
assert '〔填写〕' not in alltext and 'CPU 实测' not in alltext
print(OUT)
