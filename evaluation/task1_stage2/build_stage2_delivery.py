"""Build the report and standalone figures from verified Stage 2 results."""
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '最终交付'
FIG = OUT / '图表'
JUDGE = ROOT / 'outputs/judge_four_model'
MODELS = ('V1', 'V2', 'V3.2', 'V4')
FONT = Path('C:/Windows/Fonts/simhei.ttf')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.reader(f))[1:]


def font(size):
    return ImageFont.truetype(str(FONT), size)


def panels(path, title, panels_data, domain, suffix=''):
    picture = Image.new('RGB', (1440, 750), 'white')
    draw = ImageDraw.Draw(picture)
    draw.text((46, 24), title, font=font(37), fill='black')
    for i, (name, values) in enumerate(panels_data):
        x = 50 + i % 2 * 705
        y = 100 + i // 2 * 320
        draw.text((x, y), name, font=font(29), fill='black')
        max_value = domain[i]
        for j, (label, score) in enumerate(zip(MODELS, values)):
            top = y + 58 + 53 * j
            draw.text((x, top), label, font=font(25), fill='black')
            width = max(1, round(440 * score / max_value))
            draw.rectangle((x + 93, top, x + 93 + width, top + 27),
                           fill=('#202020', '#535353', '#818181', '#b0b0b0')[j], outline='black')
            draw.text((x + 104 + width, top),
                      f'{score:.2f}{suffix}' if max_value >= 5 else f'{score:.3f}',
                      font=font(23), fill='black')
    picture.save(path)


def wireframe(path):
    image = Image.new('RGB', (1320, 660), 'white')
    d = ImageDraw.Draw(image)
    d.text((40, 25), 'FoodLens 食物图像分析', font=font(39), fill='black')
    blocks = [
        ((40, 100, 435, 458), '上传图片', ['选择一张 RGB 食物照片', '图片预览', '状态：待分析']),
        ((455, 100, 1280, 245), '图像描述', ['模型版本 · 解码方式', '生成的 Caption']),
        ((455, 260, 1280, 458), '食材与营养预测', ['食材名称与分量（g）', '总重、热量、蛋白质、脂肪、碳水']),
        ((40, 478, 650, 620), '视觉 Judge', ['四项分数 · 理由 · 运行状态']),
        ((670, 478, 1280, 620), '词级注意力', ['选择生成词，显示对应热图']),
    ]
    for (x, y, right, bottom), title, lines in blocks:
        d.rectangle((x, y, right, bottom), outline='black', width=3)
        d.text((x + 20, y + 15), title, font=font(31), fill='black')
        for i, line in enumerate(lines):
            d.text((x + 20, y + 65 + i * 39), line, font=font(24), fill='black')
    image.save(path)


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    for cell, value in zip(t.rows[0].cells, headers):
        cell.paragraphs[0].add_run(value).bold = True
    for row in rows:
        for cell, value in zip(t.add_row().cells, row):
            cell.text = str(value)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(3)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(10)


def picture(doc, path, caption, size=6.15):
    doc.add_picture(str(path), width=Inches(size))
    p = doc.paragraphs[-1]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.keep_with_next = True
    p = doc.add_paragraph(caption)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    test = json.loads((ROOT / 'outputs/v1_v4_test/v1_v2_reevaluation.json').read_text(encoding='utf-8'))
    results = test['results']
    judge = csv_rows(JUDGE / '正式盲评_四模型均分.csv')
    ingredients = csv_rows(ROOT / 'task2_same_image/ingredient_summary.csv')
    assert len(results) == len(judge) == len(ingredients) == 4 and test['reference_count'] == 4938
    panels(FIG / '四模型测试集比较.png', '统一测试集指标（4,938 张）', [
        ('BLEU-1（%）', [r['bleu1_to_4'][0] for r in results]),
        ('BLEU-4（%）', [r['bleu1_to_4'][3] for r in results]),
        ('ROUGE-L', [r['rouge_l'] for r in results]),
        ('CIDEr-D', [r['cider_d'] for r in results]),
    ], [40, 6, .32, .45])
    panels(FIG / '四模型视觉盲评比较.png', '视觉盲评四项均分（每个模型 30 张图）', [
        (name, [float(row[i + 2]) for row in judge])
        for i, name in enumerate(('视觉一致性', '食物识别准确性', '细节准确性', '语言质量'))
    ], [5] * 4)
    wireframe(FIG / 'Web展示布局草图.png')

    doc = Document()
    doc.core_properties.author = '李馨瑶'
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.top_margin = sec.bottom_margin = Inches(.78)
    sec.left_margin = sec.right_margin = Inches(.85)
    for name in ('Normal', 'Title', 'Heading 1', 'Heading 2'):
        style = doc.styles[name]
        style.font.name = '宋体' if name == 'Normal' else '黑体'
        style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), style.font.name)
        style.font.color.rgb = RGBColor(0, 0, 0)
    doc.styles['Normal'].font.size = Pt(11)
    doc.styles['Normal'].paragraph_format.space_after = Pt(7)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.2
    title = doc.add_heading('FoodLens 统一评测与可信分析', 0)
    borders = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'nil')
    borders.append(bottom)
    title._p.get_or_add_pPr().append(borders)
    doc.add_paragraph('李馨瑶｜阶段 2 组内报告章节')
    doc.add_heading('1  四模型统一测试', 1)
    doc.add_paragraph('使用同一组 4,938 张测试图逐条核对预测和参考描述。四个模型的主表统一采用 Greedy 解码。英文描述转小写后按空格切词，保留标点，每图使用一条参考描述。BLEU 为组内语料级实现，ROUGE-L 和 CIDEr-D 采用组内本地实现。')
    table(doc, ['模型', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L', 'CIDEr-D'], [
        [name, *(f'{v:.2f}' for v in r['bleu1_to_4']), f"{r['rouge_l']:.4f}", f"{r['cider_d']:.4f}"]
        for name, r in zip(MODELS, results)])
    picture(doc, FIG/'四模型测试集比较.png', '图 1  四模型测试集指标')
    doc.add_paragraph('V2 的测试集 BLEU-4 为 5.2914%，比同设置重训的 V1 高 0.4102 个百分点。V3.2 采用 layer4 微调权重，作为 V3 的代表。V3 Baseline 和 V3.2 的验证集 BLEU-4 分别为 3.1558% 和 3.8173%。组长早期提供的 V1 测试 BLEU-4 为 4.5702%，缺少对应权重，因此另存作历史记录。')

    doc.add_page_break()
    doc.add_heading('2  多模态 Judge 与独立复评', 1)
    doc.add_paragraph('DeepSeek V4.1 Flash 1017（OpenCode Go，Max）查看原图并参考文字，对 30 张图的 120 条匿名描述评分。四个维度各为 1 至 5 分。正式评分时，每条候选由独立子 Agent 评估。')
    table(doc, ['模型', '样本数', '视觉一致性', '食物识别', '细节准确性', '语言质量'],
          [[name, row[1], *(f'{float(v):.2f}' for v in row[2:])] for name, row in zip(MODELS, judge)])
    picture(doc, FIG/'四模型视觉盲评比较.png', '图 2  DeepSeek 对 30 张图的平均评分')
    doc.add_paragraph('V4 在本批样本上的食物识别与细节准确性均分最高。V3.2 的语言质量均分最高。')
    doc.add_paragraph('另用同版本的独立子 Agent 复评 5 张图，共 20 条候选。以种子 20261116 抽取 08、23、14，并加入问题图 03、19。复评时每个子 Agent 看同图四条候选，未获知正式分数或候选模型身份。80 个分数对中 54 个完全一致（67.5%），78 个相差至多 1 分（97.5%）。两处分差为 2 的案例是 14C 的视觉一致性（1 与 3）及 19A 的食物识别（5 与 3）。分歧涉及容器误写的扣分力度，以及食物识别是否计入配料遗漏。')
    doc.add_paragraph('经组长确认，本次子 Agent 复评作为组内交付的复核环节。结果反映该 Judge 对五张抽样图的两轮一致性，评分与看图依据均来自子 Agent。')

    doc.add_page_break()
    doc.add_heading('3  Task2 同图分析', 1)
    doc.add_paragraph('使用 Task2 模型 B（CLIP + LoRA）在同一批 30 张原图上生成营养 JSON。按 file_name 将每条 JSON 与四条 Caption 对齐，共得到 120 条对照。用 Task2 食材词表提取候选描述中的词或短语，兼容部分单复数，比较两侧的食材集合。')
    table(doc, ['Caption 模型', '图片数', '描述命中词表', '至少有一项重合'],
          [[name, row[1], row[3], row[2]] for name, row in zip(MODELS, ingredients)])
    doc.add_paragraph('03A 的描述写 fish steak，程序将 steak 计为重合，但这个词在句中指鱼排，词面命中不足以确认食材一致。19 号图的营养 JSON 有 13 个食材条目，其中没有米饭、虾或贝类。08 号图的 JSON 输出 parmesan cheese，但未列出蘑菇。词表匹配适合筛查案例，重合数量不能直接用来衡量营养估计精度或幻觉率。')
    doc.add_page_break()
    doc.add_heading('4  注意力示例与 Web 展示草图', 1)
    doc.add_paragraph('词级热图展示 V2 在生成各词时的注意力权重，可辅助查看模型关注的区域。三组热图及样例记录随交付包提供。')
    picture(doc, ROOT/'attention_examples/sample_0190/attention.png', '图 3  V2 词级注意力示例', size=4.3)
    picture(doc, FIG/'Web展示布局草图.png', '图 4  图片上传后各模块的展示布局', size=5.55)
    doc.add_paragraph('展示区按模块返回图片预览、描述、营养预测、Judge 和注意力结果。各模块分别显示运行状态，接口字段见随附的展示说明。')
    doc.save(OUT/'FoodLens阶段2报告_李馨瑶.docx')


if __name__ == '__main__':
    main()
