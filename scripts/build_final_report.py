"""Build the student's final Word report from a sanitized reference and frozen logs.

Use the bundled document Python runtime (python-docx, Pillow, lxml). No training,
new test inference, experiment-log edits, or Word automation occurs here.
Pagination/field refresh and render QA are a separate Microsoft Word step.
"""
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import csv
import hashlib
import json

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree as E
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'scripts/templates/van_lang_report_base.docx'
DEFAULT = ROOT / 'report/Bao_cao_Do_an_Toi_uu_Random_Forest_Le_Quyet_Tien_2026-11-27.docx'
FIG = ROOT / 'figures/report_2026-11-27'
DATE = '27/11/2026'
AUTHOR = 'Lê Quyết Tiến'
STUDENT_ID = '2474802010386'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
PRESERVE = ['word/styles.xml', 'word/numbering.xml', 'word/fontTable.xml',
            'word/theme/theme1.xml', 'word/header1.xml', 'word/webSettings.xml',
            'word/media/image1.png', 'word/media/image2.svg', 'word/media/image3.png']


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def rows(path):
    with (ROOT / path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def fnum(x, n=5):
    return f'{float(x):.{n}f}'.replace('.', ',')


def pct(x, n=2):
    return fnum(float(x) * 100, n) + '%'


def xml(x):
    return E.tostring(x, xml_declaration=True, encoding='UTF-8', standalone=True)


def make_figures():
    """Only visualize existing development samples; never load a fitted estimator."""
    FIG.mkdir(exist_ok=True)
    font = lambda size: ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)
    data = load('web/dist/data/dashboard.json')
    samples = [next(s for s in data['samples'] if s['label'] == i) for i in range(10)]
    im = Image.new('RGB', (1600, 690), 'white')
    d = ImageDraw.Draw(im)
    d.text((800, 38), 'Ảnh Digits 8 × 8 từ tập development', font=font(35), fill='black', anchor='mm')
    for i, s in enumerate(samples):
        x, y = 70 + i % 5 * 310, 85 + i // 5 * 290
        pix = Image.new('L', (8, 8))
        pix.putdata([round(v / 16 * 255) for v in s['pixels']])
        im.paste(pix.resize((210, 210), Image.Resampling.NEAREST).convert('RGB'), (x, y))
        d.text((x + 105, y + 235), f"Nhãn {s['label']} · chỉ số {s['index']}", font=font(26), fill='black', anchor='mm')
    im.save(FIG / 'digits_examples.png')
    im = Image.new('RGB', (1600, 1000), 'white'); d = ImageDraw.Draw(im)
    def box(rect, lines, color='#edf2f5'):
        d.rounded_rectangle(rect, radius=15, outline='#52606b', width=3, fill=color)
        cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        for j, line in enumerate(lines):
            d.text((cx, cy + (j - (len(lines)-1)/2) * 43), line, font=font(31), fill='black', anchor='mm')
    def arrow(points):
        d.line(points, fill='#52606b', width=5)
        x,y=points[-1]; px,py=points[-2]
        if y>py: d.polygon([(x,y),(x-12,y-23),(x+12,y-23)],fill='#52606b')
        elif x>px:d.polygon([(x,y),(x-23,y-12),(x-23,y+12)],fill='#52606b')
    box((450, 20, 1150, 130), ['Digits 1.797 ảnh · 64 đặc trưng'])
    box((100, 220, 1000, 345), ['Development 1.437 mẫu', '3 folds cố định · seed 42'])
    box((1130, 220, 1550, 345), ['Test 360 mẫu', 'Giữ riêng khi chọn'])
    arrow([(800,130),(800,170),(550,170),(550,220)])
    arrow([(800,130),(800,170),(1340,170),(1340,220)])
    box((100, 420, 1000, 560), ['Baseline + Random Search + TPE', '30 trials × 2 phương pháp × 3 seeds'])
    arrow([(550,345),(550,420)])
    box((100, 635, 1000, 775), ['Khóa lựa chọn theo CV và epsilon', 'Nhỏ nhất với F1 ≥ F1 tốt nhất − 0,005'])
    arrow([(550,560),(550,635)])
    box((390, 855, 1430, 975), ['Refit development → đánh giá test → lưu minh chứng'])
    arrow([(550,775),(550,855)])
    arrow([(1340,345),(1340,855)])
    im.save(FIG / 'experiment_pipeline.png')


class Report:
    def __init__(self):
        self.doc = Document(BASE)
        self.score = deepcopy(self.doc.tables[0]._tbl)
        self.student = deepcopy(self.doc.tables[1]._tbl)
        self.sections = [deepcopy(s) for s in self.doc._element.xpath('.//w:sectPr')]
        body = self.doc._element.body
        for e in list(body)[24:]: body.remove(e)
        body.append(deepcopy(self.sections[-1]))
        self.nav, self.figures, self.tables = [], [], []
        self.uid = 1000
        self.contents = []
        self.source_paths = set()
        self.doc.core_properties.title = 'Tối ưu siêu tham số Random Forest cho nhận dạng chữ số viết tay'
        self.doc.core_properties.subject = 'Báo cáo đồ án cá nhân môn Nhập môn tối ưu hóa'
        self.doc.core_properties.author = AUTHOR
        self.doc.core_properties.last_modified_by = AUTHOR
        self.doc.core_properties.keywords = 'Digits, Random Forest, Random Search, TPE, tối ưu hóa'
        self.doc.core_properties.comments = 'Báo cáo tổng hợp từ kết quả thực nghiệm được lưu trong repository.'

    def p(self, text='', style=None, **kw):
        p = self.doc.add_paragraph(text, style=style)
        if kw.get('center'): p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if kw.get('left'): p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if kw.get('italic'):
            for r in p.runs:r.italic = True
        if kw.get('size'):
            for r in p.runs:r.font.size = Pt(kw['size'])
        if kw.get('keep'): p.paragraph_format.keep_with_next = True
        return p

    def bookmark(self, p, prefix='b'):
        self.uid += 1
        name = f'{prefix}{self.uid}'
        start = OxmlElement('w:bookmarkStart');start.set(qn('w:id'),str(self.uid));start.set(qn('w:name'),name)
        end = OxmlElement('w:bookmarkEnd');end.set(qn('w:id'),str(self.uid))
        p._p.insert(1 if p._p.pPr is not None else 0,start);p._p.append(end)
        return name

    def h(self, text, level=1, page=False):
        p=self.p(text,style=f'Heading {level}')
        pp=p._p.get_or_add_pPr();num=OxmlElement('w:numPr');n=OxmlElement('w:numId');n.set(qn('w:val'),'0');num.append(n);pp.append(num)
        if page:p.paragraph_format.page_break_before = True
        self.nav.append((text,level,self.bookmark(p,'h')))
        return p

    def page(self, text, level=2):
        return self.h(text,level,page=True)

    def note(self,text):
        p=self.p(text,center=True,italic=True,size=11)
        p.paragraph_format.line_spacing=1.1
        return p

    def caption(self, text, typ):
        p=self.p(text,style='Ảnh' if typ=='figure' else 'Bảng')
        p.paragraph_format.keep_with_next=typ=='table'
        name=self.bookmark(p,'f' if typ=='figure' else 't')
        (self.figures if typ=='figure' else self.tables).append((text,1,name))
        return p

    def figure(self,path,caption,source,width=15.8):
        path=Path(path); full=ROOT/path
        self.source_paths.add(path.as_posix())
        p=self.p(center=True,keep=True)
        p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(3)
        shape=p.add_run().add_picture(str(full),width=Cm(width))
        shape._inline.docPr.set('descr',caption)
        self.caption(caption,'figure');self.note('Nguồn: '+source)

    def table(self,caption,headers,data,widths=None,align_left=(0,),size=11.5):
        self.caption(caption,'table')
        t=self.doc.add_table(rows=1,cols=len(headers));t.autofit=False
        widths=widths or [16/len(headers)]*len(headers)
        for col,w in zip(t.columns,widths):col.width=Cm(w)
        pr=t._tbl.tblPr
        borders=OxmlElement('w:tblBorders')
        for side in ['top','bottom','left','right','insideH','insideV']:
            el=OxmlElement('w:'+side)
            for k,v in {'val':'single','sz':'4','color':'D9D9D9'}.items():el.set(qn('w:'+k),v)
            borders.append(el)
        pr.append(borders)
        for i,h in enumerate(headers):t.rows[0].cells[i].text=str(h)
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for values in data:
            for c,v in zip(t.add_row().cells,values):c.text=str(v)
        for ri,row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for ci,c in enumerate(row.cells):
                c.width=Cm(widths[ci]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cp=c._tc.get_or_add_tcPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E8ECEF' if ri==0 else 'FFFFFF');cp.append(sh)
                mar=OxmlElement('w:tcMar')
                for side in ['top','bottom','left','right']:
                    e=OxmlElement('w:'+side);e.set(qn('w:w'),'70');e.set(qn('w:type'),'dxa');mar.append(e)
                cp.append(mar)
                for p in c.paragraphs:
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci in align_left else WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.space_before=Pt(1);p.paragraph_format.space_after=Pt(1);p.paragraph_format.line_spacing=1.1
                    for run in p.runs:
                        run.font.name='Times New Roman';run.font.size=Pt(size);run.bold=ri==0;run.font.color.rgb=RGBColor(0,0,0)
        return t

    def equation(self, text):
        # Native OMML fractions, scripts and summations; editable in Word.
        p=self.p(center=True)
        def mr(s):
            rr=OxmlElement('m:r');tt=OxmlElement('m:t');tt.text=s;rr.append(tt);return rr
        def wrap(tag,*children):
            x=OxmlElement('m:'+tag)
            for c in children:x.append(mr(c) if isinstance(c,str) else c)
            return x
        def frac(a,b):
            def argument(tag,value):
                x=wrap(tag)
                if not isinstance(value,str) and value.tag==qn('m:e'):
                    for child in list(value):x.append(child)
                else:x.append(mr(value) if isinstance(value,str) else value)
                return x
            return wrap('f',argument('num',a),argument('den',b))
        def sub(a,b):return wrap('sSub',wrap('e',a),wrap('sub',b))
        def sup(a,b):return wrap('sSup',wrap('e',a),wrap('sup',b))
        def summation(index,upper,*expr):
            props=wrap('naryPr');char=OxmlElement('m:chr');char.set(qn('m:val'),'∑');props.append(char)
            loc=OxmlElement('m:limLoc');loc.set(qn('m:val'),'subSup');props.append(loc)
            return wrap('nary',props,wrap('sub',index),wrap('sup',upper),wrap('e',*expr))
        if text.startswith('p(c'):
            seq=['p(c | u) = ',frac('1','B'),summation('b = 1','B',sub('p','b'),'(c | u)'), '    ;    ŷ = arg max p(c | u)']
        elif text.startswith('F1꜀'):
            seq=[sub('F1','c'),' = ',frac(wrap('e','2',sub('TP','c')),wrap('e','2',sub('TP','c'),' + ',sub('FP','c'),' + ',sub('FN','c')))]
        elif text.startswith('Macro-F1'):
            seq=['Macro-F1 = ',frac('1','10'),summation('c = 0','9',sub('F1','c'))]
        elif text.startswith('Log loss'):
            seq=['Log loss = −',frac('1','n'),summation('i = 1','n','log p(',sub('y','i'),' | ',sub('u','i'),')')]
        elif text.startswith('f(x)'):
            seq=['f(x) = ',frac('1','3'),summation('k = 1','3',sub('F1','macro'),'(',sup('y','k'),', ',sup('ŷ','k'),'(x))'), '    ;    maximize f(x), x ∈ X']
        else:seq=[text]
        om=OxmlElement('m:oMathPara');m=OxmlElement('m:oMath')
        for item in seq:m.append(mr(item) if isinstance(item,str) else item)
        om.append(m);p._p.append(om)
        return p

    def toc_entries(self,title,entries):
        self.h(title,1,page=True) if title!='MỤC LỤC' else self.p(title,style='Heading 1').paragraph_format.__setattr__('page_break_before',True)
        anchor=self.doc.paragraphs[-1]._p
        self.contents.append((anchor,entries))

    def insert_navigation(self):
        # Use source TOC styles, explicit hyperlinks and live PAGEREF fields.
        # Their cached page values are populated by refresh_report_fields.ps1.
        for anchor,kind in self.contents:
            entries={'toc':self.nav,'figures':self.figures,'tables':self.tables}[kind]
            prev=anchor
            start = None
            for text,level,name in entries:
                p=OxmlElement('w:p');pp=OxmlElement('w:pPr');s=OxmlElement('w:pStyle');s.set(qn('w:val'),'50' if level==1 else '51');pp.append(s)
                tabs=OxmlElement('w:tabs');tab=OxmlElement('w:tab')
                for k,v in {'val':'right','leader':'dot','pos':'9061'}.items():tab.set(qn('w:'+k),v)
                tabs.append(tab);pp.append(tabs)
                sp=OxmlElement('w:spacing');sp.set(qn('w:before'),'0');sp.set(qn('w:after'),'65');sp.set(qn('w:line'),'276');pp.append(sp);p.append(pp)
                link=OxmlElement('w:hyperlink');link.set(qn('w:anchor'),name)
                r=OxmlElement('w:r');t=OxmlElement('w:t');t.text=text;r.append(t);link.append(r);p.append(link)
                r=OxmlElement('w:r');r.append(OxmlElement('w:tab'));p.append(r)
                field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),f' PAGEREF {name} \\h ')
                r=OxmlElement('w:r');t=OxmlElement('w:t');t.text='1';r.append(t);field.append(r);p.append(field)
                prev.addnext(p);prev=p
                if start is None:start=p
            if start is not None:
                instructions={'toc':' TOC \\o "1-2" \\h \\u ', 'figures':' TOC \\t "Ảnh,1" \\h ', 'tables':' TOC \\t "Bảng,1" \\h '}
                runs=[]
                for typ in ['begin','instruction','separate']:
                    rr=OxmlElement('w:r')
                    if typ=='instruction':
                        it=OxmlElement('w:instrText');it.set(qn('xml:space'),'preserve');it.text=instructions[kind];rr.append(it)
                    else:
                        it=OxmlElement('w:fldChar');it.set(qn('w:fldCharType'),typ);rr.append(it)
                    runs.append(rr)
                for offset,rr in enumerate(runs):start.insert(1+offset,rr)
                rr=OxmlElement('w:r');end=OxmlElement('w:fldChar');end.set(qn('w:fldCharType'),'end');rr.append(end);prev.append(rr)

    def save(self,path):
        self.insert_navigation()
        settings=self.doc.settings.element
        uf=settings.find(qn('w:updateFields'))
        if uf is None:uf=OxmlElement('w:updateFields');settings.append(uf)
        uf.set(qn('w:val'),'true')
        self.doc.save(path)
        with ZipFile(BASE) as z: base={n:z.read(n) for n in z.namelist()}
        with ZipFile(path) as z: parts={n:z.read(n) for n in z.namelist()}
        for n in PRESERVE:
            if n in base:parts[n]=base[n]
        with ZipFile(path,'w',ZIP_DEFLATED) as z:
            for n,b in parts.items():z.writestr(n,b)
        return sorted(self.source_paths)


def build(output):
    make_figures();r=Report()
    rec={x['label']:x for x in rows('results/report_evaluation_2026-11-03/comparison.csv')}
    selection=load('results/selection.json');run=load('results/run_manifest.json')
    counts=rows('results/report_evaluation_2026-11-03/class_counts.csv')
    classes=load('results/report_evaluation_2026-11-03/classification_reports.json')
    names={'default':'RF mặc định','random':'RF Random','tpe':'RF TPE','selected':'Selected'}
    keys=list(names)
    def val(k,metric,n=5,percent=False):
        return (pct if percent else fnum)(rec[k][metric],n)

    # FRONT MATTER — source's original component roles and section geometry.
    r.h('NHẬN XÉT CỦA GIẢNG VIÊN')
    for _ in range(10):r.p('........................................................................................................')
    r.p('Điểm đánh giá',center=True)
    r.doc._element.body.insert(len(r.doc._element.body)-1,deepcopy(r.score))
    r.page('THÔNG TIN SINH VIÊN',1)
    from docx.table import Table
    student=deepcopy(r.student);r.doc._element.body.insert(len(r.doc._element.body)-1,student)
    tab=Table(student,r.doc._body)
    values=['1',AUTHOR,STUDENT_ID,'Đồ án cá nhân','100%']
    for cell,value in zip(tab.rows[1].cells,values):
        cell.text=value
        for p in cell.paragraphs:
            for run_ in p.runs:run_.font.name='Times New Roman';run_.font.size=Pt(13)
    r.p('Học phần: Nhập môn Tối ưu hóa.')
    r.p('Đề tài: Tối ưu siêu tham số Random Forest cho nhận dạng chữ số viết tay.')
    r.p('Ngày hoàn thiện báo cáo: '+DATE+'.')
    r.p('Phạm vi thực hiện cá nhân gồm mô hình hóa bài toán, mã nguồn, tổ chức thí nghiệm, kiểm chứng kết quả và trình diễn. Công cụ Codex được sử dụng để hỗ trợ lập trình, kiểm tra và biên soạn; sinh viên chịu trách nhiệm giải thích và xác nhận nội dung khi bảo vệ.')
    r.page('LỜI MỞ ĐẦU',1)
    r.p('Đề tài nghiên cứu cách tìm cấu hình Random Forest nhận dạng chữ số viết tay trong một ngân sách đánh giá hữu hạn. Trọng tâm của môn học được thể hiện qua biến quyết định, hàm mục tiêu, miền ràng buộc, thuật toán tìm kiếm và quy tắc chọn nghiệm có xét kích thước mô hình.')
    r.p('Báo cáo sử dụng bộ Digits gồm 1.797 ảnh 8 × 8 từ scikit-learn, có nguồn gốc UCI. Dữ liệu được tách thành 1.437 mẫu development và 360 mẫu test cố định. Random Search và Optuna TPE được so sánh trên cùng ba folds validation, cùng không gian tham số và cùng ngân sách 30 trials cho mỗi phương pháp và search seed.')
    r.p('Trong thí nghiệm gốc, cấu hình có CV Macro-F1 cao nhất thuộc Random Search, đạt 0,97485. Quy tắc ưu tiên mô hình nhỏ trong ngưỡng giảm F1 bằng 0,005 lại chọn RF mặc định. Cấu hình được chọn đạt test accuracy 97,22%, đúng 350/360 mẫu, và nhỏ hơn 40,28% so với cấu hình Random tốt nhất; không có cải thiện so với chính baseline mặc định.')
    r.p('Báo cáo trình bày riêng kết quả train, validation, test, lịch sử tìm kiếm, nghiên cứu mở rộng chỉ trên development và bằng chứng tái lập. Train accuracy 100% được ghi đúng là kết quả trên dữ liệu đã fit; không dùng con số này để kết luận khả năng tổng quát hóa.')
    r.p('Em xin cảm ơn giảng viên học phần đã hướng dẫn cách xây dựng và đánh giá bài toán tối ưu. Em chịu trách nhiệm về cách diễn giải số liệu và các giới hạn được nêu trong báo cáo.')
    r.p('Sinh viên thực hiện: '+AUTHOR,italic=True)
    r.toc_entries('MỤC LỤC','toc')
    r.toc_entries('DANH MỤC HÌNH ẢNH','figures')
    r.toc_entries('DANH MỤC BẢNG','tables')
    r.page('DANH MỤC TỪ VIẾT TẮT',1)
    r.table('Bảng M.1. Các thuật ngữ dùng trong báo cáo',['Từ viết tắt','Diễn giải'],[
        ['RF','Random Forest, rừng ngẫu nhiên'],['HPO','Hyperparameter optimization, tối ưu siêu tham số'],
        ['CV','Cross-validation, đánh giá chéo'],['OOF','Out-of-fold, dự đoán ngoài fold đã dùng để fit'],
        ['TPE','Tree-structured Parzen Estimator'],['F1 macro','Trung bình F1 của 10 lớp với trọng số bằng nhau'],
        ['SD','Độ lệch chuẩn mẫu, không phải khoảng tin cậy'],['MB','1.000.000 bytes của tệp model đã tuần tự hóa'],
        ['Trial','Một lần đề xuất và đánh giá cấu hình'],['Seed','Hạt giống của bộ sinh số ngẫu nhiên'],
        ['Development','Tập phát triển dùng để tìm kiếm và refit'],['Test holdout','Tập giữ riêng để đánh giá sau khi khóa lựa chọn']],[3,13],align_left=(0,1))
    p=r.p();p._p.get_or_add_pPr().append(deepcopy(r.sections[1]))

    # CHAPTER I — three pages.
    r.h('CHƯƠNG I TỔNG QUAN VÀ DỮ LIỆU')
    r.h('1.1 Bài toán và mục tiêu',2)
    r.p('Đầu vào là ma trận cường độ ảnh xám 8 × 8, trải phẳng thành vector 64 giá trị trong khoảng 0–16. Đầu ra là một nhãn chữ số từ 0 đến 9 và vector điểm số xác suất của 10 lớp. Bài toán học máy là phân loại đa lớp; bài toán tối ưu là chọn bốn siêu tham số của Random Forest sao cho chất lượng validation cao trong ngân sách tính toán cho trước.')
    r.p('Đồ án so sánh RF mặc định với Random Search và TPE. Sau tìm kiếm, quy tắc epsilon chọn mô hình có dung lượng nhỏ nhất trong nhóm cấu hình đạt gần điểm CV tốt nhất. Cách đặt bài toán này cho phép thảo luận sự đánh đổi giữa chất lượng và tài nguyên, thay vì chỉ trình bày một giá trị accuracy cuối cùng.')
    r.h('1.2 Nguồn và phạm vi dữ liệu',2)
    r.p('Dữ liệu được nạp bằng sklearn.datasets.load_digits. Bản này gồm 1.797 mẫu, 64 đặc trưng, 10 lớp và không có giá trị thiếu [1], [2]. Nó tương ứng tệp optdigits.tes của UCI; bộ UCI đầy đủ có 5.620 mẫu. Đồ án không sử dụng toàn bộ 5.620 mẫu và không sử dụng cách chia train/test gốc của UCI.')
    r.p('Tên optdigits.tes là tên tệp nguồn, không có nghĩa tất cả 1.797 dòng là tập test của đồ án. Trên chính tập con này, mã nguồn tạo lại development/test bằng phép chia phân tầng seed 42. Đối chiếu độc lập đã xác nhận toàn bộ pixel và nhãn khớp nguồn UCI; không sửa dữ liệu để đạt metric mong muốn.')
    r.h('1.3 Tiêu chí đánh giá',2)
    r.p('Chỉ số tối ưu chính là Macro-F1 trên validation, vì mỗi lớp chữ số đóng góp như nhau. Accuracy, precision macro, recall macro và log loss được báo cáo bổ sung. Kích thước tệp model là tiêu chí thứ hai ở bước chọn nghiệm; thời gian chạy chỉ được mô tả theo phạm vi đo, không thay thế hàm mục tiêu.')

    r.page('1.4 Phân chia dữ liệu và phân bố nhãn')
    r.table('Bảng 1.1. Số mẫu từng lớp trong cách chia của đồ án',['Nhãn','Development','Test','Tổng'],[
        [c['class'],c['development'],c['test'],c['total']] for c in counts]+[['Tổng',1437,360,1797]],[2,5,4,5],align_left=(),size=12)
    r.p('Development chiếm xấp xỉ 80%, test chiếm xấp xỉ 20%. Tỷ lệ là gần đúng vì số mẫu nguyên và phép chia có phân tầng. Các chỉ số mẫu được lưu trong results/split_indices.json để mọi cấu hình dùng đúng cùng dữ liệu.')
    r.table('Bảng 1.2. Vai trò của từng phân vùng',['Phân vùng','Số mẫu','Vai trò'],[
        ['Train trong mỗi fold',958,'Fit mô hình khi đánh giá một trial'],['Validation trong mỗi fold',479,'Tính điểm CV; không fit trên các dòng này trong fold tương ứng'],
        ['Development đầy đủ',1437,'Refit cấu hình đã xác định và đo train in-sample'],['Test holdout',360,'Đánh giá cấu hình sau bước lựa chọn']],[5,2,9],align_left=(0,2))
    r.p('Ba folds phân tầng được cố định bằng seed 42. Mỗi mẫu development xuất hiện đúng một lần ở phần validation OOF. Cùng một mẫu có thể thuộc train ở fold khác; điều này là cơ chế CV, không phải chồng lấn train/validation trong cùng một lần fit.')

    r.page('1.5 Biểu diễn đầu vào và tiền xử lý')
    r.figure('figures/report_2026-11-27/digits_examples.png','Ảnh 1.1. Mười ảnh development đại diện cho các chữ số','web/dist/data/dashboard.json, mẫu development đã xuất; không lấy mẫu test.',15.8)
    r.p('Ảnh 1.1 hiển thị trực tiếp lưới 8 × 8, không nội suy để thay đổi nét. Mỗi ô là một đặc trưng; nhãn nằm riêng trong y và không được ghép vào X. Dữ liệu được giữ ở thang 0–16; không chuẩn hóa, bù khuyết hay tăng cường ảnh vì bộ dữ liệu không thiếu và RF không yêu cầu chuẩn hóa thang đo ở đây.')
    r.table('Bảng 1.3. Đặc tả đầu vào và đầu ra',['Thành phần','Đặc tả'],[
        ['X','Ma trận 1.797 × 64; mỗi dòng là một ảnh 8 × 8'],['y','Vector 1.797 nhãn nguyên 0–9, lưu tách biệt X'],
        ['Một mẫu demo','64 số hữu hạn trong [0,16] hoặc ma trận 8 × 8'],['Dự đoán','Một nhãn và 10 điểm số lớp từ Random Forest']],[4,12],align_left=(0,1))
    r.p('Kiểm tra độc lập không tìm thấy chỉ số mẫu giao nhau hoặc hai dòng đặc trưng hoàn toàn giống nhau giữa development và test. Kết quả này hỗ trợ kiểm soát rò rỉ cụ thể của cách chia; nó không chứng minh bộ dữ liệu nhỏ có thể đại diện mọi kiểu chữ viết thực tế.')

    # CHAPTER II — four pages.
    r.page('CHƯƠNG II MÔ HÌNH VÀ PHƯƠNG PHÁP TỐI ƯU',1)
    r.h('2.1 Random Forest và các thước đo',2)
    r.p('Random Forest kết hợp nhiều cây quyết định học trên các mẫu bootstrap và các tập đặc trưng được lấy ngẫu nhiên [3]. Với triển khai scikit-learn, điểm số một lớp là trung bình xác suất do từng cây trả về; xác suất tại lá dựa trên tỷ lệ nhãn của các mẫu tại lá [4]. Dự đoán cuối lấy lớp có điểm số lớn nhất.')
    r.equation('p(c | u) = (1 / B) ∑ᵦ₌₁ᴮ pᵦ(c | u)     ;     ŷ = arg max꜀ p(c | u)')
    r.p('B là số cây, u là vector ảnh và c là lớp. Các cây được xây bằng tiêu chí Gini của cấu hình gốc. Log loss trên đồ thị là phép đo từ xác suất dự đoán, không phải hàm loss được tối ưu bằng gradient trong quá trình huấn luyện RF.')
    r.equation('F1꜀ = 2TP꜀ / (2TP꜀ + FP꜀ + FN꜀)')
    r.equation('Macro-F1 = (1 / 10) ∑꜀₌₀⁹ F1꜀')
    r.p('Mỗi lớp được lần lượt xem là lớp dương để tính TP, FP, FN. F1 macro là trung bình không trọng số qua mười lớp [9]. Accuracy là tỷ lệ dự đoán đúng; log loss là trung bình âm log xác suất của lớp thật [10]. Giá trị log loss nhỏ hơn là tốt hơn, nhưng không dùng nó thay mục tiêu CV Macro-F1.')
    r.equation('Log loss = −(1 / n) ∑ᵢ₌₁ⁿ log p(yᵢ | uᵢ)')
    r.p('Khi một cây chưa cho xác suất dương cho lớp thật, hàm tính log loss giới hạn xác suất theo epsilon số học. Vì vậy log loss đầu đường cong có thể lớn; không thể xem điểm này là lỗi dữ liệu chỉ từ hình dạng đường.')

    r.page('2.2 Biến quyết định và hàm mục tiêu')
    r.p('Biến quyết định là x = (B, d, l, m): số cây, độ sâu tối đa, số mẫu tối thiểu ở lá và số đặc trưng xét tại một nút. Với ba folds cố định, mục tiêu của một trial là tối đa hóa trung bình F1 macro trên validation.')
    r.equation('f(x) = (1 / 3) ∑ₖ₌₁³ F1macro(yᵏ, ŷᵏ(x))     ;     maximize f(x), x ∈ X')
    r.table('Bảng 2.1. Miền tìm kiếm siêu tham số',['Biến','Miền giá trị','Số lựa chọn'],[
        ['B · n_estimators','50, 75, 100, 125, 150, 175, 200',7],['d · max_depth','5, 10, 15, 20, None',5],
        ['l · min_samples_leaf','1, 2, 3, 4',4],['m · max_features','sqrt, log2, half, all',4]],[5,9,2],align_left=(0,1))
    r.p('Miền X có 7 × 5 × 4 × 4 = 560 cấu hình. “half” và “all” được ánh xạ thành 0,5 và 1,0 trong RF; None nghĩa không đặt trần độ sâu. Ràng buộc là x thuộc miền này, trial phải hoàn thành hợp lệ và mỗi phương pháp chỉ được dùng 30 trials cho một search seed. Đây là tìm kiếm hữu hạn có ngân sách, không phải chứng minh tối ưu toàn cục.')
    r.h('2.3 Quy tắc chọn nghiệm theo epsilon',2)
    r.p('Gọi C là tập các cấu hình khả thi đã quan sát, bao gồm baseline. Đặt f* = max f(x) trên C, rồi chọn cấu hình có số bytes nhỏ nhất trong nhóm f(x) ≥ f* − ε, với ε = 0,005. Kích thước được đo sau refit trên development; khi cần phá hòa, dùng điểm CV rồi định danh theo quy tắc cố định trong mã nguồn.')
    r.equation('x̂ = arg min { size(x) : x ∈ C, f(x) ≥ f* − 0,005 }')
    r.p('Ngưỡng cho phép giảm tối đa 0,5 điểm phần trăm F1 so với nghiệm tốt nhất đã thấy. Test không tham gia tính f*, ε hoặc lựa chọn mô hình. Do C chỉ gồm các điểm đã đánh giá, kết luận “nhỏ nhất” chỉ áp dụng trong tập quan sát này.')

    r.page('2.4 Random Search và Optuna TPE')
    r.p('Random Search lấy mẫu các cấu hình trong miền đã xác định mà không dùng điểm số trước đó để định hướng [5]. Nó là baseline tìm kiếm phù hợp vì cùng chịu giới hạn 30 lần đánh giá với TPE, thay vì được cho một ngân sách khác.')
    r.p('TPE sử dụng lịch sử trial để mô hình hóa vùng tham số của các quan sát tốt và phần còn lại, rồi ưu tiên đề xuất mới dựa trên tỷ lệ các mật độ ước lượng [6]. Đồ án dùng TPESampler của Optuna, startup 10 ở nghiên cứu gốc, seed theo từng lần chạy và không pruning [7], [8]. Mười trial khởi động dùng lấy mẫu ngẫu nhiên trước khi TPE tận dụng lịch sử.')
    r.table('Bảng 2.2. Các bước thực hiện một lượt tìm kiếm',['Bước','Thao tác'],[
        [1,'Nạp config, chỉ số development/test và ba folds đã cố định'],[2,'Đánh giá RF mặc định làm mốc chung'],
        [3,'Sampler đề xuất x; fit 3 mô hình trên train của từng fold'],[4,'Tính và lưu ba F1, trung bình, thời gian, trạng thái'],
        [5,'Refit development để đo bytes và profiling trong nghiên cứu gốc'],[6,'Lặp đủ 30 trials; giữ tất cả trial và lịch sử best-so-far'],
        [7,'Áp dụng quy tắc epsilon trên mọi cấu hình khả thi và baseline'],[8,'Khóa lựa chọn; đánh giá test các cấu hình đại diện, lưu predictions']],[1.5,14.5],align_left=(1,))
    r.p('Search seeds 11, 22, 33 tạo ba lịch sử cho mỗi phương pháp; model seed 123 luôn cố định. Lịch sử có thể lặp một cấu hình nên 30 trials không nhất thiết là 30 cấu hình khác nhau. ID trial trong CSV bắt đầu từ 0; trial 21 là lần đánh giá thứ 22.')
    r.p('Mã cài đặt không sửa kết quả sampler theo thứ hạng mong muốn. Lịch sử best-so-far chỉ giữ điểm cao nhất đã có trong prefix, nên đường này không giảm là đúng định nghĩa; các điểm trial riêng lẻ vẫn có thể tăng giảm.')

    r.page('2.5 Quy trình thực nghiệm và môi trường')
    r.figure('figures/report_2026-11-27/experiment_pipeline.png','Ảnh 2.1. Quy trình tìm kiếm và khóa lựa chọn trước test','Tổng hợp từ configs/experiment.json và src/experiment.py.',15.5)
    r.table('Bảng 2.3. Môi trường và thiết lập của nghiên cứu gốc',['Hạng mục','Thiết lập đã lưu'],[
        ['Hệ thống','Windows 11; Python 3.12.5; 12 logical CPUs'],['Thư viện','numpy 2.2.2; pandas 2.2.3; scikit-learn 1.8.0'],
        ['Thư viện tiếp theo','Optuna 5.0.0; joblib 1.5.3; matplotlib 3.10.0'],['Seed và song song','split/CV 42; RF 123; search 11/22/33; n_jobs=1'],
        ['Ngân sách gốc','2 phương pháp × 3 seeds × 30 = 180 search trials']],[4.5,11.5],align_left=(0,1),size=11)
    r.p('Tổng cộng có 540 lần fit cho CV tìm kiếm và 3 lần fit CV baseline; các bước refit/profiling là phần công việc bổ sung. Manifest lưu phiên bản, thời điểm, config hash và tổng thời gian. Phần mở rộng 450 trials được mô tả riêng ở Chương III vì khác phạm vi đo.')

    # CHAPTER III — results, explicitly distinguish statistical populations.
    r.page('CHƯƠNG III KẾT QUẢ VÀ ĐÁNH GIÁ',1)
    r.h('3.1 Cấu hình đại diện và kết quả train',2)
    r.table('Bảng 3.1. Các cấu hình đại diện của thí nghiệm gốc',['Cấu hình','Cây','Sâu','Lá','Đặc trưng','Nguồn'],[
        ['RF mặc định',100,'None',1,'sqrt','Baseline'],['RF Random',175,10,1,'log2','Seed22, trial21'],
        ['RF TPE',175,20,1,'sqrt','Seed22, trial26'],['Selected',100,'None',1,'sqrt','Trùng baseline']],[3.1,1.2,1.4,1.1,2.7,6.5],align_left=(0,5),size=11)
    r.table('Bảng 3.2. Kết quả trên development sau refit',['Cấu hình','Đúng / tổng','Accuracy','F1 macro','Log loss'],[
        [names[k],'1.437 / 1.437',val(k,'train_accuracy',2,True),val(k,'train_f1_macro'),val(k,'train_log_loss')] for k in keys],[3.7,3.8,2.5,2.5,3.5])
    r.p('Các giá trị train là in-sample: mô hình được fit và chấm trên cùng 1.437 mẫu development. Cây sâu và lá nhỏ có thể khớp toàn bộ dữ liệu học, nên accuracy 100% là khả thi; con số này không có nghĩa dự đoán đúng mọi chữ viết. Baseline và Selected là cùng một cấu hình, không phải hai nghiệm độc lập.')
    r.p('Kiểm tra độc lập đã refit ba cấu hình phân biệt, đếm đúng 1.437/1.437 và xác nhận mảng dự đoán tách biệt nhãn thật. Pixel/nhãn không đổi sau fit; phân chia và folds được dựng lại để đối chiếu. Bằng chứng chi tiết lưu trong results/train_accuracy_audit_2026-11-03/audit.json.')
    r.p('Không dùng accuracy train để xếp hạng phương pháp. Chênh lệch với validation và test ở các bảng tiếp theo mới cho thấy mức giảm khi dự đoán dữ liệu ngoài phần đã học.')

    r.page('3.2 Bảng validation và so sánh baseline')
    r.table('Bảng 3.3. Chất lượng validation qua ba folds cố định',['Cấu hình','Accuracy (%)','Macro-F1','Log loss'],[
        [names[k],f"{fnum(float(rec[k]['validation_accuracy_mean'])*100,2)} ± {fnum(float(rec[k]['validation_accuracy_sd'])*100,2)}",
         f"{val(k,'validation_f1_macro_mean')} ± {val(k,'validation_f1_macro_sd')}",
         f"{val(k,'validation_log_loss_mean')} ± {val(k,'validation_log_loss_sd')}"] for k in keys],[3.3,4,4.7,4],size=11)
    r.note('Giá trị là trung bình ± SD mẫu qua 3 folds; SD không phải khoảng tin cậy.')
    folds=rows('results/report_evaluation_2026-11-03/validation_per_fold.csv')
    r.table('Bảng 3.4. Macro-F1 validation của từng fold',['Cấu hình','Fold 1','Fold 2','Fold 3','OOF đúng'],[
        [names[k]]+[fnum(next(x['f1_macro'] for x in folds if x['label']==k and x['fold']==str(j))) for j in [1,2,3]]+
        [{'default':'1.395/1.437','random':'1.401/1.437','tpe':'1.400/1.437','selected':'1.395/1.437'}[k]] for k in keys],[3.5,2.7,2.7,2.7,4.4])
    r.p('So với RF mặc định, đại diện Random tăng CV Macro-F1 0,00423 và đại diện TPE tăng 0,00353. Đây là chênh lệch của cấu hình tốt nhất quan sát được, chưa chứng minh cải thiện có ý nghĩa thống kê. Việc chọn tốt nhất sau nhiều trials có thể làm điểm CV lạc quan hơn khả năng tổng quát hóa.')
    r.p('Dự đoán OOF dùng mô hình không học mẫu đó trong fold tương ứng. Accuracy OOF bằng trung bình accuracy fold vì ba fold cùng 479 mẫu. Macro-F1 trung bình fold không được tự động thay bằng Macro-F1 tính trên toàn bộ OOF vì F1 là hàm phi tuyến.')
    r.p('Mục tiêu của lựa chọn cuối còn xét dung lượng. Vì vậy Selected có điểm CV thấp hơn đại diện Random nhưng vẫn thỏa ngưỡng đã đặt; báo cáo không đổi luật để ép một mô hình tuned trở thành nghiệm cuối.')

    r.page('3.3 Kết quả test của các cấu hình đã chốt')
    r.table('Bảng 3.5. Kết quả trên 360 mẫu test giữ riêng',['Cấu hình','Đúng','Sai','Accuracy','F1 macro','MB'],[
        [names[k],rec[k]['test_correct'],rec[k]['test_errors'],val(k,'test_accuracy',2,True),val(k,'test_f1_macro'),val(k,'model_mb',3)] for k in keys],[3.2,1.3,1.2,3.1,3.4,3.8])
    r.table('Bảng 3.6. Precision và recall macro trên test',['Cấu hình','Precision macro','Recall macro'],[
        [names[k],val(k,'test_precision_macro'),val(k,'test_recall_macro')] for k in keys],[5,5.5,5.5])
    r.p('Selected đúng 350/360 mẫu, accuracy 97,22%; Random đúng 351/360, accuracy 97,50%. Chênh lệch là một mẫu test, tương đương 0,28 điểm phần trăm accuracy. TPE và Selected có các metric tổng hợp bằng nhau trong lần chạy này; điều đó không phải bằng chứng chúng luôn dự đoán giống nhau trên mọi đầu vào.')
    r.p('Test được dùng để mô tả khả năng tổng quát hóa sau khi khóa lựa chọn bằng development CV. Không đổi cấu hình, epsilon hoặc kết luận “phương pháp thắng” dựa trên chênh lệch test này. Các hình lỗi được dùng để phân tích hậu nghiệm, không được đưa trở lại quy trình chọn mô hình đã công bố.')
    r.p('MB ở đây là số bytes của tệp joblib chia 1.000.000; đây không phải bộ nhớ RAM hoặc kích thước model JSON của web. Việc lưu định dạng khác có thể đổi số bytes, nên mọi so sánh dung lượng trong bảng dùng cùng quy trình tuần tự hóa.')

    r.page('3.4 Hai đường train và validation theo số cây')
    r.figure('figures/train_validation_two_lines/accuracy.png','Ảnh 3.1. Accuracy train và validation theo số cây','600 phép đo trong results/curve_audit/all_tree_metrics_per_fold.csv; trung bình 3 folds.',15.8)
    r.p('Trục ngang là số cây từ 1 đến 100 của cấu hình mặc định, không phải epoch của mạng nơ-ron. Với mỗi số cây, đường train được chấm trên 958 mẫu đã fit và đường validation trên 479 mẫu giữ ra ở từng fold, rồi lấy trung bình ba folds. Ảnh này chỉ mô tả cấu hình Selected, không phải lịch sử tối ưu Random/TPE.')
    r.p('Đường train tiến tới 100% trong khi validation thấp hơn. Khoảng cách phản ánh việc chấm trên dữ liệu đã học so với dữ liệu ngoài phần fit. Không nên suy diễn rằng tăng cây luôn cải thiện accuracy: đường validation trung bình có 31 bước giảm khi quan sát đủ 100 mốc.')
    r.p('Đường cong được nối từ các số đo thật, không spline, không lọc nhiễu hoặc chỉnh nhãn. Bản đồ thị hai đường chỉ đơn giản hóa cách hiển thị, còn số đo từng fold vẫn được lưu để kiểm tra.')

    r.page('3.5 Log loss và kiểm tra tính trung thực của đường cong')
    r.figure('figures/train_validation_two_lines/log_loss.png','Ảnh 3.2. Log loss train và validation theo số cây','Cùng 600 phép đo nguồn của Ảnh 3.1; không làm mượt.',15.8)
    r.p('Loss ở đây là log loss tính từ xác suất RF. Tại một cây, validation có 122, 118 và 142 mẫu ở ba folds nhận xác suất lớp thật bằng 0 trước clipping, nên loss đầu đường lớn. Thang đầy đủ làm các dao động cuối đường khó thấy; không được từ đó kết luận đường đã bị chỉnh sửa.')
    r.p('Kiểm tra dày từng cây ghi nhận 34 bước tăng của validation log loss. 63 lần refit độc lập tái lập 126 dòng mốc cũ với sai khác metric lớn nhất khoảng 1,78 × 10⁻¹⁵. Thử đối chứng xáo nhãn train trên bản sao đạt validation accuracy 11,48%, gần mức ngẫu nhiên 10%; đối chứng không thay dữ liệu thí nghiệm chính.')
    r.p('Train loss ở cuối hình được tính trong folds nên khác train log loss 0,08883 khi refit toàn bộ 1.437 mẫu ở Bảng 3.2. Hai con số thuộc hai mô hình huấn luyện trên số lượng mẫu khác nhau; không dùng sự khác nhau này làm bằng chứng chỉnh sửa số liệu.')

    r.page('3.6 Lịch sử tìm kiếm và độ biến thiên theo seed')
    r.figure('figures/search_history.png','Ảnh 3.3. Điểm CV tốt nhất đã quan sát theo ngân sách trial','results/trials.csv; nghiên cứu gốc 3 search seeds, 30 trials mỗi lượt.',14.8)
    best=rows('results/best_per_seed.csv')
    r.table('Bảng 3.7. Nghiệm tốt nhất của từng lượt tìm kiếm gốc',['Phương pháp','Seed','Best CV F1','CV search (s)'],[
        ['Random' if x['method']=='random' else 'TPE',x['seed'],fnum(x['best_cv_f1']),fnum(x['search_cv_seconds'],2)] for x in best],[4,2,5,5],size=11.5)
    r.p('Best CV F1 trung bình qua ba search seeds là 0,97340 ± 0,00146 cho Random và 0,97221 ± 0,00270 cho TPE. Đây là SD giữa các lượt tìm kiếm, khác SD giữa ba folds ở Bảng 3.3. Kết quả gốc nghiêng về Random nhưng không chứng minh Random luôn tốt hơn TPE trên các bài toán khác.')
    r.p('Đường best-so-far không giảm do được tính bằng cực đại tích lũy. Nó phản ánh tiến trình tìm nghiệm theo ngân sách; đồ thị accuracy/loss theo cây ở hai mục trước phản ánh một câu hỏi khác.')

    r.page('3.7 Đánh đổi chất lượng và kích thước mô hình')
    r.figure('figures/quality_size_tradeoff.png','Ảnh 3.4. Quan hệ giữa CV Macro-F1 và dung lượng model','results/pareto_candidates.csv và results/selection.json.',15.8)
    r.p('Điểm CV tốt nhất đã thấy là 0,9748483764, nên ngưỡng chấp nhận bằng 0,9698483764. RF mặc định đạt 0,9706155160 và dung lượng 4.975.993 bytes, nhỏ nhất trong tập cấu hình khả thi đáp ứng ngưỡng. Do đó Selected trùng baseline.')
    r.table('Bảng 3.8. Độ nhạy của lựa chọn khi thay epsilon',['Epsilon','CV F1 chọn','Khoảng giảm F1','MB','Nguồn'],[
        [fnum(x['epsilon'],4),fnum(x['selected_cv_f1']),fnum(x['cv_gap']),fnum(x['model_mb'],3),x['method']] for x in rows('results/epsilon_sensitivity.csv')],[2.4,3.2,4.2,2.5,3.7],size=11)
    r.p('Selected nhỏ hơn 40,28% so với RF Random tốt nhất, với giảm CV F1 0,00423; so với baseline thì giảm dung lượng 0%. Bảng độ nhạy là phân tích lại các kết quả CV đã lưu, không tìm epsilon bằng test và không tạo lựa chọn triển khai mới.')

    r.page('3.8 Nghiên cứu mở rộng theo ngân sách và startup')
    r.p('Phần mở rộng đánh giá ba phương pháp Random, TPE startup 10 và TPE startup 5 trên năm search seeds 11, 22, 33, 44, 55. Mỗi lượt có 30 trials, tổng 450 trials và 1.350 lần fit CV, cộng ba lần fit baseline. Toàn bộ nghiên cứu chỉ dùng development; không refit để profiling, không chấm test mới và không thay model Selected.')
    r.figure('figures/robustness/search_budget.png','Ảnh 3.5. So sánh mở rộng theo ngân sách 10 đến 30 trials','results/robustness/budget_summary.csv; 5 seeds, chỉ development.',15.8)
    bud=rows('results/robustness/budget_summary.csv')
    r.table('Bảng 3.9. Best CV F1 qua năm seeds tại 30 trials',['Phương pháp','Mean ± SD','Elapsed mean (s)','Median (s)'],[
        [dict(random='Random',tpe_startup_10='TPE startup10',tpe_startup_5='TPE startup5')[x['method']],f"{fnum(x['best_cv_f1_mean'])} ± {fnum(x['best_cv_f1_std'])}",
         fnum(x['elapsed_seconds_mean'],2),fnum(x['elapsed_seconds_median'],2)] for x in bud if x['budget']=='30'],[4.3,4.9,3.7,3.1],size=11)
    r.p('Trong nghiên cứu này TPE startup 5 đạt điểm trung bình cao nhất và SD nhỏ hơn. Đây là kết quả của một nghiên cứu development bổ sung; không gộp nó với ba seeds gốc để trình bày như cùng một thí nghiệm hoặc như một cải thiện test.')

    r.page('3.9 Phân tích startup và giới hạn so sánh thời gian')
    r.figure('figures/robustness/startup_ablation.png','Ảnh 3.6. So sánh TPE startup 5 và startup 10 theo từng seed','results/robustness/startup_ablation_pairs.csv.',13.8)
    r.p('So sánh ghép cặp cùng năm seeds cho chênh lệch F1 trung bình startup 5 trừ startup 10 là 0,00160, SD 0,00239; startup 5 thắng ba seeds và hòa hai seeds. Đây là thống kê mô tả, chưa phải kiểm định ý nghĩa hay khoảng tin cậy cho nhiều bộ dữ liệu.')
    r.p('Elapsed trung bình của TPE startup 10 bị kéo lên bởi seed 44 mất 490,17 giây; quan sát ngoại lệ được giữ nguyên. Vì vậy phải xem cả median và độ phân tán, không dùng riêng mean để kết luận một sampler chậm hơn về bản chất. Thời gian máy bận và chi phí cấu hình đều có thể ảnh hưởng.')
    r.table('Bảng 3.10. Phạm vi các phép đo thời gian',['Phép đo','Phạm vi và giới hạn'],[
        ['CV search gốc','Tổng thời gian fit/score 3 folds của các trials; không gồm mọi refit/profiling'],['Elapsed mở rộng','Thời gian tích lũy theo lượt nghiên cứu CV-only'],
        ['So sánh cùng thời gian','Prefix hoàn tất trong 3,119–21,398 giây, 60 mốc chung; không nội suy kết quả chưa có'],['Latency model cuối','Đo riêng bằng batch development, không phải thời gian tìm kiếm']],[4.3,11.7],align_left=(0,1))

    r.page('3.10 Ma trận nhầm lẫn và kết quả từng lớp')
    r.figure('figures/week10_2026-11-09/confusion_matrix.png','Ảnh 3.7. Ma trận nhầm lẫn của Selected trên test','results/test_predictions.csv; không suy luận test mới khi vẽ.',11.2)
    r.table('Bảng 3.11. Precision recall và F1 từng lớp của Selected',['Lớp','Support','Precision','Recall','F1'],[
        [i,int(classes['selected']['test'][str(i)]['support']),fnum(classes['selected']['test'][str(i)]['precision'],4),
         fnum(classes['selected']['test'][str(i)]['recall'],4),fnum(classes['selected']['test'][str(i)]['f1-score'],4)] for i in range(10)],[1.5,3,3.8,3.8,3.9],align_left=(),size=11)

    r.page('3.11 Quan sát mười mẫu test dự đoán sai')
    r.figure('figures/week10_2026-11-09/selected_test_errors.png','Ảnh 3.8. Mười ảnh test bị Selected phân loại sai','results/week10_2026-11-09/selected_errors.csv; predictions test gốc.',15.8)
    r.p('Selected sai mười mẫu, trong đó có ba trường hợp nhãn 8 bị dự đoán thành 1. Lớp 8 có recall thấp nhất trong lần chia này, 31/35 = 88,57%. RF Random sửa được hai lỗi của Selected nhưng đồng thời sai một mẫu mà Selected đoán đúng; hai mô hình có tám lỗi chung. Vì vậy Random chỉ giảm ròng một lỗi trên test.')
    r.p('Hình ảnh giúp kiểm tra nhãn thật, nhãn dự đoán và chỉ số nguồn, nhưng nhận xét “nét giống chữ khác” chỉ là cách diễn giải trực quan. Không có thí nghiệm can thiệp để chứng minh nét ảnh cụ thể là nguyên nhân gây lỗi. Không chỉnh pixel, bỏ mẫu khó hoặc chuyển các mẫu này vào development để làm đẹp metric đã công bố.')
    r.p('Phân tích lỗi cho thấy cần thận trọng khi chuyển mô hình sang chữ viết tự vẽ. Dữ liệu thực tế có thể khác độ dày nét, vị trí và phân bố cường độ. Nếu cải tiến tiền xử lý trong tương lai, cần xác định protocol mới và tập đánh giá chưa dùng; không tiếp tục chọn tham số trên holdout đã quan sát này.')

    r.page('3.12 Tái lập và độ biến thiên runtime')
    r.table('Bảng 3.12. Đối chiếu lần tái lập ngày 09/11/2026',['Nội dung','Kết quả đối chiếu'],[
        ['Trial records','181 bản ghi: baseline và 180 search trials'],['Tham số và điểm folds','Khớp toàn bộ; sai khác CV lớn nhất bằng 0'],
        ['Lựa chọn và test predictions','Khớp kết quả đã lưu'],['Model Selected','SHA256 nhị phân khớp'],
        ['Thời gian toàn bộ gốc','431,21 giây'],['Thời gian lần tái lập','855,77 giây; xấp xỉ 1,98 lần gốc']],[6.5,9.5],align_left=(0,1))
    r.figure('figures/week10_2026-11-09/latency_stability.png','Ảnh 3.9. Biến thiên latency của Selected trong năm vòng đo','results/week10_2026-11-09/latency_measurements.csv; batch 256 development samples.',13.3)
    r.p('Năm vòng đo gồm ba warmups và 15 phép đo mỗi vòng, tổng 75 số đo được giữ lại. Median gộp là 20,82 ms/batch và p95 là 84,03 ms/batch. Trung bình các median vòng là 25,49 ± 13,17 ms, hệ số biến thiên 51,67%; số liệu không hỗ trợ tuyên bố latency ổn định.')
    r.p('Lần tái lập dùng đúng các phiên bản thư viện gốc và cùng holdout đã biết, nên không phải một test độc lập mới. Phiên chạy do Codex thực hiện; sinh viên tự chạy và giải thích kết quả vẫn cần được xác nhận riêng. Thời gian hai phiên chỉ là quan sát mô tả, không phải benchmark kiểm soát tải máy.')

    # CHAPTER IV.
    r.page('CHƯƠNG IV TRÌNH DIỄN VÀ KẾT LUẬN',1)
    r.h('4.1 Giao diện Digits Lab',2)
    r.p('Đồ án có giao diện web tĩnh bằng HTML, CSS và JavaScript để xem kết quả và thử suy luận. Web đọc dữ liệu đã xuất từ logs và model gốc; thao tác giao diện không huấn luyện thêm hoặc thay đổi số liệu thực nghiệm. Năm màn hình tổ chức theo câu hỏi người xem cần kiểm tra.')
    r.table('Bảng 4.1. Chức năng các màn hình của web',['Màn hình','Nội dung trình diễn'],[
        ['Tổng quan','Selected, ngân sách 180 trials, đánh đổi CV và kích thước'],['So sánh mô hình','Train/validation/test, các baseline và dữ liệu CSV'],
        ['Quá trình tìm kiếm','Seed, prefix trial, phát lại Random/TPE và best-so-far'],['Train và validation','Hai đường accuracy hoặc log loss theo số cây'],
        ['Thử nhận dạng','30 mẫu development và lưới vẽ trực tiếp 8 × 8']],[4.8,11.2],align_left=(0,1))
    r.h('4.2 Cách chạy và thử đầu vào',2)
    r.p('Tại thư mục gốc, nhấp đúp web/start_web.bat hoặc chạy lệnh dưới đây. Giữ cửa sổ terminal rồi mở http://127.0.0.1:8765. Bản local dùng Python chuẩn để phục vụ tệp tĩnh, không cần cài thư viện ML chỉ để xem web.')
    r.p('py -3 -X utf8 web/serve.py',left=True,size=12)
    r.p('Đầu vào thử nhận dạng là 64 giá trị trong [0,16]. Người dùng chọn ảnh development có sẵn hoặc vẽ trên lưới rồi bấm nhận dạng. Mô hình trả nhãn và các điểm số cao nhất; các điểm này chưa được hiệu chuẩn thành xác suất đúng trong môi trường thực tế. Web không nhận trực tiếp ảnh chụp điện thoại như một hệ thống OCR hoàn chỉnh.')
    r.p('Model web xuất 100 cây từ selected.joblib, duyệt cây bằng JavaScript và dùng float32 như triển khai sklearn. Kiểm tra 1.471 fixtures gồm 1.437 development và 34 đầu vào tổng hợp cho sai khác xác suất JS/Python bằng 0; đây là kiểm tra tương thích suy luận, không phải một tập test accuracy mới.')

    r.page('4.3 Kết luận và giới hạn')
    r.p('Đồ án đã xây dựng bài toán tối ưu hữu hạn bốn biến, đặt ngân sách, so sánh hai thuật toán tìm kiếm và chọn nghiệm theo đánh đổi CV–dung lượng. Thí nghiệm gốc cung cấp lịch sử đủ 180 trials, còn nghiên cứu mở rộng bổ sung 450 trials trên development để khảo sát ngân sách và startup.')
    r.p('Đại diện Random đạt CV Macro-F1 0,97485, cao hơn RF mặc định 0,97062. Tuy nhiên, với epsilon 0,005, lựa chọn cuối vẫn là RF mặc định 100 cây, dung lượng 4,976 MB và test accuracy 97,22%. Kết quả này thể hiện đúng quy tắc đã đặt, không chứng minh mô hình cuối được cải tiến so với baseline. Mức giảm 40,28% chỉ là so với đại diện Random tốt nhất.')
    r.p('TPE startup 5 có kết quả CV trung bình tốt hơn trong phần mở rộng năm seeds, nhưng nghiên cứu chưa dùng để thay lựa chọn cuối hoặc đánh giá test mới. Số seeds ít, một bộ dữ liệu nhỏ và việc tìm nhiều cấu hình trên cùng folds giới hạn sức mạnh kết luận. Thời gian chạy có biến động lớn nên chưa thể khẳng định ưu thế tốc độ.')
    r.p('Train accuracy 100% chỉ phản ánh khả năng khớp dữ liệu đã học. Validation và test thấp hơn, còn chữ tự vẽ có thể nằm ngoài phân phối huấn luyện. Đồ án không tuyên bố hiệu năng của một sản phẩm nhận dạng chữ viết thực tế hoặc tính tối ưu toàn cục trong 560 cấu hình.')
    r.h('4.4 Hướng phát triển',2)
    r.p('Một nghiên cứu tiếp theo có thể dùng nested CV hoặc nhiều cách chia dữ liệu để đánh giá ảnh hưởng của chọn siêu tham số, mở rộng ngân sách và đo runtime trong điều kiện tải máy kiểm soát. Nếu bổ sung tiền xử lý chữ tự vẽ hoặc hiệu chuẩn xác suất, cần một protocol được khóa trước và dữ liệu đánh giá mới. Đây là đề xuất, chưa phải kết quả đã thực hiện.')
    r.p('Kho GitHub lưu các checkpoint theo thời điểm làm thật; không tạo lùi ngày để mô phỏng tiến độ năm tuần. Báo cáo này không thay xác nhận giảng viên đã duyệt, sinh viên đã tự chạy hoặc đã nộp trên LMS. Những trạng thái đó chỉ được ghi khi có xác nhận thực tế.')

    # IEEE numbered primary references, no donor project sources.
    r.page('TÀI LIỆU THAM KHẢO',1)
    refs=[
        ('E. Alpaydin and C. Kaynak, “Optical Recognition of Handwritten Digits,” UCI Machine Learning Repository, 1998, doi: 10.24432/C50P49.', 'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits'),
        ('scikit-learn developers, “load_digits,” scikit-learn 1.8 documentation. Accessed: Nov. 27, 2026.', 'https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_digits.html'),
        ('L. Breiman, “Random forests,” Machine Learning, vol. 45, pp. 5–32, Oct. 2001, doi: 10.1023/A:1010933404324.', 'https://doi.org/10.1023/A:1010933404324'),
        ('scikit-learn developers, “RandomForestClassifier,” scikit-learn 1.8 documentation. Accessed: Nov. 27, 2026.', 'https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html'),
        ('J. Bergstra and Y. Bengio, “Random search for hyper-parameter optimization,” Journal of Machine Learning Research, vol. 13, pp. 281–305, 2012.', 'https://www.jmlr.org/papers/v13/bergstra12a.html'),
        ('J. S. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, “Algorithms for hyper-parameter optimization,” in Advances in Neural Information Processing Systems, vol. 24, pp. 2546–2554, 2011.', 'https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization'),
        ('T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A next-generation hyperparameter optimization framework,” in Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining, pp. 2623–2631, 2019, doi: 10.1145/3292500.3330701.', 'https://doi.org/10.1145/3292500.3330701'),
        ('Optuna developers, “TPESampler,” Optuna 5.0.0 documentation. Accessed: Nov. 27, 2026.', 'https://optuna.readthedocs.io/en/v5.0.0/reference/samplers/generated/optuna.samplers.TPESampler.html'),
        ('scikit-learn developers, “f1_score,” scikit-learn 1.8 documentation. Accessed: Nov. 27, 2026.', 'https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.f1_score.html'),
        ('scikit-learn developers, “log_loss,” scikit-learn 1.8 documentation. Accessed: Nov. 27, 2026.', 'https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.log_loss.html'),
        ('L. Q. Tiến, “Digits Random Forest Optimization,” source code and frozen experimental results, baseline commit 7d44156, 2026. Accessed: Nov. 27, 2026.', 'https://github.com/TienMoi79/digits-rf-optimization/tree/7d44156'),
    ]
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    for i,(text,url) in enumerate(refs,1):
        p=r.p(f'[{i}] '+text+' [Online]. Available: ',style='SK Nguồn')
        if i==7:p.paragraph_format.page_break_before=True
        link=OxmlElement('w:hyperlink');link.set(qn('r:id'),r.doc.part.relate_to(url,RT.HYPERLINK,is_external=True))
        run_=OxmlElement('w:r');rp=OxmlElement('w:rPr');col=OxmlElement('w:color');col.set(qn('w:val'),'2A687D');rp.append(col);run_.append(rp)
        tx=OxmlElement('w:t');tx.text=url;run_.append(tx);link.append(run_);p._p.append(link)

    # APPENDICES — precise paths permit checking every chart and claim.
    r.page('PHỤ LỤC',1)
    r.h('A.1 Nguồn số liệu và hình ảnh',2)
    r.p('Các đường dẫn dưới đây tính từ gốc repository [11]. Số liệu được làm tròn khi trình bày; CSV/JSON giữ độ chính xác gốc. Báo cáo không sửa các artifact nguồn để phù hợp phần diễn giải.')
    r.table('Bảng A.1. Nguồn kiểm chứng của các nhóm kết quả',['Nội dung','Tệp hoặc thư mục nguồn'],[
        ['Config và cách chia','configs/experiment.json; results/split_indices.json'],['Train/validation/test','results/report_evaluation_2026-11-03/'],
        ['180 trials và lựa chọn','results/trials.csv; selection.json; best_per_seed.csv'],['450 trials mở rộng','results/robustness/'],
        ['Train accuracy độc lập','results/train_accuracy_audit_2026-11-03/audit.json'],['Đường cong và audit','results/curve_audit/'],
        ['Tái lập và phân tích lỗi','results/week10_2026-11-09/'],['Model gốc','results/models/selected.joblib']],[5.2,10.8],align_left=(0,1),size=11)
    r.table('Bảng A.2. Đối chiếu hình với thư mục lưu',['Ảnh','Tệp dưới figures/'],[
        ['1.1; 2.1','report_2026-11-27/digits_examples.png; experiment_pipeline.png'],
        ['3.1; 3.2','train_validation_two_lines/accuracy.png; log_loss.png'],['3.3; 3.4','search_history.png; quality_size_tradeoff.png'],
        ['3.5; 3.6','robustness/search_budget.png; startup_ablation.png'],['3.7; 3.8; 3.9','week10_2026-11-09/confusion_matrix.png; selected_test_errors.png; latency_stability.png']],[3,13],align_left=(0,1),size=11)

    r.page('A.2 Mã nguồn và cách kiểm chứng')
    r.p('Repository public: https://github.com/TienMoi79/digits-rf-optimization. README là điểm bắt đầu; docs/huong-dan-ky-thuat.md giải thích công thức và protocol. Mã nguồn thí nghiệm ở src/, notebook ở notebooks/, logs ở results/, hình ở figures/ và giao diện ở web/.')
    r.p('Cài môi trường đầy đủ một lần từ gốc project:')
    r.p('powershell -ExecutionPolicy Bypass -File .\\setup.ps1 -Notebook',left=True,size=11)
    r.p('Kiểm chứng kết quả gốc và các số liệu báo cáo:')
    for cmd in [r'.\.venv\Scripts\python.exe -m src.validation',r'.\.venv\Scripts\python.exe -m src.report_facts --check',r'.\.venv\Scripts\python.exe -m unittest discover -s tests -v']:
        r.p(cmd,left=True,size=11)
    r.p('Kiểm tra bản xuất web và dự đoán JavaScript:')
    r.p(r'.\.venv\Scripts\python.exe scripts/export_web_data.py --check',left=True,size=11)
    r.p('node web/tests/check-model.mjs',left=True,size=11)
    r.p('Các kiểm tra này đọc artifact đã lưu hoặc kiểm chứng suy luận; không yêu cầu chạy lại toàn bộ HPO chỉ để xem bài. Khi tái lập thí nghiệm đầy đủ, dùng thư mục output mới theo hướng dẫn để giữ nguyên dữ liệu gốc.')
    r.h('A.3 Checkpoint và trách nhiệm xác nhận',2)
    r.p('PROGRESS.md cùng report/checkpoints/ lưu các mốc tuần học 8, 9, 10 theo ngày làm thật, bên cạnh lịch sử phát triển trước đó. report/reviews/ lưu phản hồi và kiểm chứng; report/updates/ ghi các cập nhật tài liệu và hình. Các bản Word ngày 03/11 và 09/11 được giữ nguyên để đối chiếu.')
    r.p('Các phiên chạy tự động có hỗ trợ Codex được ghi rõ trong bằng chứng. Sinh viên cần tự mở bài, chạy các lệnh cần thiết và chuẩn bị giải thích biến quyết định, mục tiêu, ràng buộc, quy tắc epsilon, ý nghĩa train 100% và sự khác nhau giữa fold SD với search-seed SD. Báo cáo không xác nhận thay giảng viên hoặc sinh viên về việc duyệt và nộp bài.')
    r.p('Kết thúc báo cáo — cập nhật '+DATE+'.',center=True,italic=True,size=11)
    paths=r.save(output)
    manifest={
        'created_date':'2026-11-27','author':AUTHOR,'student_id':STUDENT_ID,
        'template':'scripts/templates/van_lang_report_base.docx',
        'template_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
        'output':DEFAULT.relative_to(ROOT).as_posix(),
        'figure_sources':paths,'tables':len(r.tables),'figures':len(r.figures),
        'scope':'Report-only compilation from frozen results; no model fitting or new test inference.',
        'template_parts_restored_before_word_refresh':PRESERVE,
        'finalization':'Microsoft Word updates native TOCs/page caches and normalizes style IDs; see scripts/finalize_report_word.ps1.',
    }
    manifest_path=ROOT/'report/updates/final-report-sources-2026-11-27.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=DEFAULT)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);build(args.output.resolve())
