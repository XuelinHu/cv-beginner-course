"""检查课程分发前的实际内容、执行输出和素材，不运行训练。"""
import ast
import hashlib
import json
from pathlib import Path
import re

import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows, errors = [], []
    imports = set()
    paths = sorted((ROOT/'notebooks').glob('*.ipynb'))
    expected = [f'{index:02d}_' for index in range(28)]
    actual = [path.name[:3] for path in paths]
    if len(paths) != 28: errors.append(f'应有28课，实际{len(paths)}')
    if actual != expected: errors.append(f'课程编号应连续为00–27，实际为{actual}')
    for p in paths:
        nb = nbformat.read(p, as_version=4)
        nbformat.validate(nb)
        code = [c for c in nb.cells if c.cell_type == 'code']
        texts = '\n'.join(c.source for c in nb.cells if c.cell_type == 'markdown')
        images = sum('image/png' in o.get('data', {}) for c in code for o in c.outputs)
        for c in nb.cells:
            if any(ord(ch)<32 and ch not in '\n\r\t' for ch in c.source):
                errors.append(f'{p.name}: 存在意外控制字符')
        for c in code:
            tree = ast.parse(c.source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): imports.update(a.name.split('.')[0] for a in node.names)
                if isinstance(node, ast.ImportFrom) and node.module: imports.add(node.module.split('.')[0])
            if c.execution_count is None: errors.append(f'{p.name}: 未执行单元')
            for out in c.outputs:
                if out.output_type == 'error': errors.append(f'{p.name}: {out.ename}')
        if images < 1: errors.append(f'{p.name}: 没有实际嵌入图像输出')
        content = texts + '\n' + '\n'.join(c.source for c in code)
        if int(p.stem[:2]) < 14 and not any(term in content for term in ['矩阵', '张量', 'NCHW', '.shape']):
            errors.append(f'{p.name}: 缺少矩阵/张量维度解释')
        rows.append({'notebook': p.name, 'code_cells': len(code), 'image_outputs': images,
                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    for p in (ROOT/'outputs').glob('*.png'):
        with Image.open(p) as im: im.verify()
    sources = json.loads((ROOT/'assets/SOURCES.json').read_text(encoding='utf-8'))
    for item in sources['photos']:
        p = ROOT/item['path']
        if hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']: errors.append(f'素材哈希不一致: {p}')
    # 本地链接要能在学生包的相对目录中找到。
    for p in [ROOT/'README.md', ROOT/'docs/CURRICULUM.md', ROOT/'docs/SOURCES.md']:
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if '://' not in target and not (p.parent/target.split('#')[0]).exists():
                errors.append(f'{p.name}: 无效相对链接 {target}')
    report = {'notebooks': rows, 'total_notebooks': len(paths),
              'total_embedded_images': sum(r['image_outputs'] for r in rows),
              'imports': sorted(imports), 'errors': errors, 'ok': not errors}
    (ROOT/'verification').mkdir(exist_ok=True)
    (ROOT/'verification/unified-course-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()
