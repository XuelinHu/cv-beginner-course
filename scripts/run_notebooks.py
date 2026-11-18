"""在独立新内核中顺序执行 Notebook，并保存可复核的输出。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', default='*.ipynb', help='如 0[0-4]*.ipynb；默认为全部28课')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'auto', 'cuda', 'mps'])
    parser.add_argument('--timeout', default=300, type=int, help='单个单元格的超时秒数')
    parser.add_argument('--report', default='execution.json')
    args = parser.parse_args()
    os.environ['CV_DEVICE'] = args.device
    os.environ['PYTHONUTF8'] = '1'
    os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
    paths = sorted((ROOT/'notebooks').glob(args.pattern))
    if not paths:
        raise SystemExit('没有找到 Notebook，请检查 --pattern')
    results = []
    (ROOT/'verification').mkdir(exist_ok=True)
    for path in paths:
        start = time.perf_counter()
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        print(f'运行 {path.name}', flush=True)
        item = {'notebook': path.name, 'ok': False}
        try:
            client = NotebookClient(nb, timeout=args.timeout, kernel_name='python3',
                                    resources={'metadata': {'path': str(path.parent)}})
            # 确保独立内核也使用当前环境，避免误用系统中另一个 Python。
            client.create_kernel_manager()
            client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
            client.execute()
            nbformat.write(nb, path)
            code = [c for c in nb.cells if c.cell_type == 'code']
            images = sum('image/png' in out.get('data', {}) for c in code for out in c.outputs)
            if any(c.execution_count is None for c in code):
                raise RuntimeError('存在未执行的代码单元')
            if not images:
                raise RuntimeError('没有嵌入图像输出，请检查绘图后端')
            item.update(ok=True, code_cells=len(code), image_outputs=images)
        except Exception as exc:
            item['error'] = str(exc)
            failure = ROOT/'verification'/f'{path.stem}.failed.ipynb'
            nbformat.write(nb, failure)
            print(f'失败 {path.name}: {exc}', flush=True)
        item['seconds'] = round(time.perf_counter()-start, 2)
        results.append(item)
        report = {'python': sys.version, 'executable': sys.executable,
                  'verified_at': datetime.now(timezone.utc).isoformat(),
                  'platform': platform.platform(), 'requested_device': args.device,
                  'results': results, 'all_ok': all(r['ok'] for r in results)}
        (ROOT/'verification'/args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"完成 {path.name}：{item['ok']}，{item['seconds']} 秒", flush=True)
    return 0 if all(r['ok'] for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
