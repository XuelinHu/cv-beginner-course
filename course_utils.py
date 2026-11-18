"""课程公共工具；算法步骤保留在各 Notebook 中。"""
from pathlib import Path
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent


def configure_plot_fonts():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK HK',
                 'Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']:
        if font in available:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font]
            return
    for font_path in [
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
    ]:
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
            font = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            return


configure_plot_fonts()


def setup(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    configure_plot_fonts()
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 105
    (ROOT / 'outputs').mkdir(exist_ok=True)
    return ROOT


def device():
    import torch
    requested = os.getenv('CV_DEVICE', 'auto').lower()
    if requested not in {'auto', 'cpu', 'cuda', 'mps'}:
        raise ValueError('CV_DEVICE 必须为 auto/cpu/cuda/mps')
    if requested == 'cpu':
        return torch.device('cpu')
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('请求了 CUDA，但当前 PyTorch/驱动没有可用 CUDA；改用 CV_DEVICE=cpu')
        return torch.device('cuda')
    mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if requested == 'mps':
        if not mps:
            raise RuntimeError('当前没有可用 MPS；改用 CV_DEVICE=cpu')
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if mps else 'cpu')


def show_images(images, titles=None, cmap=None):
    if isinstance(images, dict):
        titles, images = list(images), list(images.values())
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 3.5), squeeze=False)
    for i, (ax, im) in enumerate(zip(axes.flat, images)):
        ax.imshow(im, cmap=cmap or ('gray' if np.asarray(im).ndim == 2 else None))
        ax.set_title(titles[i] if titles else f'图 {i+1}')
        ax.axis('off')
    fig.tight_layout()
    return fig


def save_figure(name):
    name = Path(name).name
    if not name.endswith('.png'):
        name += '.png'
    path = ROOT / 'outputs' / name
    path.parent.mkdir(exist_ok=True)
    plt.gcf().savefig(path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    return path
