"""NEU-DET 教学数据：固定来源、VOC标注、按原图哈希分组划分。"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
COMMIT = '351bee21b56bbf97a1e1d18262bceb202eb7ef41'
ARCHIVE_SHA256 = 'e0f360b032342399d18ca14328e9eff905e7ac18cb5b1bc91f1fd20512e267d6'
URL = 'https://codeload.github.com/allanwong/NEU-DET-Steel-Surface-Defect-Detection/zip/' + COMMIT
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
CHINESE = ['龟裂', '夹杂', '斑块', '麻点', '氧化铁皮压入', '划痕']


def prepare_neu(root=None):
    """首次下载镜像；以后离线复用。镜像不是原作者的官方分割。"""
    root = Path(root) if root else ROOT / 'data' / 'neu_det'
    manifest = root / 'manifest.json'
    if manifest.exists():
        result = json.loads(manifest.read_text(encoding='utf-8'))
        for row in result['samples']:
            if not (root / row['image']).is_file() or not (root / row['annotation']).is_file():
                raise FileNotFoundError('缓存不完整；请检查 data/neu_det 中的图像与XML')
        return root, result
    cache = ROOT / 'data' / '.cache'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'neu-mirror-351bee21.zip'
    if not archive.exists():
        print('首次下载 NEU-DET 固定版本镜像（约110MB），完成后可以离线运行。', flush=True)
        part = archive.with_suffix('.part')
        try:
            with urllib.request.urlopen(URL, timeout=60) as response, part.open('wb') as out:
                shutil.copyfileobj(response, out)
            part.replace(archive)
        except Exception:
            raise RuntimeError('下载未完成。可稍后重试，或将已下载的固定版本zip放到 data/.cache/neu-mirror-351bee21.zip') from None
    if not zipfile.is_zipfile(archive):
        raise ValueError('下载内容不是ZIP；请检查网络，不要把登录页当成数据')
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_sha256 != ARCHIVE_SHA256:
        raise ValueError('NEU镜像版本哈希不符，请检查下载内容')
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'annotations').mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            parts = Path(member.filename).parts
            if len(parts) < 3:
                continue
            folder, name = parts[-2:]
            if folder in {'IMAGES', 'Validation_Images'} and name.lower().endswith(('.jpg', '.png', '.bmp')):
                target = root / 'images' / name
            elif folder in {'ANNOTATIONS', 'Validation_Annotations'} and name.lower().endswith('.xml'):
                target = root / 'annotations' / name
            else:
                continue
            target.write_bytes(zipped.read(member))
    rows = []
    for path in sorted((root / 'annotations').glob('*.xml')):
        tree = ET.parse(path).getroot()
        filename = tree.findtext('filename')
        image_path = root / 'images' / filename
        if not image_path.exists():
            candidates = list((root / 'images').glob(path.stem + '.*'))
            if len(candidates) != 1:
                raise ValueError(f'标注没有唯一对应图像：{path.name}')
            image_path = candidates[0]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'图片不可读：{image_path.name}')
        height, width = image.shape
        objects = []
        for obj in tree.findall('object'):
            label = obj.findtext('name')
            box = obj.find('bndbox')
            # VOC常见1-based inclusive -> 0-based半开区间；右下角保持为边界。
            x0, y0 = max(0, int(box.findtext('xmin')) - 1), max(0, int(box.findtext('ymin')) - 1)
            x1, y1 = min(width, int(box.findtext('xmax'))), min(height, int(box.findtext('ymax')))
            if label not in CLASSES or not (x0 < x1 and y0 < y1):
                raise ValueError(f'无效类别或框：{path.name}')
            objects.append({'label': label, 'xyxy': [x0, y0, x1, y1]})
        labels = {o['label'] for o in objects}
        # 部分图像还含其他类别；保留全部标注，按数据文件名的主类别教学分类。
        label = path.stem.rsplit('_', 1)[0]
        if label not in CLASSES or label not in labels:
            raise ValueError(f'主类别没有对应标注：{path.name}')
        boxes = np.array([o['xyxy'] for o in objects if o['label'] == label])
        union = [int(boxes[:, 0].min()), int(boxes[:, 1].min()),
                 int(boxes[:, 2].max()), int(boxes[:, 3].max())]
        rows.append({'id': path.stem, 'image': str(image_path.relative_to(root)).replace('\\', '/'),
                     'annotation': str(path.relative_to(root)).replace('\\', '/'),
                     'label': label, 'label_id': CLASSES.index(label), 'width': width, 'height': height,
                     'objects': objects, 'union_xyxy': union, 'all_labels': sorted(labels),
                     'classification_protocol': 'filename primary class; other defects retained',
                     'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
    if len(rows) != 1800:
        raise ValueError(f'应有1800张图像，实际{len(rows)}；不接受不完整缓存')
    groups = {}
    for row in rows:
        groups.setdefault(row['pixel_sha256'], []).append(row)
    if any(len({r['label'] for r in group}) != 1 for group in groups.values()):
        raise ValueError('发现像素相同但标签不同的样本，需人工检查')
    ids = sorted(groups)
    labels = [groups[k][0]['label'] for k in ids]
    train, remaining = train_test_split(ids, test_size=.30, stratify=labels, random_state=42)
    valid, test = train_test_split(remaining, test_size=.50,
                                  stratify=[groups[k][0]['label'] for k in remaining], random_state=42)
    membership = {k: split for split, keys in [('train', train), ('val', valid), ('test', test)] for k in keys}
    for row in rows:
        row['split'] = membership[row['pixel_sha256']]
    for split in ['train', 'val', 'test']:
        with (root / (split + '.csv')).open('w', encoding='utf-8', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=['id', 'image', 'annotation', 'label', 'label_id', 'split', 'pixel_sha256'])
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows if row['split'] == split)
    result = {'dataset': 'NEU-DET', 'official': 'https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm',
              'mirror': URL, 'mirror_commit': COMMIT, 'archive_sha256': archive_sha256,
              'split_protocol': 'course seed42, stratified by class, grouped by exact decoded pixel hash, 70/15/15',
              'not_official_split': True, 'classes': CLASSES, 'samples': rows}
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return root, result
