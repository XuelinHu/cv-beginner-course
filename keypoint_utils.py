"""真实手部/人体课程：固定模型、真实标注、网络内部张量与逐帧推理。"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import urllib.request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ZOO = '48b10fa505aeb51fbd127fc60a5c660040e823b7'
TFJS = 'c731b9ebbd6f4c9e8bf99b0df76bbdbf9c25b07f'
MMPOSE = '759b39c13fea6ba094afc1fa932f51dc1b11cbf9'
MODEL_SPECS = {
    'hand': ('handpose_estimation_mediapipe', 'handpose_estimation_mediapipe_2023feb.onnx',
             'db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c'),
    'palm': ('palm_detection_mediapipe', 'palm_detection_mediapipe_2023feb.onnx',
             '78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c'),
    'pose': ('pose_estimation_mediapipe', 'pose_estimation_mediapipe_2023mar.onnx',
             '9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f'),
    'person': ('person_detection_mediapipe', 'person_detection_mediapipe_2023mar.onnx',
               '47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f'),
}
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
              (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
              (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
COCO_TO_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def download(url, path, sha256=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print('首次下载：' + path.name, flush=True)
        part = path.with_suffix(path.suffix + '.part')
        try:
            with urllib.request.urlopen(url, timeout=60) as response, part.open('wb') as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if sha256 and hashlib.sha256(part.read_bytes()).hexdigest() != sha256:
                raise ValueError('下载哈希不一致：' + path.name)
            part.replace(path)
        except Exception as exc:
            raise RuntimeError('下载失败，请检查网络并重新运行：' + path.name) from exc
    if sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError('缓存哈希不一致：' + path.name)
    return path


def prepare_keypoints():
    """原图/视频/权重保留在忽略目录data，来源清单可公开核验。"""
    provenance = []
    expected = {r['path'].replace('\\', '/'): r['sha256'] for r in json.loads(
        (ROOT / 'reference/real-data-downloads.json').read_text(encoding='utf-8')) if 'path' in r}
    for folder, filename, sha in MODEL_SPECS.values():
        url = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{ZOO}/models/{folder}/{filename}'
        path = download(url, ROOT / 'data/models' / filename, sha)
        provenance.append({'path': path.relative_to(ROOT).as_posix(), 'url': url, 'sha256': sha})
    base = f'https://raw.githubusercontent.com/open-mmlab/mmpose/{MMPOSE}/tests/data/coco/'
    files = [(n, base + n) for n in ['000000000785.jpg', '000000040083.jpg', '000000196141.jpg',
                                   '000000197388.jpg', 'test_coco_wholebody.json']]
    files += [(n, 'https://storage.googleapis.com/mediapipe-assets/' + n)
              for n in ['thumb_up.jpg', 'pointing_up.jpg']]
    files += [('woman_hands.jpg', 'https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg'),
              ('hand_clip.mp4', f'https://raw.githubusercontent.com/tensorflow/tfjs-models/{TFJS}/hand-pose-detection/test_data/asl_hand.25fps.mp4'),
              ('pose_clip.mp4', f'https://raw.githubusercontent.com/tensorflow/tfjs-models/{TFJS}/pose-detection/test_data/pose_squats.mp4')]
    for filename, url in files:
        path = download(url, ROOT / 'data/keypoints' / filename, expected['data/keypoints/' + filename])
        provenance.append({'path': path.relative_to(ROOT).as_posix(), 'url': url,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    data = ROOT / 'data/keypoints'
    (data / 'download_manifest.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return data


def models(kind):
    sys.path.insert(0, str(ROOT / 'reference/opencv_zoo'))
    cv2.setNumThreads(2)
    if kind == 'hand':
        from mp_handpose import MPHandPose
        from mp_palmdet import MPPalmDet
        return (MPPalmDet(str(ROOT / 'data/models' / MODEL_SPECS['palm'][1]), scoreThreshold=.4),
                MPHandPose(str(ROOT / 'data/models' / MODEL_SPECS['hand'][1]), confThreshold=.5))
    from mp_persondet import MPPersonDet
    from mp_pose import MPPose
    return (MPPersonDet(str(ROOT / 'data/models' / MODEL_SPECS['person'][1]), scoreThreshold=.4),
            MPPose(str(ROOT / 'data/models' / MODEL_SPECS['pose'][1]), confThreshold=.5))


def trace_network(net, blob):
    """关闭算子融合，读取同一次真实前向的各层输出；不修改模型参数。"""
    net.enableFusion(False)
    names = net.getLayerNames()
    net.setInput(blob)
    outputs = net.forward(names)
    table = []
    for i, (name, value) in enumerate(zip(names, outputs)):
        layer = net.getLayer(net.getLayerId(name))
        table.append({'index': i, 'name': name, 'type': layer.type, 'shape': list(value.shape),
                      'weights': [list(b.shape) for b in layer.blobs]})
    return names, outputs, table


def plot_channels(values, title, save_name, limit=24, start=0):
    """按通道真实绘图；明确只显示前limit通道，不把通道平均伪装为全图。"""
    import matplotlib.pyplot as plt
    from course_utils import save_figure
    maps = np.asarray(values)[0]
    if maps.ndim != 3:
        raise ValueError('需要NCHW四维张量')
    if not 0 <= start < len(maps):
        raise ValueError('start超出通道范围')
    n = min(limit, len(maps) - start)
    cols = min(8, n)
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(cols * 1.65, np.ceil(n / cols) * 1.85),
                             squeeze=False)
    low, high = np.percentile(maps, [1, 99])
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(maps[start+i], cmap='viridis', vmin=low, vmax=high)
            ax.set_title('ch ' + str(start+i), fontsize=9)
        ax.axis('off')
    fig.suptitle(f'{title} | {tuple(values.shape)} | 通道{start}至{start+n-1} / 共{len(maps)}，共用色阶')
    fig.tight_layout()
    save_figure(save_name)


def draw_points(ax, image, points, edges, title, color='lime', numbers=True, visible=None):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    points = np.asarray(points)
    visible = np.ones(len(points), bool) if visible is None else visible
    for a, b in edges:
        if visible[a] and visible[b]:
            ax.plot(points[[a, b], 0], points[[a, b], 1], color=color, lw=1.3)
    ax.scatter(points[visible, 0], points[visible, 1], s=12, c=color)
    if numbers:
        for i in np.where(visible)[0]:
            ax.text(points[i, 0], points[i, 1], str(i), color='yellow', fontsize=7)
    ax.set_title(title)
    ax.axis('off')


def joint_angle(a, b, c):
    u, v = np.asarray(a) - b, np.asarray(c) - b
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8), -1, 1))))


def finger_state(points):
    """明确的几何规则，不声称是训练过的手语分类器。"""
    p = np.asarray(points)[:, :2]
    angles = np.array([joint_angle(p[a], p[b], p[c]) for a, b, c in
                       [(1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]])
    extended = angles > 155
    names = {(True, False, False, False, False): 'thumb_up',
             (False, True, False, False, False): 'pointing_up',
             (True, True, True, True, True): 'open_palm',
             (False, True, True, False, False): 'victory',
             (False, False, False, False, False): 'fist'}
    return angles, extended, names.get(tuple(extended), 'other/uncertain')


def export_inventory(table, path):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=['index', 'type', 'shape', 'weights', 'name'])
        writer.writeheader()
        writer.writerows(table)


def infer_video(path, detector, estimator, kind, output_prefix, max_frames=80, stride=1):
    """顺序解码真实视频；每帧重新检测，无跨帧身份跟踪，无伪造补点。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError('视频无法打开：' + str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    records, snapshots = [], []
    try:
        for index in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride:
                continue
            rois = detector.infer(frame)
            result = estimator.infer(frame, rois[0]) if len(rois) else None
            row = {'frame': index, 'time_s': index / fps if fps > 0 else None,
                   'detected': result is not None, 'roi_count': len(rois)}
            points = None
            if result is not None:
                if kind == 'hand':
                    points = result[4:67].reshape(21, 3)
                    angles, state, label = finger_state(points)
                    row.update(confidence=float(result[-1]), label=label,
                               angles=angles.tolist(), keypoints=points.tolist())
                else:
                    points = result[1][:33]
                    reliable = np.all(points[[23,24,25,26,27,28], 3:5] > .5)
                    angle = float(np.mean([joint_angle(points[a, :2], points[b, :2], points[c, :2])
                                     for a, b, c in [(23, 25, 27), (24, 26, 28)]])) if reliable else None
                    row.update(confidence=float(result[-1]), knee_angle=angle,
                               keypoints=points.tolist())
            records.append(row)
            if index % 10 == 0:
                snapshots.append((index, frame.copy(), points, row))
    finally:
        cap.release()
    if not records:
        raise RuntimeError('视频没有可解码帧')
    prefix = Path(output_prefix)
    prefix.with_suffix('.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    flat = [{k: v for k, v in r.items() if k not in {'keypoints', 'angles'}} for r in records]
    fields = sorted({k for r in flat for k in r})
    with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    return records, snapshots, fps
