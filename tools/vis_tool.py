#!/usr/bin/env python3
"""
vis_tool.py

Unified optional visualization/simulation tool for nuScenes-like infos and MMDet3D/CMT-style pkl outputs.

Integrated features:
1. Load nuScenes-like point cloud and render BEV points + GT/pred boxes within a cropped BEV range.
2. Optionally project GT/pred 3D boxes back to original camera images using lidar2img or camera calibration.
3. Optionally simulate detection predictions from GT using metric-like perturbations (mATE/mASE/mAOE/mAVE)
   and mAP-like missing/false-positive behavior.
4. Randomly drop some predictions to make visualization more realistic.
5. Build route-map / trajectory visualization using real instance ids from infos, then inject IDS-like switches
   for qualitative tracking analysis according to tracking metrics.
6. Optional presets for your Ours-old (Ch.3) and Ours-new (Ch.4) detection and tracking metrics.

This script is designed to be tolerant to slight schema variation in infos pkl.

Extended from the user-provided original version: pkl reading now supports img_bbox, real pkl predictions can use class-aware NMS, and visualization uses GT-green plus nuScenes 10-class prediction colors.
"""

import argparse
import math
import os
import pickle
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mmcv
import numpy as np


# ------------------------------------------------------------
# Visualization colors: GT is fixed green, predictions are class-colored
# ------------------------------------------------------------
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
PRED_CLASS_COLORS = {
    0: '#1f77b4',  # car
    1: '#ff7f0e',  # truck
    2: '#9467bd',  # construction_vehicle
    3: '#2ca02c',  # bus
    4: '#d62728',  # trailer
    5: '#7f7f7f',  # barrier
    6: '#e377c2',  # motorcycle
    7: '#17becf',  # bicycle
    8: '#bcbd22',  # pedestrian
    9: '#8c564b',  # traffic_cone
}
GT_COLOR = '#00cc44'

CLASS_NAME_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLASS_NAME_ALIASES = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'traffic_cone': 'traffic_cone',
    'construction_vehicle': 'construction_vehicle',
}

def normalize_class_name(name):
    name = str(name)
    if name in CLASS_NAME_TO_LABEL:
        return name
    if name in CLASS_NAME_ALIASES:
        return CLASS_NAME_ALIASES[name]
    # Fallback for nuScenes full category names or names with prefixes.
    for cls in CLASS_NAMES:
        if name.endswith(cls) or cls in name:
            return cls
    if 'trafficcone' in name.replace('_', '').lower():
        return 'traffic_cone'
    if 'pedestrian' in name.lower():
        return 'pedestrian'
    return 'car'

def class_name_to_label(name):
    return CLASS_NAME_TO_LABEL.get(normalize_class_name(name), 0)

def label_to_name(label):
    try:
        label = int(label)
    except Exception:
        return str(label)
    if 0 <= label < len(CLASS_NAMES):
        return CLASS_NAMES[label]
    return f'class_{label}'

def label_to_color(label):
    try:
        label = int(label)
    except Exception:
        label = 0
    return PRED_CLASS_COLORS.get(label, cm.get_cmap('tab20')(label % 20))

def mpl_color_to_bgr(color):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    return tuple(int(255 * c) for c in rgb[::-1])


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Unified detection/tracking visualization tool')
    p.add_argument('result_pkl', nargs='?', default=None, help='Optional test.py --out result pkl')
    p.add_argument('--infos-pkl', required=True, help='nuScenes-like infos pkl')
    p.add_argument('--data-root', default='', help='Optional data root for image/lidar path joining')
    p.add_argument('--out-dir', default='vis_tool_outputs', help='Output directory')
    p.add_argument('--mode', choices=['bev', 'image', 'route', 'all'], default='all')
    p.add_argument('--scene', default=None, help='Scene token/name for route mode; default first scene')
    p.add_argument('--start-idx', type=int, default=0, help='Start frame index')
    p.add_argument('--max-frames', type=int, default=50, help='Number of frames to process')
    p.add_argument('--score-thr', type=float, default=0.2, help='Prediction score threshold')
    p.add_argument('--pc-range', type=float, nargs=4, default=[-40, -40, 40, 40],
                   metavar=('XMIN','YMIN','XMAX','YMAX'), help='Cropped BEV range')
    p.add_argument('--show-gt', action='store_true', help='Render GT boxes/trajectories')
    p.add_argument('--show-pred', action='store_true', help='Render prediction boxes/trajectories')
    p.add_argument('--points-source', choices=['vehicle', 'infrastructure', 'none'], default='vehicle')
    p.add_argument('--point-size', type=float, default=0.2)
    p.add_argument('--point-alpha', type=float, default=0.28)
    p.add_argument('--camera', default='all', help='Camera name or all')
    p.add_argument('--seed', type=int, default=0)

    # Detection simulation / presets
    p.add_argument('--simulate-from-gt', action='store_true',
                   help='Ignore result_pkl detections and simulate detections from GT')
    p.add_argument('--det-preset', choices=['none', 'old', 'new'], default='none',
                   help='Apply Ours-old / Ours-new detection metric preset')
    p.add_argument('--sim-mate', type=float, default=None, help='Center translation noise (m)')
    p.add_argument('--sim-mase', type=float, default=None, help='Scale perturbation ratio')
    p.add_argument('--sim-maoe', type=float, default=None, help='Yaw perturbation std (rad)')
    p.add_argument('--sim-mave', type=float, default=None, help='Velocity perturbation std (m/s)')
    p.add_argument('--sim-fn-rate', type=float, default=None, help='Missing detection rate')
    p.add_argument('--sim-fp-rate', type=float, default=None, help='False positive ratio per GT')
    p.add_argument('--sim-score', type=float, default=0.75, help='Base score for simulated predictions')
    p.add_argument('--save-sim-pkl', default=None, help='Optional path to save simulated predictions pkl')

    # Realism controls
    p.add_argument('--random-drop-pred', action='store_true',
                   help='Randomly drop some predicted boxes for more realistic visualization')
    p.add_argument('--drop-prob', type=float, default=0.15, help='Random drop probability')
    p.add_argument('--keep-top1-per-class', action='store_true',
                   help='Try to keep the highest-score pred of each class')

    # Sparse4D-style anchor candidate simulation and post-processing.
    p.add_argument('--anchor-based-sim', action='store_true',
                   help='Generate multiple anchor hypotheses per GT object before filtering, similar to Sparse4D anchor/query decoding')
    p.add_argument('--anchor-candidates-per-gt', type=int, default=3,
                   help='Number of candidate anchors generated around each GT when --anchor-based-sim is enabled')
    p.add_argument('--anchor-score-decay', type=float, default=0.10,
                   help='Score decay for lower-rank anchor hypotheses of the same object')
    p.add_argument('--apply-circle-nms', '--nms', dest='apply_circle_nms', action='store_true',
                   help='Apply class-aware circle NMS to suppress duplicate anchor hypotheses / pkl predictions')
    p.add_argument('--circle-nms-radius', '--nms-radius', dest='circle_nms_radius', type=float, default=1.5,
                   help='Center-distance radius for circle NMS in meters')
    p.add_argument('--post-max-size', '--max-per-frame', dest='post_max_size', type=int, default=300,
                   help='Maximum boxes kept after score filtering / NMS, aligned with post_max_size=300 in config')
    p.add_argument('--keep-global-top1', action='store_true',
                   help='Debug option: keep only the globally highest-score prediction in each frame')

    # Tracking route-map simulation / presets
    p.add_argument('--track-preset', choices=['none', 'old', 'new', 'both'], default='both',
                   help='Apply Ours-old / Ours-new tracking preset for route-map')
    p.add_argument('--track-fn-rate', type=float, default=None, help='Miss rate in route-map simulation')
    p.add_argument('--track-pos-noise', type=float, default=None, help='Position jitter in route-map simulation')
    p.add_argument('--track-ids-total', type=int, default=None, help='Total IDS events to inject')
    p.add_argument('--min-track-len', type=int, default=3)
    p.add_argument('--no-annotate-ids', action='store_true')
    return p.parse_args()


# ------------------------------------------------------------
# Loading helpers
# ------------------------------------------------------------

def load_infos(infos_pkl: str) -> List[Dict[str, Any]]:
    with open(infos_pkl, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'infos' in obj:
        return obj['infos']
    if isinstance(obj, list):
        return obj
    raise TypeError(f'Unsupported infos format in {infos_pkl}: {type(obj)}')


def load_results(result_pkl: Optional[str]) -> Optional[List[Any]]:
    if result_pkl is None:
        return None
    results = mmcv.load(result_pkl)
    if not isinstance(results, list):
        raise TypeError(f'Expected list result pkl, got {type(results)}')
    return results


def norm_path(path: Optional[str], data_root: str = '') -> Optional[str]:
    if not path:
        return None
    candidates = [path]
    if data_root:
        candidates.append(os.path.join(data_root, path))
    for cand in candidates:
        cand = os.path.normpath(cand)
        if os.path.exists(cand):
            return cand
    return None


# ------------------------------------------------------------
# Generic schema helpers
# ------------------------------------------------------------

def get_pred_dict(item: Any) -> Dict[str, Any]:
    # Compatible with common MMDet3D/Sparse4D result formats:
    # results[i]['pts_bbox'], results[i]['img_bbox'], or a direct prediction dict.
    if isinstance(item, dict):
        if 'pts_bbox' in item:
            return item['pts_bbox']
        if 'img_bbox' in item:
            return item['img_bbox']
        if 'bbox_results' in item:
            return item['bbox_results']
        return item
    return {}


def to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if hasattr(x, 'tensor'):
        x = x.tensor
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    elif hasattr(x, 'cpu') and hasattr(x, 'numpy'):
        x = x.cpu().numpy()
    return np.asarray(x)


def get_boxes_scores_labels_ids(pred: Dict[str, Any]):
    boxes = None
    for k in ['boxes_3d', 'boxes_3d_det', 'boxes', 'bbox']:
        if k in pred:
            boxes = to_numpy(pred[k])
            break

    scores = None
    for k in ['scores_3d', 'scores', 'score', 'cls_scores']:
        if k in pred:
            scores = to_numpy(pred[k])
            break

    labels = None
    for k in ['labels_3d', 'labels', 'label']:
        if k in pred:
            labels = to_numpy(pred[k])
            break

    ids = None
    for k in ['track_ids', 'ids', 'instance_ids', 'track_id']:
        if k in pred:
            ids = to_numpy(pred[k])
            break

    return boxes, scores, labels, ids


def get_gt_boxes(info: Dict[str, Any]) -> np.ndarray:
    for k in ['gt_boxes', 'gt_bboxes_3d']:
        if k in info:
            arr = np.asarray(info[k])
            if arr.ndim == 2 and arr.shape[1] >= 7:
                return arr[:, :10] if arr.shape[1] >= 10 else arr[:, :7]
    return np.zeros((0, 7), dtype=np.float32)


def get_gt_labels(info: Dict[str, Any], n: int) -> np.ndarray:
    """Return numeric detection labels for GT boxes.

    nuScenes info files may store class labels either as numeric labels
    (gt_labels_3d / gt_labels / labels) or as strings (gt_names).
    The simulation path needs numeric labels, otherwise all simulated boxes
    fall back to class 0 and appear in one color.
    """
    for k in ['gt_labels_3d', 'gt_labels', 'labels']:
        if k in info:
            arr = np.asarray(info[k])
            if len(arr) == n:
                if arr.dtype.kind in {'U', 'S', 'O'}:
                    return np.asarray([class_name_to_label(x) for x in arr], dtype=np.int64)
                return arr.astype(np.int64)

    for k in ['gt_names', 'names', 'gt_classes']:
        if k in info:
            arr = np.asarray(info[k])
            if len(arr) == n:
                return np.asarray([class_name_to_label(x) for x in arr], dtype=np.int64)

    return np.zeros((n,), dtype=np.int64)


def get_gt_ids(info: Dict[str, Any], n: int) -> np.ndarray:
    for k in ['gt_ids', 'gt_instance_ids', 'gt_track_ids', 'instance_inds', 'instance_ids', 'instance_tokens']:
        if k in info:
            arr = np.asarray(info[k])
            if len(arr) == n:
                return arr
    return np.arange(n)


def get_scene_key(info, default_idx):
    for k in ['scene_token', 'scene_name', 'log_id', 'scene_id']:
        if k in info:
            return str(info[k])
    return f'scene_{default_idx:03d}'


def get_timestamp(info, default_val):
    for k in ['timestamp', 'lidar_timestamp', 'time_stamp']:
        if k in info:
            return info[k]
    return default_val


# ------------------------------------------------------------
# Geometry / drawing helpers
# ------------------------------------------------------------

def in_range_xy(x, y, pc_range):
    xmin, ymin, xmax, ymax = pc_range
    return xmin <= x <= xmax and ymin <= y <= ymax


def box_corners_3d(box: Sequence[float]) -> np.ndarray:
    x, y, z, dx, dy, dz, yaw = box[:7]
    dx, dy, dz = abs(dx), abs(dy), abs(dz)
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    corners = np.array([
        [ hx,  hy, -hz], [ hx, -hy, -hz], [-hx, -hy, -hz], [-hx,  hy, -hz],
        [ hx,  hy,  hz], [ hx, -hy,  hz], [-hx, -hy,  hz], [-hx,  hy,  hz],
    ], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    corners = corners @ rot.T
    corners += np.array([x, y, z], dtype=np.float32)
    return corners


def box_corners_xy(box: Sequence[float]) -> np.ndarray:
    return box_corners_3d(box)[:4, :2]


def draw_bev_boxes(ax, boxes, labels=None, scores=None, linewidth=1.3,
                   linestyle='-', score_text=False, color_prefix='C', alpha=0.95,
                   is_gt=False):
    """Draw BEV boxes.

    GT is always green dashed; predictions use class-specific colors.
    """
    if boxes is None:
        return
    for j, box in enumerate(boxes):
        if len(box) < 7:
            continue
        x, y = float(box[0]), float(box[1])
        corners = box_corners_xy(box)
        poly = np.vstack([corners, corners[0]])
        if is_gt or linestyle == '--' or color_prefix == 'GT':
            color = GT_COLOR
            draw_style = '--'
        else:
            color = label_to_color(labels[j]) if labels is not None and j < len(labels) else label_to_color(0)
            draw_style = linestyle
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=linewidth, linestyle=draw_style, alpha=alpha)
        front_mid = (corners[0] + corners[1]) / 2.0
        ax.plot([x, front_mid[0]], [y, front_mid[1]], color=color, linewidth=linewidth, linestyle=draw_style, alpha=alpha)
        if score_text and scores is not None:
            ax.text(x, y, f'{scores[j]:.2f}', color=color, fontsize=7,
                    bbox=dict(facecolor='white', alpha=0.55, edgecolor='none', pad=0.6))


def add_bev_legend(ax, pred_labels=None):
    """Legend rule:
    - GT is green dashed.
    - Predictions are represented directly by nuScenes class colors.
    - No standalone black "Pred" item is shown.
    """
    handles = [
        Line2D([0], [0], color=GT_COLOR, lw=2.0, linestyle='--', label='GT'),
    ]
    if pred_labels is not None and len(pred_labels) > 0:
        used_labels = sorted(set(int(x) for x in np.asarray(pred_labels).reshape(-1).tolist()))
    else:
        used_labels = list(range(len(CLASS_NAMES)))
    for lb in used_labels:
        handles.append(Line2D([0], [0], color=label_to_color(lb), lw=2.2,
                              linestyle='-', label=label_to_name(lb)))
    ax.legend(handles=handles, fontsize=7.5, loc='upper right', framealpha=0.88)

def project_points(points_lidar: np.ndarray, lidar2img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate([points_lidar[:, :3], np.ones((len(points_lidar), 1))], axis=1)
    pts_img = points_h @ lidar2img.T
    depth = pts_img[:, 2].copy()
    valid = depth > 1e-5
    pts_img[:, 0] /= np.maximum(depth, 1e-5)
    pts_img[:, 1] /= np.maximum(depth, 1e-5)
    return pts_img[:, :2], valid


def draw_projected_box(img: np.ndarray, box: Sequence[float], lidar2img: np.ndarray,
                       color=(0, 255, 255), thickness=2) -> bool:
    corners = box_corners_3d(box)
    pts, valid = project_points(corners, lidar2img)
    if valid.sum() < 4:
        return False
    h, w = img.shape[:2]
    if pts[:, 0].max() < -w or pts[:, 0].min() > 2 * w or pts[:, 1].max() < -h or pts[:, 1].min() > 2 * h:
        return False
    pts = pts.astype(np.int32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        if valid[a] and valid[b]:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thickness, lineType=cv2.LINE_AA)
    return True


def get_lidar2img_from_cam(cam_info: Dict[str, Any]) -> Optional[np.ndarray]:
    if 'lidar2img' in cam_info:
        return np.asarray(cam_info['lidar2img'], dtype=np.float64)
    if 'sensor2lidar_rotation' in cam_info and 'sensor2lidar_translation' in cam_info and 'cam_intrinsic' in cam_info:
        R_c2l = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float64)
        t_c2l = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float64).reshape(3)
        T_c2l = np.eye(4)
        T_c2l[:3, :3] = R_c2l
        T_c2l[:3, 3] = t_c2l
        T_l2c = np.linalg.inv(T_c2l)
        K = np.asarray(cam_info['cam_intrinsic'], dtype=np.float64)
        lidar2img = np.eye(4)
        lidar2img[:3, :4] = K @ T_l2c[:3, :4]
        return lidar2img
    return None


def get_cameras(info: Dict[str, Any], camera: str = 'all') -> Dict[str, Dict[str, Any]]:
    cams = info['cams'] if 'cams' in info and isinstance(info['cams'], dict) else {}
    if camera != 'all':
        return {camera: cams[camera]} if camera in cams else {}
    return cams


# ------------------------------------------------------------
# Point cloud loading
# ------------------------------------------------------------

def resolve_lidar_path(info: Dict[str, Any], points_source: str, data_root: str) -> Optional[str]:
    if points_source == 'none':
        return None
    keys = ['vehicle_lidar_path', 'lidar_path', 'pts_filename'] if points_source == 'vehicle' else ['infrastructure_lidar_path']
    for k in keys:
        if k in info:
            p = norm_path(info[k], data_root)
            if p is not None:
                return p
    return None


def load_points_xy(lidar_path: Optional[str]) -> Optional[np.ndarray]:
    if lidar_path is None or not os.path.exists(lidar_path):
        return None
    pts = np.fromfile(lidar_path, dtype=np.float32)
    if pts.size == 0:
        return None
    if pts.size % 5 == 0:
        pts = pts.reshape(-1, 5)
    elif pts.size % 4 == 0:
        pts = pts.reshape(-1, 4)
    else:
        return None
    return pts[:, :2]


# ------------------------------------------------------------
# Metric presets
# ------------------------------------------------------------

def detection_metric_preset(name: str) -> Dict[str, float]:
    name = name.lower()
    if name == 'old':
        return dict(mate=0.391, mase=0.240, maoe=0.652, mave=0.247, fn_rate=0.20, fp_rate=0.15)
    if name == 'new':
        return dict(mate=0.276, mase=0.132, maoe=0.153, mave=0.212, fn_rate=0.10, fp_rate=0.08)
    return dict(mate=0.0, mase=0.0, maoe=0.0, mave=0.0, fn_rate=0.0, fp_rate=0.0)


def tracking_metric_preset(name: str) -> Dict[str, float]:
    name = name.lower()
    if name == 'old':
        return dict(fn_rate=0.20, pos_noise=0.44, ids_total=20)
    if name == 'new':
        return dict(fn_rate=0.14, pos_noise=0.30, ids_total=15)
    return dict(fn_rate=0.0, pos_noise=0.0, ids_total=0)


# ------------------------------------------------------------
# Detection simulation and filtering
# ------------------------------------------------------------

def filter_boxes_by_range_score(boxes, scores, labels, ids, pc_range, score_thr):
    if boxes is None or len(boxes) == 0:
        return (np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64), None)
    boxes = np.asarray(boxes)
    scores = np.asarray(scores) if scores is not None else np.ones((len(boxes),), dtype=np.float32)
    labels = np.asarray(labels) if labels is not None else np.zeros((len(boxes),), dtype=np.int64)
    ids_arr = np.asarray(ids) if ids is not None else None
    keep = scores >= score_thr
    xmin, ymin, xmax, ymax = pc_range
    keep &= (boxes[:, 0] >= xmin) & (boxes[:, 0] <= xmax) & (boxes[:, 1] >= ymin) & (boxes[:, 1] <= ymax)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    ids_arr = ids_arr[keep] if ids_arr is not None else None
    return boxes, scores, labels, ids_arr


def random_drop_predictions(boxes, scores, labels, ids=None, drop_prob=0.15, seed=0, keep_top1_per_class=False):
    if boxes is None or len(boxes) == 0:
        return boxes, scores, labels, ids
    rng = np.random.default_rng(seed)
    keep = np.ones((len(boxes),), dtype=bool)
    must_keep = set()
    if keep_top1_per_class and labels is not None:
        for lb in np.unique(labels):
            idx = np.where(labels == lb)[0]
            if len(idx):
                best = idx[np.argmax(scores[idx])]
                must_keep.add(int(best))
    for i in range(len(boxes)):
        if i in must_keep:
            continue
        if rng.random() < drop_prob:
            keep[i] = False
    # Additional realism: sometimes one class is partially missing.
    if labels is not None and len(labels):
        for lb in np.unique(labels[keep]):
            idx = np.where((labels == lb) & keep)[0]
            if len(idx) > 1 and rng.random() < 0.2:
                j = int(rng.choice(idx))
                if j not in must_keep:
                    keep[j] = False
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    ids = ids[keep] if ids is not None else None
    return boxes, scores, labels, ids



def class_aware_circle_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, radius: float = 1.5) -> np.ndarray:
    """Simple class-aware circle NMS by BEV center distance.

    It is not a replacement for official evaluation NMS; it is only used to
    make anchor-based visualization readable by suppressing several nearby
    hypotheses from the same class/object.
    """
    if boxes is None or len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    keep_all = []
    for lb in np.unique(labels):
        idx = np.where(labels == lb)[0]
        idx = idx[np.argsort(-scores[idx])]
        kept = []
        kept_centers = []
        for ii in idx:
            c = boxes[ii, :2]
            if not kept_centers:
                kept.append(ii)
                kept_centers.append(c)
                continue
            d = np.linalg.norm(np.stack(kept_centers, axis=0) - c[None, :], axis=1)
            if np.all(d > radius):
                kept.append(ii)
                kept_centers.append(c)
        keep_all.extend(kept)
    keep_all = np.asarray(keep_all, dtype=np.int64)
    keep_all = keep_all[np.argsort(-scores[keep_all])]
    return keep_all

def simulate_detection_from_gt(info: Dict[str, Any], args, frame_idx: int):
    """Generate simulated detections from GT.

    When --anchor-based-sim is enabled, each GT object first produces several
    anchor/query hypotheses. Then the normal visualization pipeline applies
    score threshold, optional circle-NMS, and post_max_size filtering. This
    mimics Sparse4D-style anchor decoding: multiple queries can cover one
    physical object, but only high-confidence / non-duplicated candidates are
    retained for final display.
    """
    preset = detection_metric_preset(args.det_preset) if args.det_preset != 'none' else {}
    mate = args.sim_mate if args.sim_mate is not None else preset.get('mate', 0.0)
    mase = args.sim_mase if args.sim_mase is not None else preset.get('mase', 0.0)
    maoe = args.sim_maoe if args.sim_maoe is not None else preset.get('maoe', 0.0)
    mave = args.sim_mave if args.sim_mave is not None else preset.get('mave', 0.0)
    fn_rate = args.sim_fn_rate if args.sim_fn_rate is not None else preset.get('fn_rate', 0.0)
    fp_rate = args.sim_fp_rate if args.sim_fp_rate is not None else preset.get('fp_rate', 0.0)

    gt_boxes = get_gt_boxes(info)
    gt_labels = get_gt_labels(info, len(gt_boxes))
    gt_ids = get_gt_ids(info, len(gt_boxes))
    rng = np.random.default_rng(args.seed + frame_idx)

    sim_boxes, sim_scores, sim_labels, sim_ids = [], [], [], []
    num_hyp = max(1, int(args.anchor_candidates_per_gt)) if args.anchor_based_sim else 1

    for j, box in enumerate(gt_boxes):
        x, y = float(box[0]), float(box[1])
        if not in_range_xy(x, y, args.pc_range):
            continue
        if rng.random() < fn_rate:
            continue

        for h in range(num_hyp):
            b = box.copy().astype(np.float32)
            # Lower-ranked anchor hypotheses are noisier and lower-scored.
            noise_scale = 1.0 + 0.35 * h
            if mate > 0:
                angle = rng.uniform(0, 2 * np.pi)
                radius = rng.normal(0.0, mate * noise_scale)
                b[0] += radius * np.cos(angle)
                b[1] += radius * np.sin(angle)
            if mase > 0 and len(b) >= 6:
                scale = 1.0 + rng.normal(0.0, mase * noise_scale, size=3)
                scale = np.clip(scale, 0.4, 2.2)
                b[3:6] *= scale
            if maoe > 0 and len(b) >= 7:
                b[6] += rng.normal(0.0, maoe * noise_scale)
            if mave > 0 and len(b) >= 10:
                vel_dim = min(3, len(b) - 7)
                b[7:7 + vel_dim] += rng.normal(0.0, mave * noise_scale, size=vel_dim)

            # Score distribution: best anchor often survives; duplicate anchors may be removed by NMS.
            base = args.sim_score - h * args.anchor_score_decay
            score = float(np.clip(base - rng.uniform(0.0, 0.16), 0.01, 0.99))
            sim_boxes.append(b)
            sim_scores.append(score)
            sim_labels.append(int(gt_labels[j]))
            sim_ids.append(gt_ids[j])

    # False positives: approximate mAP degradation and background clutter.
    n_gt_kept = max(1, len(sim_boxes) // max(1, num_hyp))
    n_fp = int(round(n_gt_kept * fp_rate))
    xmin, ymin, xmax, ymax = args.pc_range
    for _ in range(n_fp):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        z = rng.uniform(-1.0, 2.0)
        dx = rng.uniform(1.0, 5.0)
        dy = rng.uniform(0.8, 2.5)
        dz = rng.uniform(1.2, 2.2)
        yaw = rng.uniform(-np.pi, np.pi)
        fp_box = np.array([x, y, z, dx, dy, dz, yaw], dtype=np.float32)
        sim_boxes.append(fp_box)
        sim_scores.append(float(np.clip(args.sim_score - rng.uniform(0.20, 0.45), 0.01, 0.75)))
        sim_labels.append(int(rng.choice(gt_labels)) if len(gt_labels) else int(rng.integers(0, 10)))
        sim_ids.append(f'fp_{frame_idx}_{len(sim_ids)}')

    if len(sim_boxes) == 0:
        return (np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=object))

    boxes = np.stack(sim_boxes, axis=0)
    scores = np.asarray(sim_scores, dtype=np.float32)
    labels = np.asarray(sim_labels, dtype=np.int64)
    ids = np.asarray(sim_ids, dtype=object)

    if args.apply_circle_nms and args.circle_nms_radius > 0:
        keep = class_aware_circle_nms(boxes, scores, labels, radius=args.circle_nms_radius)
        boxes, scores, labels, ids = boxes[keep], scores[keep], labels[keep], ids[keep]

    # Config-like top-k / post_max_size filtering.
    if len(scores) > args.post_max_size:
        order = np.argsort(-scores)[:args.post_max_size]
        boxes, scores, labels, ids = boxes[order], scores[order], labels[order], ids[order]

    return boxes, scores, labels, ids


# ------------------------------------------------------------
# BEV and image rendering
# ------------------------------------------------------------

def get_frame_pred(info, results, idx, args):
    if args.simulate_from_gt:
        boxes, scores, labels, ids = simulate_detection_from_gt(info, args, idx)
    else:
        if results is None or idx >= len(results):
            boxes, scores, labels, ids = None, None, None, None
        else:
            pred = get_pred_dict(results[idx])
            boxes, scores, labels, ids = get_boxes_scores_labels_ids(pred)
    boxes, scores, labels, ids = filter_boxes_by_range_score(boxes, scores, labels, ids, args.pc_range, args.score_thr)

    # Apply Sparse4D-style post-processing to both simulated detections and real pkl predictions.
    # For several anchor/query hypotheses around the same object, class-aware center NMS keeps
    # only the local highest-score box.
    if args.apply_circle_nms and boxes is not None and len(boxes) > 0:
        keep = class_aware_circle_nms(boxes, scores, labels, radius=args.circle_nms_radius)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        ids = ids[keep] if ids is not None else None

    if boxes is not None and scores is not None and len(scores) > args.post_max_size:
        order = np.argsort(-scores)[:args.post_max_size]
        boxes, scores, labels = boxes[order], scores[order], labels[order]
        ids = ids[order] if ids is not None else None

    if args.keep_global_top1 and boxes is not None and scores is not None and len(scores) > 0:
        order = np.array([int(np.argmax(scores))])
        boxes, scores, labels = boxes[order], scores[order], labels[order]
        ids = ids[order] if ids is not None else None

    if args.random_drop_pred:
        boxes, scores, labels, ids = random_drop_predictions(
            boxes, scores, labels, ids,
            drop_prob=args.drop_prob,
            seed=args.seed + idx,
            keep_top1_per_class=args.keep_top1_per_class,
        )
    return boxes, scores, labels, ids


def render_bev_and_collect(args, infos, results):
    out_dir = os.path.join(args.out_dir, 'bev')
    os.makedirs(out_dir, exist_ok=True)
    sim_outputs = []
    records = []

    start = args.start_idx
    end = min(start + args.max_frames, len(infos))
    xmin, ymin, xmax, ymax = args.pc_range

    for i in range(start, end):
        info = infos[i]
        gt_boxes = get_gt_boxes(info)
        gt_labels = get_gt_labels(info, len(gt_boxes))
        gt_ids = get_gt_ids(info, len(gt_boxes))
        gt_keep = np.array([in_range_xy(float(b[0]), float(b[1]), args.pc_range) for b in gt_boxes], dtype=bool) if len(gt_boxes) else np.zeros((0,), dtype=bool)
        gt_boxes_r = gt_boxes[gt_keep] if len(gt_boxes) else gt_boxes
        gt_labels_r = gt_labels[gt_keep] if len(gt_labels) else gt_labels
        gt_ids_r = gt_ids[gt_keep] if len(gt_ids) else gt_ids

        pred_boxes, pred_scores, pred_labels, pred_ids = get_frame_pred(info, results, i, args)

        if args.mode in ['bev', 'all']:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.18)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title(f'BEV frame {i}')

            if args.points_source != 'none':
                lidar_path = resolve_lidar_path(info, args.points_source, args.data_root)
                pts_xy = load_points_xy(lidar_path)
                if pts_xy is not None and len(pts_xy):
                    mask = ((pts_xy[:, 0] >= xmin) & (pts_xy[:, 0] <= xmax) &
                            (pts_xy[:, 1] >= ymin) & (pts_xy[:, 1] <= ymax))
                    pts_xy = pts_xy[mask]
                    if len(pts_xy):
                        ax.scatter(pts_xy[:, 0], pts_xy[:, 1], s=args.point_size, c='k', alpha=args.point_alpha, linewidths=0)

            if args.show_gt and len(gt_boxes_r):
                draw_bev_boxes(ax, gt_boxes_r, labels=gt_labels_r, linewidth=1.4, linestyle='--', color_prefix='GT', is_gt=True)
            if args.show_pred and pred_boxes is not None and len(pred_boxes):
                draw_bev_boxes(ax, pred_boxes, labels=pred_labels, scores=pred_scores, score_text=True)

            if args.show_gt or args.show_pred:
                add_bev_legend(ax, pred_labels if pred_labels is not None else [])

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'bev_{i:05d}.png'))
            plt.close(fig)

        # Save sim outputs if requested.
        if args.simulate_from_gt:
            sim_outputs.append(dict(
                boxes_3d=pred_boxes,
                scores_3d=pred_scores,
                labels_3d=pred_labels,
                track_ids=pred_ids,
            ))

        records.append(dict(
            frame_idx=i,
            info=info,
            gt_boxes=gt_boxes_r,
            gt_labels=gt_labels_r,
            gt_ids=gt_ids_r,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            pred_ids=pred_ids,
        ))

    if args.simulate_from_gt and args.save_sim_pkl:
        mmcv.dump(sim_outputs, args.save_sim_pkl)
        print(f'[SIM] Saved simulated prediction pkl to: {args.save_sim_pkl}')

    if args.mode in ['bev', 'all']:
        print(f'[BEV] Saved frames to: {out_dir}')
    return records


def render_images(records, args):
    out_dir = os.path.join(args.out_dir, 'image')
    os.makedirs(out_dir, exist_ok=True)

    for rec in records:
        frame_idx = rec['frame_idx']
        info = rec['info']
        cams = get_cameras(info, args.camera)
        if not cams:
            continue
        gt_boxes = rec['gt_boxes']
        pred_boxes = rec['pred_boxes']
        pred_scores = rec['pred_scores']
        pred_labels = rec.get('pred_labels', None)

        for cam_name, cam_info in cams.items():
            img_path = None
            for k in ['data_path', 'img_path', 'img_filename', 'filename']:
                if k in cam_info:
                    img_path = norm_path(cam_info[k], args.data_root)
                    break
            if img_path is None:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            lidar2img = get_lidar2img_from_cam(cam_info)
            if lidar2img is None:
                cv2.putText(img, 'No lidar2img/calibration in info', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                if args.show_gt and gt_boxes is not None:
                    for b in gt_boxes:
                        draw_projected_box(img, b, lidar2img, color=(0,255,0), thickness=2)
                if args.show_pred and pred_boxes is not None:
                    for j, b in enumerate(pred_boxes):
                        pred_color = mpl_color_to_bgr(label_to_color(pred_labels[j])) if pred_labels is not None and j < len(pred_labels) else (0,255,255)
                        ok = draw_projected_box(img, b, lidar2img, color=pred_color, thickness=2)
                        if ok and pred_scores is not None:
                            center = np.array([[b[0], b[1], b[2]]], dtype=np.float32)
                            pts, valid = project_points(center, lidar2img)
                            if valid[0]:
                                x, y = pts[0].astype(int)
                                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                                    cv2.putText(img, f'{pred_scores[j]:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, pred_color, 1, cv2.LINE_AA)
            # Image legend: GT is green; prediction boxes use nuScenes class colors.
            cv2.line(img, (20, 28), (70, 28), (0, 255, 0), 3)
            cv2.putText(img, 'GT', (80, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            safe_cam = cam_name.replace('/', '_')
            cv2.imwrite(os.path.join(out_dir, f'img_{frame_idx:05d}_{safe_cam}.jpg'), img)
    print(f'[IMAGE] Saved projected images to: {out_dir}')


# ------------------------------------------------------------
# Route-map tracking visualization based on real instance ids
# ------------------------------------------------------------

def build_scene_frames(infos, scene=None, start_idx=0, max_frames=80):
    scene_map = defaultdict(list)
    for idx, info in enumerate(infos):
        scene_map[get_scene_key(info, idx)].append((idx, info))
    if scene is None:
        scene = sorted(scene_map.keys())[0]
    frames = scene_map[scene]
    frames = sorted(frames, key=lambda x: get_timestamp(x[1], x[0]))
    frames = frames[start_idx:start_idx + max_frames]
    return scene, frames


def build_gt_tracks(frames, pc_range):
    tracks = defaultdict(list)
    for local_fidx, (global_idx, info) in enumerate(frames):
        boxes = get_gt_boxes(info)
        labels = get_gt_labels(info, len(boxes))
        tids = get_gt_ids(info, len(boxes))
        for j, box in enumerate(boxes):
            x, y = float(box[0]), float(box[1])
            if not in_range_xy(x, y, pc_range):
                continue
            tracks[str(tids[j])].append({
                'frame_idx': local_fidx,
                'global_idx': global_idx,
                'center': np.array([x, y], dtype=np.float32),
                'box': box.copy(),
                'label': int(labels[j]) if len(labels) else 0,
                'gt_id': str(tids[j]),
            })
    return tracks


def allocate_ids_events(track_lengths, total_ids):
    trans_counts = {k: max(0, v - 1) for k, v in track_lengths.items()}
    total_trans = max(1, sum(trans_counts.values()))
    alloc = {}
    remaining = total_ids
    keys = list(track_lengths.keys())
    for i, k in enumerate(keys):
        if i == len(keys) - 1:
            a = min(trans_counts[k], remaining)
        else:
            a = min(trans_counts[k], int(round(total_ids * trans_counts[k] / total_trans)))
        alloc[k] = a
        remaining -= a
    if remaining > 0:
        for k in sorted(keys, key=lambda x: trans_counts[x] - alloc[x], reverse=True):
            can = trans_counts[x] - alloc[x] if False else trans_counts[k] - alloc[k]
            add = min(can, remaining)
            alloc[k] += add
            remaining -= add
            if remaining <= 0:
                break
    return alloc


def simulate_pred_tracks(gt_tracks, fn_rate=0.2, pos_noise=0.4, ids_total=20, seed=0):
    rnd = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    track_lengths = {tid: len(obs) for tid, obs in gt_tracks.items()}
    ids_alloc = allocate_ids_events(track_lengths, ids_total)
    pred_tracks = defaultdict(list)
    ids_events = []
    pred_id_counter = 0

    for gt_tid, obs_list in gt_tracks.items():
        obs_list = sorted(obs_list, key=lambda x: x['frame_idx'])
        if not obs_list:
            continue
        n_sw = min(ids_alloc.get(gt_tid, 0), max(0, len(obs_list) - 1))
        switch_positions = set(rnd.sample(list(range(1, len(obs_list))), n_sw)) if n_sw > 0 else set()
        current_pred_id = f'pred_{pred_id_counter}'
        pred_id_counter += 1

        for k, obs in enumerate(obs_list):
            if k in switch_positions:
                old_pred_id = current_pred_id
                current_pred_id = f'pred_{pred_id_counter}'
                pred_id_counter += 1
                ids_events.append(dict(
                    gt_id=gt_tid, frame_idx=obs['frame_idx'], point=obs['center'].copy(),
                    old_pred_id=old_pred_id, new_pred_id=current_pred_id,
                ))
            if rnd.random() < fn_rate:
                continue
            noisy_center = obs['center'] + np_rng.normal(0.0, pos_noise, size=2)
            pred_tracks[gt_tid].append(dict(
                frame_idx=obs['frame_idx'], global_idx=obs['global_idx'], center=noisy_center.astype(np.float32),
                pred_id=current_pred_id, label=obs['label'], gt_id=gt_tid,
            ))
    return pred_tracks, ids_events


def plot_route_map(scene_name, gt_tracks, pred_tracks, ids_events, out_path, args, title=''):
    xmin, ymin, xmax, ymax = args.pc_range
    fig, ax = plt.subplots(figsize=(11, 11), dpi=180)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.18)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title if title else f'Track route map - {scene_name}')

    gt_count = 0
    for gt_tid, obs in gt_tracks.items():
        obs = sorted(obs, key=lambda x: x['frame_idx'])
        if len(obs) < args.min_track_len:
            continue
        pts = np.stack([o['center'] for o in obs], axis=0)
        ax.plot(pts[:, 0], pts[:, 1], '--', color='0.75', linewidth=1.2, alpha=0.85)
        ax.scatter(pts[0, 0], pts[0, 1], marker='o', s=10, c='0.6')
        ax.scatter(pts[-1, 0], pts[-1, 1], marker='s', s=10, c='0.6')
        gt_count += 1

    cmap = cm.get_cmap('tab20', 20)
    pid_to_color = {}
    def pid_color(pid):
        if pid not in pid_to_color:
            pid_to_color[pid] = cmap(len(pid_to_color) % 20)
        return pid_to_color[pid]

    pred_count = 0
    for gt_tid, obs in pred_tracks.items():
        obs = sorted(obs, key=lambda x: x['frame_idx'])
        if len(obs) < 2:
            continue
        seg = [obs[0]]
        for cur in obs[1:]:
            prev = seg[-1]
            if cur['pred_id'] != prev['pred_id'] or cur['frame_idx'] != prev['frame_idx'] + 1:
                if len(seg) >= 2:
                    pts = np.stack([o['center'] for o in seg], axis=0)
                    color = pid_color(seg[0]['pred_id'])
                    ax.plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=2.0, alpha=0.95)
                    ax.scatter(pts[0, 0], pts[0, 1], marker='^', s=18, c=[color], edgecolors='none')
                seg = [cur]
            else:
                seg.append(cur)
        if len(seg) >= 2:
            pts = np.stack([o['center'] for o in seg], axis=0)
            color = pid_color(seg[0]['pred_id'])
            ax.plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=2.0, alpha=0.95)
            ax.scatter(pts[0, 0], pts[0, 1], marker='^', s=18, c=[color], edgecolors='none')
        pred_count += 1

    for ev in ids_events:
        x, y = ev['point']
        if not in_range_xy(float(x), float(y), args.pc_range):
            continue
        ax.scatter(x, y, marker='*', s=110, c='red', edgecolors='k', linewidths=0.5, zorder=10)
        if not args.no_annotate_ids:
            txt = f"IDS\n{ev['old_pred_id']}→{ev['new_pred_id']}"
            ax.text(x + 0.8, y + 0.8, txt, color='red', fontsize=7,
                    bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=1.0))

    note = [
        f'Scene: {scene_name}', f'GT tracks: {gt_count}', f'Pred tracks: {pred_count}',
        f'IDS events: {len(ids_events)}', 'Gray dashed = GT route',
        'Colored solid = predicted route', 'Red star = identity switch point'
    ]
    ax.text(0.01, 0.99, '\n'.join(note), transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='0.8'))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_route_maps(args, infos):
    out_dir = os.path.join(args.out_dir, 'route_map')
    os.makedirs(out_dir, exist_ok=True)
    scene_name, frames = build_scene_frames(infos, scene=args.scene, start_idx=args.start_idx, max_frames=args.max_frames)
    gt_tracks = build_gt_tracks(frames, args.pc_range)
    if len(gt_tracks) == 0:
        raise RuntimeError('No GT tracks in selected scene/range. Try a larger --pc-range or another --scene.')

    presets = ['old', 'new'] if args.track_preset == 'both' else ([args.track_preset] if args.track_preset != 'none' else ['new'])
    for tag in presets:
        p0 = tracking_metric_preset(tag)
        fn_rate = args.track_fn_rate if args.track_fn_rate is not None else p0['fn_rate']
        pos_noise = args.track_pos_noise if args.track_pos_noise is not None else p0['pos_noise']
        ids_total = args.track_ids_total if args.track_ids_total is not None else p0['ids_total']
        pred_tracks, ids_events = simulate_pred_tracks(gt_tracks, fn_rate=fn_rate, pos_noise=pos_noise,
                                                       ids_total=ids_total, seed=args.seed)
        out_path = os.path.join(out_dir, f'route_map_{tag}.png')
        title = (f'{tag} route map | range={tuple(args.pc_range)} | '
                 f'fn_rate={fn_rate:.2f}, pos_noise={pos_noise:.2f}, IDS={ids_total}')
        plot_route_map(scene_name, gt_tracks, pred_tracks, ids_events, out_path, args, title)
        print(f'[ROUTE] Saved: {out_path} | GT tracks={len(gt_tracks)} | IDS markers={len(ids_events)}')


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    infos = load_infos(args.infos_pkl)
    results = load_results(args.result_pkl)

    records = None
    if args.mode in ['bev', 'image', 'all']:
        if args.mode in ['bev', 'all'] or args.mode == 'image':
            records = render_bev_and_collect(args, infos, results)
        if args.mode in ['image', 'all']:
            if records is None:
                records = render_bev_and_collect(args, infos, results)
            render_images(records, args)

    if args.mode in ['route', 'all']:
        render_route_maps(args, infos)

    print('Done.')


if __name__ == '__main__':
    main()
