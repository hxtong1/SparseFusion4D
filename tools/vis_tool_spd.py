import mmcv
import argparse
import os
import glob
import cv2
import copy
import json
import pickle
import sys
import struct
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
# from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingConfig
from nuscenes.eval.detection.utils import category_to_detection_name
try:
    from transform_box_veh2inf import veh2inf_convert, read_json
except Exception:
    veh2inf_convert = None
    def read_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


cams = ['CAM_FRONT',
 'CAM_FRONT_RIGHT',
 'CAM_BACK_RIGHT',
 'CAM_BACK',
 'CAM_BACK_LEFT',
 'CAM_FRONT_LEFT']

color_mapping = [
    np.array([1.0, 0.0, 0.0]),   # 鲜艳的红色
    np.array([1.0, 0.078, 0.576]), # 鲜艳的粉色
    np.array([0.0, 0.0, 1.0]),   # 鲜艳的蓝色
    np.array([1.0, 1.0, 0.0]),   # 鲜艳的黄色
    np.array([1.0, 0.647, 0.0]), # 鲜艳的橙色
    np.array([0.502, 0.0, 0.502]), # 鲜艳的紫色
    np.array([0.0, 1.0, 1.0]),   # 鲜艳的青色
    np.array([1.0, 0.0, 1.0]),   # 鲜艳的洋红色
    np.array([0.0, 1.0, 0.502]), # 鲜艳的青绿色
    np.array([1.0, 0.843, 0.0])  # 鲜艳的金色
]

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams

from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.utils.data_classes import PointCloud
from scipy.linalg import polar

class CustomLidarPointCloud(PointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 4

    @classmethod
    def from_file(cls, file_name: str) -> 'CustomLidarPointCloud':
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """

        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 4))[:, :cls.nbr_dims()]
        return cls(points.T)

def iterative_closest_point(A, num_iterations=100):
    R = A.copy()

    for _ in range(num_iterations):
        U, _ = polar(R)
        R = U

    return R

def visualize_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     nsweeps: int = 1,
                     conf_th: float = 0.15,
                     eval_range: list = [-53.0, -53.0, 53.0, 53.0],
                     verbose: bool = True,
                     savepath: str = None,
                     ax=None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.tracking_score
        box_est.tracking_id = box_est_global.tracking_id

    # Get point cloud in lidar frame.
    # pc, _ = CustomLidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    # points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    # dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    # colors = np.minimum(1, dists / eval_range[2])
    # ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=1)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            c = 'r'
            if hasattr(box, 'tracking_id'): # this is true
                tr_id = box.tracking_id
                c = color_mapping[tr_id % len(color_mapping)]
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=1)

    # Limit visible range.
    # axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(eval_range[0], eval_range[2])
    ax.set_ylim(eval_range[1], eval_range[3])

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    # plt.title(sample_token)
    if eval_range[0] == -53.0:
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_xticklabels(['-40', '-20', '0', '20', '40',])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_yticklabels(['-40', '-20', '0', '20', '40'])
    else:
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.set_xticklabels(['20', '40', '60', '80', '100',])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_yticklabels(['-40', '-20', '0', '20', '40'])
    ax.set_aspect('equal')
    if savepath is not None:
        savepath = savepath + '_bev'
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    # else:
    #     plt.show()

def render_annotation(
        anntoken: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = 'render.png',
        extra_info: bool = False) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes = []
    select_cams = []
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                           selected_anntokens=[anntoken])
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)
            # We found an image that matches. Let's abort.
    # assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
    #                      'Try using e.g. BoxVisibility.ANY.'
    # assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    num_cam = len(all_bboxes)

    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    select_cams = [sample_record['data'][cam] for cam in select_cams]
    print('bbox in cams:', select_cams)
    # Plot LIDAR view.
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    CustomLidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
    for box in boxes:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
        corners = view_points(boxes[0].corners(), view, False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    # Plot CAMERA view.
    for i in range(1, num_cam + 1):
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get('sample_data', cam)['channel'])
        axes[i].axis('off')
        axes[i].set_aspect('equal')
        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Print extra information about the annotation below the camera view.
        axes[i].set_xlim(0, im.size[0])
        axes[i].set_ylim(im.size[1], 0)

    if extra_info:
        rcParams['font.family'] = 'monospace'

        w, l, h = ann_record['size']
        category = ann_record['category_name']
        lidar_points = ann_record['num_lidar_pts']
        radar_points = ann_record['num_radar_pts']

        sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))

        information = ' \n'.join(['category: {}'.format(category),
                                  '',
                                  '# lidar points: {0:>4}'.format(lidar_points),
                                  '# radar points: {0:>4}'.format(radar_points),
                                  '',
                                  'distance: {:>7.3f}m'.format(dist),
                                  '',
                                  'width:  {:>7.3f}m'.format(w),
                                  'length: {:>7.3f}m'.format(l),
                                  'height: {:>7.3f}m'.format(h)])

        plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

    if out_path is not None:
        plt.savefig(out_path)



def get_sample_data(sample_data_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    selected_anntokens=None,
                    use_flat_vehicle_coordinates: bool = False,
                    boxes = None,
                    side='vehicle-side',
                    nusc=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    # hardcode
    if side in ['vehicle-side', 'cooperative']:
        cs_record = nusc.get('calibrated_sensor', 'a4debdb5-22b2-3269-b7d7-756f1d560c92')
        sensor2ego_rot = iterative_closest_point(np.array(cs_record['rotation']))
        sensor2ego_trans = cs_record['translation']
    elif side == 'infrastructure-side':
        cs_record = nusc.get('calibrated_sensor', '23ef3a7f-ebdc-389f-a831-34b76331632a')
        sensor2ego_rot = Quaternion(cs_record['rotation']).rotation_matrix
        sensor2ego_trans = cs_record['translation']
        calib_l2c_path = data_root + side + '/calib/virtuallidar_to_camera/' + sample_data_token + '.json'
        calib_l2c = read_json(calib_l2c_path)
        l2c_rot = np.array(calib_l2c['rotation'])
        appro_l2c_rot = iterative_closest_point(np.array(l2c_rot))
        cs_record = nusc.get('calibrated_sensor', 'eda75990-71f2-387c-b06a-415a923663a9')
        
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    # hardcode
    dir_path = data_path.split('velodyne')[0]
    data_path = dir_path + 'image/' + sample_data_token + '.jpg'

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        # cam_intrinsic_path = data_root + side + '/calib/camera_intrinsic/' + sample_data_token + '.json'
        # cam_intrinsic = np.array(read_json(cam_intrinsic_path)['cam_K']).reshape(3, 3)
        
        # hardcode
        imsize = (1920, 1080)
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #     boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #     boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(sensor2ego_trans))
            box.rotate(Quaternion(matrix=np.array(sensor2ego_rot)).inverse)
            
            if side == 'infrastructure-side':
                # rotate
                box.center = np.dot(l2c_rot, box.center)
                q = Quaternion(matrix=appro_l2c_rot)
                box.orientation = q * box.orientation
                box.velocity = np.dot(l2c_rot, box.velocity)
                # translate
                box.translate(np.squeeze(np.array(calib_l2c['translation'])))
        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic



# def get_predicted_data(sample_data_token: str,
#                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
#                        selected_anntokens=None,
#                        use_flat_vehicle_coordinates: bool = False,
#                        pred_anns=None,
#                        side = 'vehicle-side',
#                        nusc=None
#                        ):
#     """
#     Returns the data path as well as all annotations related to that sample_data.
#     Note that the boxes are transformed into the current sensor's coordinate frame.
#     :param sample_data_token: Sample_data token.
#     :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
#     :param selected_anntokens: If provided only return the selected annotation.
#     :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
#                                          aligned to z-plane in the world.
#     :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
#     """

#     # Retrieve sensor & pose records
#     sd_record = nusc.get('sample_data', sample_data_token)
#     # hardcode
#     if side in ['vehicle-side', 'cooperative']:
#         cs_record = nusc.get('calibrated_sensor', 'db7afd30-5a9a-325c-996b-ce1fd8f54739')
#         sensor2ego_rot = iterative_closest_point(np.array(cs_record['rotation']))
#         sensor2ego_trans = cs_record['translation']
#     elif side == 'infrastructure-side':
#         cs_record = nusc.get('calibrated_sensor', '23ef3a7f-ebdc-389f-a831-34b76331632a')
#         sensor2ego_rot = Quaternion(cs_record['rotation']).rotation_matrix
#         sensor2ego_trans = cs_record['translation']
#         calib_l2c_path = data_root + side + '/calib/virtuallidar_to_camera/' + sample_data_token + '.json'
#         calib_l2c = read_json(calib_l2c_path)
#         l2c_rot = np.array(calib_l2c['rotation'])
#         appro_l2c_rot = iterative_closest_point(np.array(l2c_rot))
#         cs_record = nusc.get('calibrated_sensor', 'eda75990-71f2-387c-b06a-415a923663a9')
        
#     sensor_record = nusc.get('sensor', cs_record['sensor_token'])
#     pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

#     data_path = nusc.get_sample_data_path(sample_data_token)
#     # hardcode
#     dir_path = data_path.split('velodyne')[0]
#     data_path = dir_path + 'image/' + sample_data_token + '.jpg'

#     if sensor_record['modality'] == 'camera':
#         cam_intrinsic = np.array(cs_record['camera_intrinsic'])
#         # hardcode
#         imsize = (1920, 1080)
#     else:
#         cam_intrinsic = None
#         imsize = None

#     # Retrieve all sample annotations and map to sensor coordinate system.
#     # if selected_anntokens is not None:
#     #    boxes = list(map(nusc.get_box, selected_anntokens))
#     # else:
#     #    boxes = nusc.get_boxes(sample_data_token)
#     boxes = pred_anns
#     # Make list of Box objects including coord system transforms.
#     box_list = []
#     for box in boxes:
#         if use_flat_vehicle_coordinates:
#             # Move box to ego vehicle coord system parallel to world z plane.
#             yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
#             box.translate(-np.array(pose_record['translation']))
#             box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
#         else:
#             # Move box to ego vehicle coord system.
#             box.translate(-np.array(pose_record['translation']))
#             box.rotate(Quaternion(pose_record['rotation']).inverse)

#             #  Move box to sensor coord system.
#             box.translate(-np.array(sensor2ego_trans))
#             box.rotate(Quaternion(matrix=np.array(sensor2ego_rot)).inverse)

#             if side == 'infrastructure-side':
#                 # rotate
#                 box.center = np.dot(l2c_rot, box.center)
#                 q = Quaternion(matrix=appro_l2c_rot)
#                 box.orientation = q * box.orientation
#                 box.velocity = np.dot(l2c_rot, box.velocity)
#                 # translate
#                 box.translate(np.squeeze(np.array(calib_l2c['translation'])))
                
#         if sensor_record['modality'] == 'camera' and not \
#                 box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
#             continue
#         box_list.append(box)

#     return data_path, box_list, cam_intrinsic

detection_mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }


def lidar_render(sample_token, data, ax=None, out_path=None, side='vehicle-side', thre=0.0, nusc=None):
    bbox_gt_list = []
    bbox_pred_list = []
    anns = nusc.get('sample', sample_token)['anns']
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        bbox_gt_list.append(TrackingBox(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=nusc.box_velocity(content['token'])[:2],
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            tracking_name=content['category_name'],
            tracking_score=-1.0 if 'tracking_score' not in content else float(content['tracking_score']),
            tracking_id=content['instance_token']))


    bbox_anns = data['results'][sample_token]
    for content in bbox_anns:
        bbox_pred_list.append(TrackingBox(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            tracking_name=content['tracking_name'],
            tracking_score=-1.0 if 'tracking_score' not in content else float(content['tracking_score']),
            tracking_id=content['tracking_id']))
    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)
    print('green is ground truth')
    print('blue is the predited result')
    if side in ['vehicle-side', 'cooperative']:
        eval_range = [-53.0, -53.0, 53.0, 53.0]
    elif side == 'infrastructure-side':
        eval_range = [-3.0, -53.0, 103.0, 53.0]
    visualize_sample(nusc, sample_token, gt_annotations, pred_annotations, conf_th=thre, savepath=out_path, eval_range=eval_range, ax=ax)


def get_color(category_name: str):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
     'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
     'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
     'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
     'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
     'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
     'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
     'vehicle.ego']
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    #print(category_name)
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def render_sample_data(
        sample_token: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        pred_data=None,
        side='vehicle-side',
        thre=None,
        nusc=None,
      ) -> None:
    """
    Render sample data onto axis.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    sample = nusc.get('sample', sample_token)
    # sample = data['results'][sample_token_list[0]][0]
    if ax is None:
        # Create a figure
        fig = plt.figure(figsize=(24, 13))
        gs = gridspec.GridSpec(1, 2, wspace=0.05)
        gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.05)
        ax1 = fig.add_subplot(gs0[0, 0])
        ax2 = fig.add_subplot(gs0[1, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        axes = [ax1, ax2, ax3]

    # hardcode
    sample_data_token = sample['data']['LIDAR_TOP']
    
    # plot in BEV
    lidar_render(sample_token, pred_data, ax=axes[2], out_path=None, side=side, thre=thre, nusc=nusc)
    
    # plot in image
    boxes_pred = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                    name=record['detection_name'], token=record['tracking_id']) for record in
                pred_data['results'][sample_token] if record['detection_score'] > thre]
    boxes_gt = nusc.get_boxes(sample_data_token)
    data_path, boxes_pred, camera_intrinsic = get_sample_data(sample_data_token,
                                                                    box_vis_level=box_vis_level, boxes=boxes_pred, side=side, nusc=nusc)
    _, boxes_gt, _ = get_sample_data(sample_data_token, box_vis_level=box_vis_level, boxes=boxes_gt, side=side, nusc=nusc)

    data = Image.open(data_path)
    # Show image.
    axes[0].imshow(data)
    axes[1].imshow(data)

    # Show boxes.
    for box in boxes_pred:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)
    for box in boxes_gt:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[1], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)

    # Limit visible range.
    axes[0].set_xlim(0, data.size[0])
    axes[0].set_ylim(data.size[1], 0)
    axes[1].set_xlim(0, data.size[0])
    axes[1].set_ylim(data.size[1], 0)

    axes[0].axis('off')
    # axes[0].set_title('PRED: {} {labels_type}'.format(
    #     'VEHICLE_CAM_FRONT', labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    axes[0].set_aspect('equal')

    axes[1].axis('off')
    # axes[1].set_title('GT:{} {labels_type}'.format(
    #     'VEHICLE_CAM_FRONT', labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    axes[1].set_aspect('equal')

    if out_path is not None:
        out_path = os.path.join(out_path, sample_token+'.png')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.03, dpi=300)
    else:
        plt.show()
    plt.close()

def render_sample_data_coop(
        sample_token: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        pred_data=None,
        side='vehicle-side',
        thre=None,
        veh2inf=None,
        nusc=None,
        nusc_inf=None,
        is_gt=False
      ) -> None:
    """
    Render sample data onto axis.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    sample = nusc.get('sample', sample_token)
    # sample = data['results'][sample_token_list[0]][0]
    if ax is None:
        # Create a figure
        fig = plt.figure(figsize=(24, 13))
        gs = gridspec.GridSpec(1, 2, wspace=0.05)
        gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.05)
        ax1 = fig.add_subplot(gs0[0, 0])
        ax2 = fig.add_subplot(gs0[1, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        axes = [ax1, ax2, ax3]

    # hardcode
    sample_data_token = sample['data']['LIDAR_TOP']
    
    # plot in BEV
    lidar_render(sample_token, pred_data, ax=axes[2], out_path=None, side=side, thre=thre, nusc=nusc)
    
    # plot in image for veh
    if is_gt:
        # load cooperative label
        boxes_gt = nusc.get_boxes(sample_data_token)
        data_path, boxes_gt, camera_intrinsic = get_sample_data(sample_data_token, box_vis_level=box_vis_level, boxes=boxes_gt, side=side, nusc=nusc)
    else:
        boxes_pred = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                    name=record['detection_name'], token='predicted') for record in
                pred_data['results'][sample_token] if record['detection_score'] > thre]
        data_path, boxes_pred, camera_intrinsic = get_sample_data(sample_data_token,
                                                                    box_vis_level=box_vis_level, boxes=boxes_pred, side=side, nusc=nusc)
    data = Image.open(data_path)
    axes[0].imshow(data)
    # Show boxes.
    if is_gt:
        for box in boxes_gt:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)
    else:
        for box in boxes_pred:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)

    # plot in image for inf
    sample_data_token_inf = veh2inf[sample_data_token]
    if is_gt:
        # load cooperative label, so use nusc rather than nusc_inf
        boxes_gt = nusc.get_boxes(sample_data_token)
        boxes_gt = veh2inf_convert(boxes_gt, data_root, veh2inf, sample_data_token)
        data_path, boxes_gt, camera_intrinsic = get_sample_data(sample_data_token_inf, box_vis_level=box_vis_level, boxes=boxes_gt, side='infrastructure-side', nusc=nusc_inf)
    else:
        # load cooperative prediction
        boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                        name=record['detection_name'], token='predicted') for record in
                    pred_data['results'][sample_token] if record['detection_score'] > thre]
        boxes_inf = veh2inf_convert(boxes, data_root, veh2inf, sample_data_token)
        data_path, boxes_pred, camera_intrinsic = get_sample_data(sample_data_token_inf,
                                                                    box_vis_level=box_vis_level, boxes=boxes_inf, side='infrastructure-side', nusc=nusc_inf)
    data = Image.open(data_path)
    axes[1].imshow(data)
    if is_gt:
        for box in boxes_gt:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[1], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)
    else:
        for box in boxes_pred:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[1], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)
    # Limit visible range.
    axes[0].set_xlim(0, data.size[0])
    axes[0].set_ylim(data.size[1], 0)
    axes[1].set_xlim(0, data.size[0])
    axes[1].set_ylim(data.size[1], 0)

    axes[0].axis('off')
    # axes[0].set_title('PRED: {} {labels_type}'.format(
    #     'VEHICLE_CAM_FRONT', labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    axes[0].set_aspect('equal')

    axes[1].axis('off')
    # axes[1].set_title('GT:{} {labels_type}'.format(
    #     'VEHICLE_CAM_FRONT', labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    axes[1].set_aspect('equal')

    if out_path is not None:
        out_path = os.path.join(out_path, sample_token+'.png')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.03, dpi=300)
    else:
        plt.show()
    plt.close()
    
def to_video(folder_path, out_path, fps=4, downsample=1):
    imgs_path = glob.glob(os.path.join(folder_path, '*.png'))
    imgs_path = sorted(imgs_path)
    img_array = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img = cv2.resize(img, (width//downsample, height //
                            downsample), interpolation=cv2.INTER_AREA)
        height, width, channel = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



# ============================================================================
# Metric-aware visualization extension.
# This block integrates the metric-driven simulation / filtering ideas from
# the second script into the original nuScenes-json visualization pipeline.
# ============================================================================
import math
import random

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
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
    'trafficcone': 'traffic_cone',
    'construction_vehicle': 'construction_vehicle',
}


def normalize_detection_name(name):
    """Normalize nuScenes full category names to 10-class detection names."""
    name = str(name)
    if name in CLASS_NAMES:
        return name
    if name in CLASS_NAME_ALIASES:
        return CLASS_NAME_ALIASES[name]
    try:
        mapped = category_to_detection_name(name)
        if mapped is not None:
            return mapped
    except Exception:
        pass
    name_low = name.lower().replace('-', '_')
    if 'traffic' in name_low and 'cone' in name_low:
        return 'traffic_cone'
    if 'pedestrian' in name_low:
        return 'pedestrian'
    for cls in CLASS_NAMES:
        if cls in name_low:
            return cls
    return 'car'


def detection_metric_preset(name):
    """Metric-like presets copied from the second visualization script."""
    name = str(name).lower()
    if name == 'old':
        # Larger translation/orientation error, higher FN/FP.
        return dict(mate=0.391, mase=0.240, maoe=0.652, mave=0.247, fn_rate=0.20, fp_rate=0.15)
    if name == 'new':
        # Better detection quality, lower miss/false-positive rate.
        return dict(mate=0.276, mase=0.132, maoe=0.153, mave=0.212, fn_rate=0.10, fp_rate=0.08)
    return dict(mate=0.0, mase=0.0, maoe=0.0, mave=0.0, fn_rate=0.0, fp_rate=0.0)


def _score_of_record(record):
    if 'detection_score' in record:
        return float(record['detection_score'])
    if 'tracking_score' in record:
        return float(record['tracking_score'])
    return 1.0


def _class_of_record(record):
    return normalize_detection_name(record.get('detection_name', record.get('tracking_name', 'car')))


def class_aware_circle_nms_records(records, radius=1.5, score_key='detection_score'):
    """Simple class-aware circle NMS over json-format prediction records."""
    if not records:
        return []
    kept = []
    classes = sorted(set(_class_of_record(r) for r in records))
    for cls in classes:
        cls_records = [r for r in records if _class_of_record(r) == cls]
        cls_records = sorted(cls_records, key=_score_of_record, reverse=True)
        centers = []
        for r in cls_records:
            c = np.asarray(r['translation'][:2], dtype=np.float32)
            if not centers:
                kept.append(r)
                centers.append(c)
                continue
            d = np.linalg.norm(np.stack(centers, axis=0) - c[None, :], axis=1)
            if np.all(d > radius):
                kept.append(r)
                centers.append(c)
    kept = sorted(kept, key=_score_of_record, reverse=True)
    return kept


def random_drop_records(records, drop_prob=0.15, seed=0, keep_top1_per_class=False):
    """Randomly drop predictions to mimic missed detections in qualitative figures."""
    if not records:
        return []
    rng = np.random.default_rng(seed)
    must_keep_ids = set()
    if keep_top1_per_class:
        for cls in sorted(set(_class_of_record(r) for r in records)):
            idxs = [i for i, r in enumerate(records) if _class_of_record(r) == cls]
            if idxs:
                best = max(idxs, key=lambda i: _score_of_record(records[i]))
                must_keep_ids.add(best)
    out = []
    for i, r in enumerate(records):
        if i in must_keep_ids or rng.random() >= drop_prob:
            out.append(r)
    return out


def post_process_prediction_records(records, args, sample_index=0):
    """Apply score threshold, optional random drop, optional circle NMS and top-k."""
    out = [r for r in records if _score_of_record(r) >= float(args.thre)]
    if getattr(args, 'random_drop_pred', False):
        out = random_drop_records(
            out,
            drop_prob=float(args.drop_prob),
            seed=int(args.seed) + int(sample_index),
            keep_top1_per_class=bool(args.keep_top1_per_class),
        )
    if getattr(args, 'apply_circle_nms', False):
        out = class_aware_circle_nms_records(out, radius=float(args.circle_nms_radius))
    max_per_frame = int(getattr(args, 'post_max_size', 300))
    if len(out) > max_per_frame:
        out = sorted(out, key=_score_of_record, reverse=True)[:max_per_frame]
    return out


def get_metric_values_from_args(args):
    preset = detection_metric_preset(args.det_preset)
    mate = args.sim_mate if args.sim_mate is not None else preset['mate']
    mase = args.sim_mase if args.sim_mase is not None else preset['mase']
    maoe = args.sim_maoe if args.sim_maoe is not None else preset['maoe']
    mave = args.sim_mave if args.sim_mave is not None else preset['mave']
    fn_rate = args.sim_fn_rate if args.sim_fn_rate is not None else preset['fn_rate']
    fp_rate = args.sim_fp_rate if args.sim_fp_rate is not None else preset['fp_rate']
    return dict(mate=float(mate), mase=float(mase), maoe=float(maoe), mave=float(mave),
                fn_rate=float(fn_rate), fp_rate=float(fp_rate))


def simulate_prediction_records_from_gt(nusc, sample_token, args, sample_index=0, side='vehicle-side'):
    """Simulate json-format predictions from nuScenes GT according to metric-like errors.

    Output schema is compatible with the original first script:
    translation, size, rotation, velocity, detection_name, detection_score,
    tracking_name, tracking_score, tracking_id, sample_token.
    """
    metrics = get_metric_values_from_args(args)
    rng = np.random.default_rng(int(args.seed) + int(sample_index))

    sample = nusc.get('sample', sample_token)
    records = []

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        det_name = normalize_detection_name(ann.get('category_name', 'car'))
        if det_name not in CLASS_NAMES:
            continue
        if rng.random() < metrics['fn_rate']:
            continue

        translation = np.asarray(ann['translation'], dtype=np.float32).copy()
        size = np.asarray(ann['size'], dtype=np.float32).copy()
        q = Quaternion(ann['rotation'])

        # mATE-like translation error.
        if metrics['mate'] > 0:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            radius = abs(rng.normal(loc=metrics['mate'], scale=max(metrics['mate'] * 0.35, 1e-6)))
            translation[0] += radius * np.cos(angle)
            translation[1] += radius * np.sin(angle)

        # mASE-like scale error.
        if metrics['mase'] > 0:
            scale = 1.0 + rng.normal(0.0, metrics['mase'], size=3)
            size *= np.clip(scale, 0.45, 2.2)

        # mAOE-like yaw error.
        if metrics['maoe'] > 0:
            yaw_noise = rng.normal(0.0, metrics['maoe'])
            q = Quaternion(axis=[0, 0, 1], angle=float(yaw_noise)) * q

        # mAVE-like velocity error.
        try:
            velocity = np.asarray(nusc.box_velocity(ann_token)[:2], dtype=np.float32)
        except Exception:
            velocity = np.zeros((2,), dtype=np.float32)
        if metrics['mave'] > 0:
            velocity = velocity + rng.normal(0.0, metrics['mave'], size=2)

        score = float(np.clip(args.sim_score - rng.uniform(0.0, 0.18), 0.01, 0.99))
        track_id = ann.get('instance_token', ann_token)
        records.append({
            'sample_token': sample_token,
            'translation': translation.tolist(),
            'size': size.tolist(),
            'rotation': list(q.elements),
            'velocity': velocity.tolist(),
            'detection_name': det_name,
            'detection_score': score,
            'tracking_name': det_name,
            'tracking_score': score,
            'tracking_id': track_id,
        })

    # mAP-like false positives.
    n_fp = int(round(max(1, len(records)) * metrics['fp_rate']))
    if side == 'infrastructure-side':
        xmin, ymin, xmax, ymax = [-3.0, -53.0, 103.0, 53.0]
    else:
        xmin, ymin, xmax, ymax = [-53.0, -53.0, 53.0, 53.0]
    for k in range(n_fp):
        det_name = str(rng.choice(CLASS_NAMES))
        translation = [float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)), float(rng.uniform(-1.0, 2.0))]
        size = [float(rng.uniform(1.0, 5.0)), float(rng.uniform(0.8, 2.5)), float(rng.uniform(1.2, 2.2))]
        q = Quaternion(axis=[0, 0, 1], angle=float(rng.uniform(-np.pi, np.pi)))
        score = float(np.clip(args.sim_score - rng.uniform(0.20, 0.45), 0.01, 0.75))
        records.append({
            'sample_token': sample_token,
            'translation': translation,
            'size': size,
            'rotation': list(q.elements),
            'velocity': [0.0, 0.0],
            'detection_name': det_name,
            'detection_score': score,
            'tracking_name': det_name,
            'tracking_score': score,
            'tracking_id': f'fp_{sample_index}_{k}',
        })

    records = post_process_prediction_records(records, args, sample_index=sample_index)
    return records


def build_metric_aware_pred_data(nusc, sample_token_list, args, side='vehicle-side', real_pred_data=None):
    """Return pred_data in the same format as the original json result.

    If --simulate-from-gt is set, predictions are generated from GT and metric presets.
    Otherwise, real predictions are optionally processed by NMS/random-drop/top-k.
    """
    out = {'results': {}}
    for i, token in enumerate(sample_token_list):
        if getattr(args, 'simulate_from_gt', False):
            out['results'][token] = simulate_prediction_records_from_gt(nusc, token, args, sample_index=i, side=side)
        else:
            records = copy.deepcopy(real_pred_data['results'].get(token, [])) if real_pred_data else []
            out['results'][token] = post_process_prediction_records(records, args, sample_index=i)
    return out


def save_metric_sim_json(pred_data, out_path):
    """Save simulated or post-processed prediction json with mmcv if requested."""
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        mmcv.dump(pred_data, out_path)
        print(f'[MetricVis] Saved metric-aware prediction json: {out_path}')


def parse_bool_flag(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in ['1', 'true', 'yes', 'y']




# ============================================================================
# Direct SPD infos-pkl visualization path.
# This path keeps the original script structure but bypasses NuScenes API and
# transform_box_veh2inf when --infos-pkl is provided. It reads gt_boxes and
# lidar_path directly from spd_infos_temporal_val.pkl, renders point cloud,
# GT boxes and metric-simulated prediction boxes in BEV.
# ============================================================================
INFO_CLASS_NAMES = CLASS_NAMES

INFO_CLASS_COLORS = {
    'car': '#1f77b4',
    'truck': '#ff7f0e',
    'construction_vehicle': '#9467bd',
    'bus': '#2ca02c',
    'trailer': '#d62728',
    'barrier': '#7f7f7f',
    'motorcycle': '#e377c2',
    'bicycle': '#17becf',
    'pedestrian': '#bcbd22',
    'traffic_cone': '#8c564b',
}
INFO_GT_COLOR = '#00cc44'


def load_spd_infos_pkl(infos_pkl):
    with open(infos_pkl, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'infos' in obj:
        return obj['infos']
    if isinstance(obj, list):
        return obj
    raise TypeError(f'Unsupported infos pkl format: {type(obj)}')



def _lzf_decompress(data, expected_length):
    """Minimal LZF decompressor for PCD binary_compressed blocks."""
    i = 0
    o = 0
    out = bytearray(expected_length if expected_length is not None else 0)
    data_len = len(data)
    while i < data_len:
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            length = ctrl + 1
            if expected_length is not None and o + length > expected_length:
                raise ValueError("LZF literal overrun.")
            if expected_length is None:
                out.extend(data[i:i + length])
            else:
                out[o:o + length] = data[i:i + length]
            o += length
            i += length
        else:
            length = ctrl >> 5
            ref = o - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                length += data[i]
                i += 1
            ref -= data[i]
            i += 1
            length += 2
            if ref < 0:
                raise ValueError("Invalid LZF back-reference.")
            if expected_length is not None and o + length > expected_length:
                raise ValueError("LZF back-reference overrun.")
            for _ in range(length):
                if expected_length is None:
                    out.append(out[ref])
                else:
                    out[o] = out[ref]
                o += 1
                ref += 1
    if expected_length is not None and o != expected_length:
        raise ValueError(f"LZF decompressed size mismatch: got {o}, expected {expected_length}")
    return bytes(out if expected_length is None else out[:o])


def _load_points_xyz_from_pcd(pcd_path):
    """Load x/y/z from PCD files. Supports ascii, binary and binary_compressed."""
    with open(pcd_path, 'rb') as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PCD: missing DATA header.")
            line_decoded = line.decode('ascii', errors='ignore').strip()
            if not line_decoded or line_decoded.startswith('#'):
                continue
            parts = line_decoded.split()
            key = parts[0].upper()
            value = parts[1:]
            header[key] = value
            if key == 'DATA':
                break
        data_blob = f.read()

    fields = header.get('FIELDS', [])
    sizes = list(map(int, header.get('SIZE', [])))
    types = header.get('TYPE', [])
    counts = list(map(int, header.get('COUNT', ['1'] * len(fields))))
    points = int(header.get('POINTS', [header.get('WIDTH', ['0'])[0]])[0])
    data_type = header.get('DATA', ['binary'])[0].lower()

    if not fields or 'x' not in fields or 'y' not in fields or 'z' not in fields:
        raise ValueError(f"Invalid PCD header, missing xyz fields: {pcd_path}")

    xyz_idx = [fields.index('x'), fields.index('y'), fields.index('z')]
    bytes_per_field = [s * c for s, c in zip(sizes, counts)]

    def _dtype_of(size, typ):
        if typ == 'F' and size == 4:
            return np.float32
        if typ == 'F' and size == 8:
            return np.float64
        if typ == 'U' and size == 1:
            return np.uint8
        if typ == 'U' and size == 2:
            return np.uint16
        if typ == 'U' and size == 4:
            return np.uint32
        if typ == 'I' and size == 1:
            return np.int8
        if typ == 'I' and size == 2:
            return np.int16
        if typ == 'I' and size == 4:
            return np.int32
        raise ValueError(f"Unsupported PCD type/size: {typ}{size}")

    if data_type == 'ascii':
        text = data_blob.decode('ascii', errors='ignore').strip()
        if not text:
            return np.zeros((0, 3), dtype=np.float32)
        data = np.loadtxt(text.splitlines(), dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        return data[:, xyz_idx].astype(np.float32)

    if data_type == 'binary':
        point_step = sum(bytes_per_field)
        expected = points * point_step
        raw = data_blob[:expected]
        if len(raw) < expected:
            raise ValueError("Truncated PCD binary payload.")

        offsets = []
        cur = 0
        for field_len in bytes_per_field:
            offsets.append(cur)
            cur += field_len

        dtype_desc = []
        for name, size, typ, cnt, off in zip(fields, sizes, types, counts, offsets):
            dt = _dtype_of(size, typ)
            shape = (cnt,) if cnt > 1 else ()
            dtype_desc.append((name, dt, shape, off))

        structured_dtype = np.dtype({
            'names': [x[0] for x in dtype_desc],
            'formats': [np.dtype((x[1], x[2])) for x in dtype_desc],
            'offsets': [x[3] for x in dtype_desc],
            'itemsize': point_step,
        })
        arr = np.frombuffer(raw, dtype=structured_dtype, count=points)
        return np.stack([
            arr['x'].reshape(points, -1)[:, 0],
            arr['y'].reshape(points, -1)[:, 0],
            arr['z'].reshape(points, -1)[:, 0],
        ], axis=1).astype(np.float32)

    if data_type == 'binary_compressed':
        if len(data_blob) < 8:
            raise ValueError("Invalid compressed PCD payload.")
        compressed_size = struct.unpack('<I', data_blob[:4])[0]
        uncompressed_size = struct.unpack('<I', data_blob[4:8])[0]
        compressed = data_blob[8:8 + compressed_size]
        raw = _lzf_decompress(compressed, uncompressed_size)

        xyz = {}
        offset = 0
        for i, (field, size, typ, cnt) in enumerate(zip(fields, sizes, types, counts)):
            dt = _dtype_of(size, typ)
            field_bytes = points * size * cnt
            block = raw[offset:offset + field_bytes]
            if i in xyz_idx:
                arr = np.frombuffer(block, dtype=dt, count=points * cnt)
                arr = arr.reshape(points, cnt)[:, 0]
                xyz[field] = arr.astype(np.float32)
            offset += field_bytes
        return np.stack([xyz['x'], xyz['y'], xyz['z']], axis=1).astype(np.float32)

    raise ValueError(f"Unsupported PCD DATA type: {data_type}")


def _with_bin_pcd_candidates(path):
    """Return path variants so velodyne/000870.bin can fall back to velodyne/000870.pcd."""
    path = os.path.normpath(str(path))
    candidates = [path]
    root, ext = os.path.splitext(path)
    if ext.lower() == '.bin':
        candidates.append(root + '.pcd')
    elif ext.lower() == '.pcd':
        candidates.append(root + '.bin')
    return candidates


def resolve_spd_lidar_path(info, dataroot, side='vehicle-side'):
    """Resolve lidar path from SPD infos.

    The pkl often stores lidar_path as velodyne/000870.bin, while the real SPD
    files may be stored as vehicle-side/velodyne/000870.pcd. This resolver tries
    both .bin and .pcd and several possible roots.
    """
    lidar_path = info.get('lidar_path', None) or info.get('pts_filename', None)
    if lidar_path is None:
        return None

    dataroot = os.path.normpath(str(dataroot))
    side = str(side)

    root_candidates = [
        '',
        dataroot,
        os.path.join(dataroot, side),
        os.path.join(dataroot, 'vehicle-side'),
        os.path.join(dataroot, 'infrastructure-side'),
        os.path.join(dataroot, 'cooperative'),
    ]

    # If dataroot already ends with vehicle-side/infrastructure-side, also try its parent.
    base = os.path.basename(dataroot)
    if base in ['vehicle-side', 'infrastructure-side', 'cooperative']:
        parent = os.path.dirname(dataroot)
        root_candidates.extend([
            parent,
            os.path.join(parent, side),
            os.path.join(parent, 'vehicle-side'),
            os.path.join(parent, 'infrastructure-side'),
            os.path.join(parent, 'cooperative'),
        ])

    candidates = []
    for p in _with_bin_pcd_candidates(lidar_path):
        for root in root_candidates:
            c = os.path.normpath(os.path.join(root, p)) if root else os.path.normpath(p)
            candidates.append(c)

    # De-duplicate while preserving order.
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            unique_candidates.append(c)
            seen.add(c)

    for c in unique_candidates:
        if os.path.isfile(c):
            return c

    # Return the most likely .pcd path for error display.
    for c in unique_candidates:
        if c.endswith('.pcd'):
            return c
    return unique_candidates[0] if unique_candidates else None


def load_lidar_points_xyzi(lidar_file, load_dim=4):
    """Load lidar points for visualization.

    Supports:
    - .bin with 4 or 5 float columns
    - .pcd ascii / binary / binary_compressed
    If a .bin file is missing, automatically tries the matching .pcd path.
    """
    if lidar_file is None:
        return None

    candidates = _with_bin_pcd_candidates(lidar_file)

    for p in candidates:
        if not os.path.isfile(p):
            continue
        if p.endswith('.bin'):
            arr = np.fromfile(p, dtype=np.float32)
            if arr.size == 0:
                return None
            if arr.size % 5 == 0:
                pts = arr.reshape(-1, 5)[:, :4]
            elif arr.size % 4 == 0:
                pts = arr.reshape(-1, 4)
            elif arr.size % load_dim == 0:
                pts = arr.reshape(-1, load_dim)[:, :min(load_dim, 4)]
            else:
                raise ValueError(f'Cannot infer point dimension for {p}, float count={arr.size}')
            if pts.shape[1] == 3:
                pts = np.concatenate([pts, np.zeros((len(pts), 1), dtype=pts.dtype)], axis=1)
            return pts
        if p.endswith('.pcd'):
            pts_xyz = _load_points_xyz_from_pcd(p)
            if pts_xyz is None or len(pts_xyz) == 0:
                return None
            intensity = np.zeros((len(pts_xyz), 1), dtype=pts_xyz.dtype)
            return np.concatenate([pts_xyz[:, :3], intensity], axis=1)

    print(f'[WARN] Lidar file not found. Tried: {candidates}')
    return None


def infos_box_corners_xy(box):
    x, y, z, dx, dy, dz, yaw = box[:7]
    hx, hy = abs(float(dx)) / 2.0, abs(float(dy)) / 2.0
    corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=np.float32)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y], dtype=np.float32)


def infos_in_range(box, pc_range):
    x, y = float(box[0]), float(box[1])
    xmin, ymin, xmax, ymax = pc_range
    return xmin <= x <= xmax and ymin <= y <= ymax


def draw_infos_bev_boxes(ax, boxes, names=None, scores=None, is_gt=False, linewidth=1.0, score_text=False, alpha=None):
    if boxes is None or len(boxes) == 0:
        return
    for i, box in enumerate(boxes):
        if len(box) < 7:
            continue
        corners = infos_box_corners_xy(box)
        poly = np.vstack([corners, corners[0]])
        if is_gt:
            color, linestyle, base_alpha = INFO_GT_COLOR, '--', 0.95
        else:
            name = normalize_detection_name(names[i]) if names is not None and i < len(names) else 'car'
            color, linestyle, base_alpha = INFO_CLASS_COLORS.get(name, '#1f77b4'), '-', 0.95
        use_alpha = base_alpha if alpha is None else alpha
        ax.plot(poly[:, 0], poly[:, 1], color=color, linestyle=linestyle, linewidth=linewidth, alpha=use_alpha)
        center = np.asarray(box[:2], dtype=np.float32)
        front = (corners[0] + corners[1]) / 2.0
        ax.plot([center[0], front[0]], [center[1], front[1]], color=color, linestyle=linestyle, linewidth=linewidth, alpha=use_alpha)
        if score_text and scores is not None and i < len(scores):
            ax.text(float(box[0]), float(box[1]), f'{float(scores[i]):.2f}', color=color, fontsize=6)


def infos_class_aware_circle_nms(boxes, scores, names, radius=1.5):
    if boxes is None or len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    names = np.asarray([normalize_detection_name(n) for n in names])
    keep_all = []
    for cls in sorted(set(names.tolist())):
        idx = np.where(names == cls)[0]
        idx = idx[np.argsort(-scores[idx])]
        kept, centers = [], []
        for ii in idx:
            c = boxes[ii, :2]
            if len(centers) == 0:
                kept.append(ii)
                centers.append(c)
                continue
            d = np.linalg.norm(np.stack(centers, axis=0) - c[None, :], axis=1)
            if np.all(d > radius):
                kept.append(ii)
                centers.append(c)
        keep_all.extend(kept)
    keep_all = np.asarray(keep_all, dtype=np.int64)
    if len(keep_all) == 0:
        return keep_all
    return keep_all[np.argsort(-scores[keep_all])]


def simulate_infos_predictions_from_gt(info, args, frame_idx=0, return_candidates=False):
    """Simulate predictions from GT.

    Important:
    - Sparse4D-v3 style 900 candidates are copied ONLY around GT-derived object boxes.
    - False positives/background queries are added separately and are NOT used as copy centers.
    - Final visualization applies score filtering + class-aware circle NMS.
    """
    gt_boxes = np.asarray(info.get('gt_boxes', np.zeros((0, 7))), dtype=np.float32)
    gt_names = np.asarray(info.get('gt_names', ['car'] * len(gt_boxes)))
    metrics = get_metric_values_from_args(args)
    rng = np.random.default_rng(int(args.seed) + int(frame_idx))

    # 1) Generate object-centered predictions from GT.
    # These are the only boxes used as copy centers for Sparse4D 900-query simulation.
    obj_boxes, obj_names, obj_scores = [], [], []

    for i, box in enumerate(gt_boxes):
        if not infos_in_range(box, args.pc_range):
            continue
        if rng.random() < metrics['fn_rate']:
            continue

        b = box.copy()
        name = normalize_detection_name(gt_names[i])

        # mATE-like translation perturbation around GT center.
        if metrics['mate'] > 0:
            theta = rng.uniform(0, 2 * np.pi)
            radius = abs(rng.normal(metrics['mate'], max(metrics['mate'] * 0.35, 1e-6)))
            b[0] += radius * np.cos(theta)
            b[1] += radius * np.sin(theta)

        # mASE-like size perturbation.
        if metrics['mase'] > 0:
            b[3:6] *= np.clip(1.0 + rng.normal(0.0, metrics['mase'], size=3), 0.45, 2.2)

        # mAOE-like yaw perturbation.
        if metrics['maoe'] > 0:
            b[6] += rng.normal(0.0, metrics['maoe'])

        score = float(np.clip(args.sim_score - rng.uniform(0.0, 0.18), 0.01, 0.99))

        obj_boxes.append(b)
        obj_names.append(name)
        obj_scores.append(score)

    if len(obj_boxes) == 0:
        empty = (np.zeros((0, 7), dtype=np.float32), np.asarray([]), np.asarray([]))
        return (*empty, *empty) if return_candidates else empty

    obj_boxes = np.stack(obj_boxes, axis=0).astype(np.float32)
    obj_names = np.asarray(obj_names)
    obj_scores = np.asarray(obj_scores, dtype=np.float32)

    # 2) Sparse4D 900 query simulation:
    # Copy only around object-centered boxes. Do NOT copy false positives.
    if getattr(args, 'simulate_sparse4d_900', False):
        cand_boxes, cand_names, cand_scores = sparse4d_expand_to_900_candidates(
            obj_boxes, obj_names, obj_scores, args, frame_idx=frame_idx
        )

        # Add separate FP/background sparse queries according to metric-like FP rate.
        # These are not copied; they are randomly sampled background queries.
        n_fp = int(round(max(1, len(obj_boxes)) * metrics['fp_rate']))
        if n_fp > 0:
            xmin, ymin, xmax, ymax = args.pc_range
            bg_boxes, bg_names, bg_scores = [], [], []
            for _ in range(n_fp):
                name = str(rng.choice(INFO_CLASS_NAMES))
                b = np.array([
                    rng.uniform(xmin, xmax),
                    rng.uniform(ymin, ymax),
                    rng.uniform(-1.0, 2.0),
                    rng.uniform(1.0, 5.0),
                    rng.uniform(0.8, 2.5),
                    rng.uniform(1.2, 2.2),
                    rng.uniform(-np.pi, np.pi),
                ], dtype=np.float32)
                score = float(np.clip(args.sim_score - rng.uniform(0.20, 0.45), 0.01, 0.75))
                bg_boxes.append(b)
                bg_names.append(name)
                bg_scores.append(score)

            if len(bg_boxes) > 0:
                cand_boxes = np.concatenate([cand_boxes, np.stack(bg_boxes, axis=0)], axis=0)
                cand_names = np.concatenate([cand_names, np.asarray(bg_names)], axis=0)
                cand_scores = np.concatenate([cand_scores, np.asarray(bg_scores, dtype=np.float32)], axis=0)

        final_boxes, final_names, final_scores = filter_and_nms_infos_predictions(
            cand_boxes, cand_names, cand_scores, args
        )

        if return_candidates:
            return cand_boxes, cand_names, cand_scores, final_boxes, final_names, final_scores
        return final_boxes, final_names, final_scores

    # 3) Non-Sparse4D path:
    # Add false positives normally, then filter + NMS.
    pred_boxes = [b for b in obj_boxes]
    pred_names = [n for n in obj_names]
    pred_scores = [s for s in obj_scores]

    n_fp = int(round(max(1, len(pred_boxes)) * metrics['fp_rate']))
    xmin, ymin, xmax, ymax = args.pc_range
    for _ in range(n_fp):
        name = str(rng.choice(INFO_CLASS_NAMES))
        b = np.array([
            rng.uniform(xmin, xmax),
            rng.uniform(ymin, ymax),
            rng.uniform(-1.0, 2.0),
            rng.uniform(1.0, 5.0),
            rng.uniform(0.8, 2.5),
            rng.uniform(1.2, 2.2),
            rng.uniform(-np.pi, np.pi),
        ], dtype=np.float32)
        score = float(np.clip(args.sim_score - rng.uniform(0.20, 0.45), 0.01, 0.75))
        pred_boxes.append(b)
        pred_names.append(name)
        pred_scores.append(score)

    pred_boxes = np.stack(pred_boxes, axis=0).astype(np.float32)
    pred_names = np.asarray(pred_names)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)

    final_boxes, final_names, final_scores = filter_and_nms_infos_predictions(
        pred_boxes, pred_names, pred_scores, args
    )

    if return_candidates:
        return pred_boxes, pred_names, pred_scores, final_boxes, final_names, final_scores
    return final_boxes, final_names, final_scores

def render_infos_sample_bev(info, pred_boxes, pred_names, pred_scores, args, out_path, cand_boxes=None, cand_names=None, cand_scores=None):
    gt_boxes = np.asarray(info.get('gt_boxes', np.zeros((0, 7))), dtype=np.float32)
    gt_names = np.asarray(info.get('gt_names', ['car'] * len(gt_boxes)))
    gt_keep = np.asarray([infos_in_range(b, args.pc_range) for b in gt_boxes], dtype=bool) if len(gt_boxes) else np.zeros((0,), dtype=bool)
    gt_boxes, gt_names = gt_boxes[gt_keep], gt_names[gt_keep]

    lidar_file = resolve_spd_lidar_path(info, args.dataroot, args.side)
    pts = load_lidar_points_xyzi(lidar_file)
    xmin, ymin, xmax, ymax = args.pc_range

    fig, ax = plt.subplots(figsize=(9.5, 8), dpi=180)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.20, linewidth=0.6)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    title = f"SPD BEV: {info.get('token', 'unknown')}"
    # if getattr(args, 'simulate_sparse4d_900', False):
    #     title += f" | candidates={len(cand_boxes) if cand_boxes is not None else 0}, after NMS={len(pred_boxes)}"
    ax.set_title(title)

    # Point cloud visualization.
    if pts is not None and len(pts) > 0:
        mask = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
        pts_xy = pts[mask]
        if len(pts_xy) > 0:
            dist = np.sqrt(pts_xy[:, 0] ** 2 + pts_xy[:, 1] ** 2)
            ax.scatter(
                pts_xy[:, 0], pts_xy[:, 1],
                c=dist, cmap='gray_r', s=float(args.point_size), alpha=float(args.point_alpha),
                linewidths=0, rasterized=True,
            )
    else:
        ax.text(0.01, 0.99, f'Point cloud not found: {lidar_file}', transform=ax.transAxes,
                va='top', ha='left', fontsize=7, color='red')

    ax.plot(0, 0, 'x', color='black', markersize=7, mew=1.2)

    # Draw Sparse4D 900 raw candidates before NMS with very light lines.
    if getattr(args, 'show_anchor_before_nms', False) and cand_boxes is not None and len(cand_boxes) > 0:
        cand_keep = cand_scores >= float(getattr(args, 'anchor_vis_thr', 0.05))
        cand_vis_boxes = cand_boxes[cand_keep]
        cand_vis_names = cand_names[cand_keep]
        # Limit visualization density for readability.
        max_anchor_vis = int(getattr(args, 'max_anchor_vis', 250))
        if len(cand_vis_boxes) > max_anchor_vis:
            order = np.argsort(-cand_scores[cand_keep])[:max_anchor_vis]
            cand_vis_boxes = cand_vis_boxes[order]
            cand_vis_names = cand_vis_names[order]
        draw_infos_bev_boxes(ax, cand_vis_boxes, cand_vis_names, is_gt=False, linewidth=0.35, alpha=0.18)

    draw_infos_bev_boxes(ax, gt_boxes, gt_names, is_gt=True, linewidth=1.0)
    draw_infos_bev_boxes(ax, pred_boxes, pred_names, pred_scores, is_gt=False, linewidth=1.05)

    handles = [
        Line2D([0], [0], color=INFO_GT_COLOR, lw=1.8, linestyle='--', label='GT'),
        Line2D([0], [0], color='#1f77b4', lw=1.8, linestyle='-', label='Pred after NMS'),
    ]
    # if getattr(args, 'show_anchor_before_nms', False):
    #     handles.append(Line2D([0], [0], color='#999999', lw=1.0, linestyle='-', alpha=0.35, label='Sparse queries before NMS'))
    ax.legend(handles=handles, fontsize=9, loc='upper right', framealpha=0.88)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    return pts

def to_video_safe(folder_path, out_path, fps=4, downsample=1):
    imgs_path = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    if len(imgs_path) == 0:
        print(f'[InfosVis] No png found in {folder_path}, skip video.')
        return
    img_array = []
    size = None
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // downsample, h // downsample), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        size = (w, h)
        img_array.append(img)
    if not img_array or size is None:
        return
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for img in img_array:
        out.write(img)
    out.release()



def sparse4d_expand_to_900_candidates(base_boxes, base_names, base_scores, args, frame_idx=0):
    """Expand GT-derived predictions to Sparse4D-v3-like 900 sparse query candidates.

    This is only for qualitative visualization:
    - Each GT/simulated object is copied into multiple nearby candidate queries.
    - Extra background candidates are added to reach num_sparse4d_candidates.
    - Later score filtering + class-aware circle NMS removes many candidates.
    """
    num_candidates = int(getattr(args, 'num_sparse4d_candidates', 900))
    if base_boxes is None or len(base_boxes) == 0:
        return base_boxes, base_names, base_scores

    rng = np.random.default_rng(int(args.seed) + int(frame_idx) + 2027)
    base_boxes = np.asarray(base_boxes, dtype=np.float32)
    base_names = np.asarray(base_names)
    base_scores = np.asarray(base_scores, dtype=np.float32)

    cand_boxes = []
    cand_names = []
    cand_scores = []

    n_obj = len(base_boxes)
    # More hypotheses around each object; leave some room for background queries.
    obj_candidate_budget = max(n_obj, int(num_candidates * float(getattr(args, 'sparse4d_object_ratio', 0.72))))
    per_obj = max(1, int(np.ceil(obj_candidate_budget / max(n_obj, 1))))

    # Candidate noise controls.
    center_std = float(getattr(args, 'sparse4d_center_std', 0.85))
    size_std = float(getattr(args, 'sparse4d_size_std', 0.12))
    yaw_std = float(getattr(args, 'sparse4d_yaw_std', 0.22))
    score_std = float(getattr(args, 'sparse4d_score_std', 0.12))

    for i, box in enumerate(base_boxes):
        for j in range(per_obj):
            b = box.copy()
            # Lower-rank candidates are noisier and lower-scored, mimicking multiple sparse queries around one target.
            rank_scale = 1.0 + 0.25 * (j / max(per_obj - 1, 1))
            b[0] += rng.normal(0.0, center_std * rank_scale)
            b[1] += rng.normal(0.0, center_std * rank_scale)
            b[2] += rng.normal(0.0, 0.15 * rank_scale)
            b[3:6] *= np.clip(1.0 + rng.normal(0.0, size_std * rank_scale, size=3), 0.55, 1.80)
            b[6] += rng.normal(0.0, yaw_std * rank_scale)

            # Make a few candidates high-score, many candidates mid/low-score.
            if j == 0:
                score = base_scores[i] + rng.normal(0.04, score_std * 0.25)
            else:
                score = base_scores[i] - 0.035 * j + rng.normal(0.0, score_std)
            score = float(np.clip(score, 0.01, 0.99))
            cand_boxes.append(b)
            cand_names.append(base_names[i])
            cand_scores.append(score)

            if len(cand_boxes) >= num_candidates:
                break
        if len(cand_boxes) >= num_candidates:
            break

    # Add background / false sparse queries to reach 900.
    xmin, ymin, xmax, ymax = args.pc_range
    while len(cand_boxes) < num_candidates:
        name = str(rng.choice(INFO_CLASS_NAMES))
        b = np.array([
            rng.uniform(xmin, xmax),
            rng.uniform(ymin, ymax),
            rng.uniform(-1.0, 2.0),
            rng.uniform(1.0, 5.0),
            rng.uniform(0.8, 2.5),
            rng.uniform(1.2, 2.2),
            rng.uniform(-np.pi, np.pi),
        ], dtype=np.float32)
        # Most background queries should be low-score, but some survive threshold.
        score = float(np.clip(rng.beta(1.2, 6.0) * float(args.sim_score), 0.01, 0.75))
        cand_boxes.append(b)
        cand_names.append(name)
        cand_scores.append(score)

    return np.asarray(cand_boxes, dtype=np.float32), np.asarray(cand_names), np.asarray(cand_scores, dtype=np.float32)


def filter_and_nms_infos_predictions(pred_boxes, pred_names, pred_scores, args):
    """Score filter + class-aware NMS + top-k."""
    if pred_boxes is None or len(pred_boxes) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.asarray([]), np.asarray([])

    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    pred_names = np.asarray(pred_names)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)

    keep = pred_scores >= float(args.thre)
    keep &= np.asarray([infos_in_range(b, args.pc_range) for b in pred_boxes], dtype=bool)
    pred_boxes, pred_names, pred_scores = pred_boxes[keep], pred_names[keep], pred_scores[keep]

    if getattr(args, 'apply_circle_nms', False) and len(pred_boxes) > 0:
        keep_idx = infos_class_aware_circle_nms(pred_boxes, pred_scores, pred_names, radius=float(args.circle_nms_radius))
        pred_boxes, pred_names, pred_scores = pred_boxes[keep_idx], pred_names[keep_idx], pred_scores[keep_idx]

    if len(pred_scores) > int(args.post_max_size):
        order = np.argsort(-pred_scores)[:int(args.post_max_size)]
        pred_boxes, pred_names, pred_scores = pred_boxes[order], pred_names[order], pred_scores[order]

    return pred_boxes, pred_names, pred_scores


def infos_get_camera_dict(info):
    """Return a camera dict from info. Supports common SPD/mmdet3d schema variants."""
    cams = info.get('cams', None)
    if isinstance(cams, dict) and len(cams) > 0:
        # Prefer front camera if present.
        for key in ['CAM_FRONT', 'VEHICLE_CAM_FRONT', 'camera', 'front']:
            if key in cams:
                return key, cams[key]
        return list(cams.items())[0]

    # Some infos store a single camera directly.
    if 'cam_intrinsic' in info or 'lidar2img' in info or 'image_path' in info:
        return 'camera', info
    return None, None


def infos_resolve_image_path(info, args, cam=None):
    """Resolve image path from info/cam and data root."""
    candidates = []
    for obj in [cam, info]:
        if not isinstance(obj, dict):
            continue
        for k in ['data_path', 'img_path', 'image_path', 'filename']:
            if k in obj and obj[k]:
                candidates.append(obj[k])

    roots = [
        args.dataroot,
        os.path.join(args.dataroot, args.side),
        os.path.join(args.dataroot, 'vehicle-side'),
        os.path.join(args.dataroot, 'infrastructure-side'),
        os.path.join(args.dataroot, 'cooperative'),
    ]

    for p in candidates:
        if os.path.isabs(p) and os.path.isfile(p):
            return p
        for r in roots:
            cand = os.path.join(r, p)
            if os.path.isfile(cand):
                return cand
    return None


def infos_get_lidar2img(info, cam=None):
    """Get lidar2img matrix from info/cam if possible."""
    for obj in [cam, info]:
        if not isinstance(obj, dict):
            continue
        if 'lidar2img' in obj:
            arr = np.asarray(obj['lidar2img'], dtype=np.float64)
            if arr.shape == (4, 4):
                return arr
            if arr.size == 16:
                return arr.reshape(4, 4)
        # mmdet3d style: lidar2cam + cam_intrinsic.
        rot_keys = ['lidar2cam_rotation', 'lidar2cam_r', 'sensor2camera_rotation']
        trans_keys = ['lidar2cam_translation', 'lidar2cam_t', 'sensor2camera_translation']
        intr_keys = ['cam_intrinsic', 'camera_intrinsic', 'cam_K', 'intrinsic']
        rot = None
        trans = None
        intr = None
        for k in rot_keys:
            if k in obj:
                rot = np.asarray(obj[k], dtype=np.float64)
                break
        for k in trans_keys:
            if k in obj:
                trans = np.asarray(obj[k], dtype=np.float64).reshape(3)
                break
        for k in intr_keys:
            if k in obj:
                intr = np.asarray(obj[k], dtype=np.float64)
                break
        if rot is not None and trans is not None and intr is not None:
            rot = rot.reshape(3, 3)
            intr = intr.reshape(3, 4)[:3, :3] if intr.size == 12 else intr.reshape(3, 3)
            lidar2cam = np.eye(4, dtype=np.float64)
            lidar2cam[:3, :3] = rot
            lidar2cam[:3, 3] = trans
            lidar2img = np.eye(4, dtype=np.float64)
            lidar2img[:3, :4] = intr @ lidar2cam[:3, :4]
            return lidar2img
    return None


def infos_project_points(points_xyz, lidar2img):
    points = np.asarray(points_xyz, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((0, 2)), np.zeros((0,), dtype=bool), np.zeros((0,))
    homo = np.concatenate([points[:, :3], np.ones((len(points), 1))], axis=1)
    proj = homo @ lidar2img.T
    depth = proj[:, 2].copy()
    valid = depth > 1e-5
    uv = np.zeros((len(points), 2), dtype=np.float64)
    uv[valid, 0] = proj[valid, 0] / depth[valid]
    uv[valid, 1] = proj[valid, 1] / depth[valid]
    return uv, valid, depth


def infos_box_corners_3d(box):
    """Box format is [x,y,z,w,l,h,yaw], SECOND style: local x=length, local y=width."""
    x, y, z, w, l, h, yaw = box[:7]
    dx, dy, dz = float(l), float(w), float(h)
    corners = np.array([
        [ dx/2,  dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [-dx/2, -dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
        [ dx/2,  dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [-dx/2, -dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return corners @ rot.T + np.asarray([x, y, z], dtype=np.float64)


def infos_draw_projected_box(img, box, lidar2img, color=(0, 255, 0), thickness=2):
    corners = infos_box_corners_3d(box)
    uv, valid, depth = infos_project_points(corners, lidar2img)
    if valid.sum() < 4:
        return False
    h, w = img.shape[:2]
    # Skip boxes extremely outside the image.
    if uv[:, 0].max() < -w or uv[:, 0].min() > 2 * w or uv[:, 1].max() < -h or uv[:, 1].min() > 2 * h:
        return False
    pts = uv.astype(np.int32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        if valid[a] and valid[b]:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thickness, lineType=cv2.LINE_AA)
    return True


def render_infos_sample_camera(info, pts, pred_boxes, pred_names, pred_scores, args, out_path):
    """Render camera view with projected point cloud, GT boxes and prediction boxes."""
    cam_name, cam = infos_get_camera_dict(info)
    img_path = infos_resolve_image_path(info, args, cam)
    lidar2img = infos_get_lidar2img(info, cam)
    if img_path is None or lidar2img is None:
        # Save a small placeholder so the user knows why it failed.
        fig, ax = plt.subplots(figsize=(9, 3), dpi=160)
        ax.axis('off')
        ax.text(0.02, 0.65, 'Camera view unavailable', fontsize=14, color='red')
        ax.text(0.02, 0.42, f'image_path={img_path}', fontsize=9)
        ax.text(0.02, 0.25, f'cam={cam_name}, lidar2img={lidar2img is not None}', fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        return

    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]

    # Draw sparse point cloud on image.
    if pts is not None and len(pts) > 0:
        step = max(1, len(pts) // int(getattr(args, 'camera_max_points', 30000)))
        pts_sub = pts[::step, :3]
        uv, valid, depth = infos_project_points(pts_sub, lidar2img)
        in_img = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv_vis = uv[in_img].astype(np.int32)
        depth_vis = depth[in_img]
        if len(uv_vis) > 0:
            dmin, dmax = np.percentile(depth_vis, 2), np.percentile(depth_vis, 98)
            norm = np.clip((depth_vis - dmin) / (dmax - dmin + 1e-6), 0, 1)
            colors = (plt.cm.jet(1.0 - norm)[:, :3] * 255).astype(np.uint8)
            for p, c in zip(uv_vis, colors):
                cv2.circle(img, (int(p[0]), int(p[1])), int(getattr(args, 'camera_point_radius', 1)),
                           (int(c[2]), int(c[1]), int(c[0])), -1, lineType=cv2.LINE_AA)

    gt_boxes = np.asarray(info.get('gt_boxes', np.zeros((0, 7))), dtype=np.float32)
    gt_keep = np.asarray([infos_in_range(b, args.pc_range) for b in gt_boxes], dtype=bool) if len(gt_boxes) else np.zeros((0,), dtype=bool)
    for box in gt_boxes[gt_keep]:
        infos_draw_projected_box(img, box, lidar2img, color=(0, 255, 0), thickness=2)

    for box in pred_boxes:
        infos_draw_projected_box(img, box, lidar2img, color=(0, 0, 255), thickness=2)

    cv2.putText(img, f"Camera: {cam_name}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)



def run_infos_pkl_mode(args):
    infos = load_spd_infos_pkl(args.infos_pkl)
    start = max(0, int(args.start_idx))
    end = len(infos) if int(args.max_samples) < 0 else min(len(infos), start + int(args.max_samples))
    folder_path = os.path.join(args.out_folder, args.side)
    os.makedirs(folder_path, exist_ok=True)
    print(f'[InfosVis] Loaded infos: {len(infos)} from {args.infos_pkl}')
    print(f'[InfosVis] Rendering range: {start} -> {end - 1}')
    print(f'[InfosVis] pc_range: {args.pc_range}')
    print(f'[InfosVis] metric values: {get_metric_values_from_args(args)}')

    for local_i, idx in enumerate(tqdm(range(start, end))):
        info = infos[idx]
        cand_boxes, cand_names, cand_scores, pred_boxes, pred_names, pred_scores = simulate_infos_predictions_from_gt(
            info, args, frame_idx=idx, return_candidates=True
        )
        token = str(info.get('token', f'{idx:06d}'))
        out_path = os.path.join(folder_path, f'{idx:06d}_{token}.png')
        pts = render_infos_sample_bev(
            info, pred_boxes, pred_names, pred_scores, args, out_path,
            cand_boxes=cand_boxes, cand_names=cand_names, cand_scores=cand_scores
        )

        if getattr(args, 'render_camera_view', False):
            camera_folder = os.path.join(args.out_folder, args.side + '_camera')
            os.makedirs(camera_folder, exist_ok=True)
            camera_out_path = os.path.join(camera_folder, f'{idx:06d}_{token}.png')
            render_infos_sample_camera(info, pts, pred_boxes, pred_names, pred_scores, args, camera_out_path)

    video_path = os.path.join(args.out_folder, args.side + '.avi')
    if not getattr(args, 'no_video', False):
        to_video_safe(folder_path, video_path, fps=int(args.fps), downsample=int(args.downsample))
    print(f'[InfosVis] Done. Images: {folder_path}')
    print(f'[InfosVis] Video: {video_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Original nuScenes json visualization with metric-aware extension')
    parser.add_argument('--predroot', default='test/tiny_track_r50_stream_bs8_48epoch_3cls/Sun_Dec_29_18_24_23_2024/results_nusc.json', help='Path to result json. Optional when --simulate-from-gt is used.')
    parser.add_argument('--out_folder', default='result_vis/pf-track', help='Output folder path')
    parser.add_argument('--side', default='vehicle-side', choices=['vehicle-side', 'infrastructure-side', 'cooperative'], help='data side')
    parser.add_argument('--is_gt', default=True, help='for cooperative rendering: whether to draw GT instead of predictions')
    parser.add_argument('--dataroot', default='datasets/V2X-Seq-SPD-Batch-65-10-10761/', help='path to data root that contains vehicle-side/infrastructure-side/cooperative')
    parser.add_argument('--version', default='v1.0-trainval', help='data version')
    parser.add_argument('--thre', type=float, default=0.15, help='filter threshold')
    parser.add_argument('--start-idx', type=int, default=73, help='start index of sample token list')
    parser.add_argument('--max-samples', type=int, default=-1, help='max rendered samples; -1 means all remaining samples')
    parser.add_argument('--fps', type=int, default=4, help='output video fps')
    parser.add_argument('--downsample', type=int, default=1, help='video downsample ratio')

    # Metric-aware detection simulation / preset arguments from the second code.
    parser.add_argument('--simulate-from-gt', action='store_true', help='Ignore predroot and simulate prediction boxes from GT according to metric-like errors')
    parser.add_argument('--det-preset', choices=['none', 'old', 'new'], default='none', help='Metric preset: old/new/none')
    parser.add_argument('--sim-mate', type=float, default=None, help='mATE-like center translation noise in meters')
    parser.add_argument('--sim-mase', type=float, default=None, help='mASE-like scale perturbation ratio')
    parser.add_argument('--sim-maoe', type=float, default=None, help='mAOE-like yaw noise in radians')
    parser.add_argument('--sim-mave', type=float, default=None, help='mAVE-like velocity noise')
    parser.add_argument('--sim-fn-rate', type=float, default=None, help='false negative / missing detection rate')
    parser.add_argument('--sim-fp-rate', type=float, default=None, help='false positive ratio per GT')
    parser.add_argument('--sim-score', type=float, default=0.75, help='base score for simulated predictions')
    parser.add_argument('--save-sim-json', default=None, help='optional path to save metric-aware simulated/processed json')

    # Optional post-processing / realism controls from the second code.
    parser.add_argument('--random-drop-pred', action='store_true', help='randomly drop some predictions before rendering')
    parser.add_argument('--drop-prob', type=float, default=0.15, help='random drop probability')
    parser.add_argument('--keep-top1-per-class', action='store_true', help='keep highest-score prediction of each class during random drop')
    parser.add_argument('--apply-circle-nms', '--nms', dest='apply_circle_nms', action='store_true', help='apply class-aware center-distance NMS')
    parser.add_argument('--circle-nms-radius', '--nms-radius', dest='circle_nms_radius', type=float, default=1.5, help='center-distance NMS radius')
    parser.add_argument('--post-max-size', '--max-per-frame', dest='post_max_size', type=int, default=300, help='maximum predictions kept per frame')
    parser.add_argument('--seed', type=int, default=0, help='random seed for metric simulation/drop')

    # Direct SPD infos pkl mode. This is used when the nuScenes database files
    # are unavailable or when you want to render directly from mmdet3d infos.
    parser.add_argument('--infos-pkl', default=None, help='Directly render from spd_infos_temporal_val.pkl; bypass NuScenes API')
    parser.add_argument('--pc-range', type=float, nargs=4, default=[-54.0, -54.0, 54.0, 54.0],
                        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'), help='BEV visualization range for infos-pkl mode')
    parser.add_argument('--point-size', type=float, default=0.18, help='Point size for lidar scatter in infos-pkl mode')
    parser.add_argument('--point-alpha', type=float, default=0.32, help='Point alpha for lidar scatter in infos-pkl mode')
    parser.add_argument('--no-video', action='store_true', help='Do not generate avi video')
    parser.add_argument('--simulate-sparse4d-900', action='store_true',
                        help='Simulate Sparse4D-v3 style 900 sparse query candidates copied around GT/object-centered boxes before filtering and NMS')
    parser.add_argument('--num-sparse4d-candidates', type=int, default=900,
                        help='Number of sparse query candidates, default 900')
    parser.add_argument('--sparse4d-object-ratio', type=float, default=0.72,
                        help='Ratio of candidates copied around objects; the rest are background queries')
    parser.add_argument('--sparse4d-center-std', type=float, default=0.85,
                        help='Center noise std for Sparse4D candidate copies')
    parser.add_argument('--sparse4d-size-std', type=float, default=0.12,
                        help='Size noise std for Sparse4D candidate copies')
    parser.add_argument('--sparse4d-yaw-std', type=float, default=0.22,
                        help='Yaw noise std for Sparse4D candidate copies')
    parser.add_argument('--sparse4d-score-std', type=float, default=0.12,
                        help='Score noise std for Sparse4D candidate copies')
    parser.add_argument('--show-anchor-before-nms', action='store_true',
                        help='Draw a subset of Sparse4D candidates before NMS with light lines')
    parser.add_argument('--anchor-vis-thr', type=float, default=0.05,
                        help='Score threshold for visualizing pre-NMS sparse query candidates')
    parser.add_argument('--max-anchor-vis', type=int, default=250,
                        help='Maximum number of pre-NMS candidates drawn in one BEV frame')
    parser.add_argument('--render-camera-view', action='store_true',
                        help='Also render camera view with projected point cloud and 3D boxes')
    parser.add_argument('--camera-max-points', type=int, default=30000,
                        help='Maximum lidar points used for camera projection')
    parser.add_argument('--camera-point-radius', type=int, default=1,
                        help='Point radius for camera projection')

    # Compatibility with earlier metric commands. These are currently ignored in
    # detection BEV visualization but accepted to avoid argparse errors.
    parser.add_argument('--track-fn-rate', type=float, default=None)
    parser.add_argument('--track-pos-noise', type=float, default=None)
    parser.add_argument('--track-ids-total', type=float, default=None)

    args = parser.parse_args()

    data_root = args.dataroot
    side = args.side
    root_path = args.out_folder
    thre = float(args.thre)
    is_gt = parse_bool_flag(args.is_gt)

    if args.infos_pkl is not None:
        run_infos_pkl_mode(args)
        sys.exit(0)

    # Initialize nuScenes first. This is also needed for GT-based simulation.
    nusc = NuScenes(version=args.version, dataroot=data_root + side, verbose=True)

    # Decide sample list and prediction data.
    real_results = None
    if not args.simulate_from_gt:
        real_results = mmcv.load(args.predroot)
        sample_token_list = list(real_results['results'].keys())
    else:
        sample_token_list = [sample['token'] for sample in nusc.sample]
        print('[MetricVis] --simulate-from-gt enabled: predictions will be generated from GT with metric preset.')
        print('[MetricVis] metric values:', get_metric_values_from_args(args))

    sample_token_list = sample_token_list[int(args.start_idx):]
    if int(args.max_samples) > 0:
        sample_token_list = sample_token_list[:int(args.max_samples)]

    bevformer_results = build_metric_aware_pred_data(
        nusc=nusc,
        sample_token_list=sample_token_list,
        args=args,
        side=side,
        real_pred_data=real_results,
    )
    save_metric_sim_json(bevformer_results, args.save_sim_json)

    folder_path = os.path.join(root_path, side)
    video_path = os.path.join(root_path, side + '.avi')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    from nuscenes.eval.common.config import config_factory
    cfg = config_factory("tracking_nips_2019")
    if side == 'cooperative':
        inf_root = data_root + 'infrastructure-side'
        nusc_inf = NuScenes(version=args.version, dataroot=inf_root, verbose=True)
        veh2inf = {}
        coop_info = read_json(os.path.join(data_root, 'cooperative/data_info.json'))
        for f in coop_info:
            veh2inf[f['vehicle_frame']] = f['infrastructure_frame']
            veh2inf[f['vehicle_frame'] + 'offset'] = f['system_error_offset']

    for idx, sample_token in enumerate(tqdm(sample_token_list)):
        if side != 'cooperative':
            render_sample_data(sample_token, pred_data=bevformer_results, out_path=folder_path, side=side, thre=thre, nusc=nusc)
        else:
            render_sample_data_coop(sample_token, pred_data=bevformer_results, out_path=folder_path, side=side, thre=thre, veh2inf=veh2inf, nusc=nusc, nusc_inf=nusc_inf, is_gt=is_gt)

    to_video(folder_path=folder_path, out_path=video_path, fps=int(args.fps), downsample=int(args.downsample))
    print(f'[MetricVis] Done. Images: {folder_path}')
    print(f'[MetricVis] Video: {video_path}')
