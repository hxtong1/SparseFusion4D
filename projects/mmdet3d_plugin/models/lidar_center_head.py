from typing import Dict, List, Tuple
from collections import defaultdict
import torch
import numpy as np
from torch import Tensor, nn
from torch.cuda.amp import autocast
from mmcv.ops import nms_rotated
# from mmengine.structures import InstanceData
from mmcv.utils import print_log
from mmdet.models.utils.gaussian_target import get_local_maximum
# from mmdet.models.utils import multi_apply
from mmdet.core import multi_apply
from mmdet3d.models.builder import HEADS, build_head
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.core import circle_nms
from mmdet3d.core.bbox import LiDARInstance3DBoxes
# from detection.models.heads.bev_head. _centerpoint_head import  CenterHead
from .center_head import SparseFusionCenterHead
from mmcv.utils import build_from_cfg
# from mmdet3d.core.bbox import build_bbox_coder
# from detection.models.utils.det3d_utils. _gaussian import draw_heatmap_gaussian_with_yaw
# from detection.models.utils.det3d_utils import VelocityCalculator

from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead  # noqa: F401
from mmdet3d.core.bbox.coders.centerpoint_bbox_coders import CenterPointBBoxCoder  # noqa: F401

def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y

@HEADS.register_module()
class LidarSparseFusionCenterHead(SparseFusionCenterHead):

    def __init__(self,
                 *args,
                 instance_feat_type='cls_embedding',
                 embed_dim=128,
                 filter_invisible=False,
                 loss_motion_type=None,
                 loss_frame_offset=None,
                 init_cfg=None,
                 **kwargs):
        super(LidarSparseFusionCenterHead, self).__init__(
            *args,
            init_cfg=init_cfg,
            **kwargs
        )

        self.instance_feat_type = instance_feat_type
        self.embed_dim = embed_dim
        self.filter_invisible = filter_invisible

        assert self.instance_feat_type in [
            'cls_embedding',
            'grid_feat',
            'cls_guide_grid_feat'
        ], f'Unknown instance_feat_type: {self.instance_feat_type}'

        self._init_instance_feat_generator()

        # self.with_motion_loss = self.train_cfg.get('loss_motion', False)
        self.with_motion_loss = False
        self.loss_motion_type = None
        self.loss_frame_offset = None

        if self.with_motion_loss:
            assert loss_motion_type is not None, \
                'loss_motion=True requires loss_motion_type.'
            assert loss_frame_offset is not None, \
                'loss_motion=True requires loss_frame_offset.'

            self.loss_motion_type = HEADS.build(loss_motion_type)
            self.loss_frame_offset = HEADS.build(loss_frame_offset)
        else:
            self.loss_motion_type = None
            self.loss_frame_offset = None

    def _init_instance_feat_generator(self):
        num_classes = sum(self.num_classes)
        padding_cls_id = num_classes

        if self.instance_feat_type == 'cls_embedding':
            self.cls_embedding = nn.Embedding(
                num_classes + 1,
                self.embed_dim,
                padding_idx=padding_cls_id
            )

        elif self.instance_feat_type == 'grid_feat':
            self.cls_embedding = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dim)
            )

        elif self.instance_feat_type == 'cls_guide_grid_feat':
            self.cls_embedding = nn.Embedding(
                num_classes + 1,
                self.embed_dim,
                padding_idx=padding_cls_id
            )
            self.cls_guide_fusion = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dim)
            )

        else:
            raise NotImplementedError(
                f'Unknown instance_feat_type: {self.instance_feat_type}'
            )

    def filter_anns(self,
                    gt_bboxes_3d: List[Tensor],
                    gt_labels_3d: List[Tensor],
                    img_metas: List[Dict] = None) -> Tuple[List[Tensor], List[Tensor]]:
        cfg = self.train_cfg

        if img_metas is None:
            return gt_bboxes_3d, gt_labels_3d

        for i, (bboxes, labels) in enumerate(zip(gt_bboxes_3d, gt_labels_3d)):
            meta = img_metas[i]

            if self.filter_invisible and 'lidar_visible' in meta:
                lidar_visible = meta['lidar_visible']
                if not torch.is_tensor(lidar_visible):
                    lidar_visible = torch.as_tensor(
                        lidar_visible,
                        device=bboxes.device
                    )
                else:
                    lidar_visible = lidar_visible.to(bboxes.device)

                lidar_visible = lidar_visible.bool()
                bboxes = bboxes[lidar_visible]
                labels = labels[lidar_visible]

            if cfg.get('filter_cam_invisible', False) and 'is_cam_visible' in meta:
                is_cam_visible = meta['is_cam_visible']
                if not torch.is_tensor(is_cam_visible):
                    is_cam_visible = torch.as_tensor(
                        is_cam_visible,
                        device=bboxes.device
                    )
                else:
                    is_cam_visible = is_cam_visible.to(bboxes.device)

                any_cam_visible = is_cam_visible[:, -1].bool()
                bboxes = bboxes[any_cam_visible]
                labels = labels[any_cam_visible]

            gt_bboxes_3d[i] = bboxes
            gt_labels_3d[i] = labels

        return gt_bboxes_3d, gt_labels_3d

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, file_name=None):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (Tensor): 
                [x, y, z, dx, dy, dz, yaw] or
                [x, y, z, dx, dy, dz, yaw, vx, vy]
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device

        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']  # type: ignore
        grid_size = torch.tensor(
            self.train_cfg['grid_size'],
            device=device,
            dtype=torch.long
        )
        pc_range = torch.tensor(
            self.train_cfg['point_cloud_range'],
            device=device,
            dtype=gt_bboxes_3d.dtype
        )
        voxel_size = torch.tensor(
            self.train_cfg['voxel_size'],
            device=device,
            dtype=gt_bboxes_3d.dtype
        )
        with_motion_loss = getattr(
            self,
            'with_motion_loss',
            self.train_cfg.get('loss_motion', False)
        )
        # ---------------------------------------------------------
        # 判断当前 GT 是否包含 nuScenes velocity.
        # gt_bboxes_3d:
        #   7 dims: [x, y, z, dx, dy, dz, yaw]
        #   9 dims: [x, y, z, dx, dy, dz, yaw, vx, vy]
        # ---------------------------------------------------------

        has_velocity = gt_bboxes_3d.shape[-1] >= 9
        # CenterHead target dim:
        #   without velocity: reg(2) + height(1) + dim(3) + rot(2) = 8
        #   with velocity:    reg(2) + height(1) + dim(3) + rot(2) + vel(2) = 10
        num_box_dim = 10 if has_velocity else 8
        # ---------------------------------------------------------
        # Reorganize GT by detection tasks.
        # self.class_names example:
        # [
        #   ['car', 'truck', ...],
        #   ['pedestrian'],
        #   ...
        # ]
        # ---------------------------------------------------------
        task_masks = []
        flag = 0

        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(cls_name) + flag)
                for cls_name in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0

        for task_id, mask_per_task in enumerate(task_masks):
            cur_task_boxes = []
            cur_task_classes = []

            for m in mask_per_task:
                if m[0].numel() > 0:
                    cur_task_boxes.append(gt_bboxes_3d[m])
                    # Each task uses local class ids starting from 1.
                    # 0 is reserved as background.
                    cur_task_classes.append(gt_labels_3d[m] + 1 - flag2)

            if len(cur_task_boxes) > 0:
                cur_task_boxes = torch.cat(cur_task_boxes, dim=0).to(device)
                cur_task_classes = torch.cat(cur_task_classes, dim=0).long().to(device)
            else:
                cur_task_boxes = gt_bboxes_3d.new_zeros(
                    (0, gt_bboxes_3d.shape[-1])
                )
                cur_task_classes = gt_labels_3d.new_zeros(
                    (0,),
                    dtype=torch.long
                )

            task_boxes.append(cur_task_boxes)
            task_classes.append(cur_task_classes)

            flag2 += len(mask_per_task)

        heatmaps = []
        anno_boxes = []
        inds = []
        cats = []
        masks = []
        segmaps = []

        for task_id, task_head in enumerate(self.task_heads):
            out_size_factor = self.train_cfg['out_size_factor'][task_id]

            feature_map_size = grid_size[:2] // out_size_factor

            heatmap = gt_bboxes_3d.new_zeros(
                (
                    len(self.class_names[task_id]),
                    int(feature_map_size[1]),
                    int(feature_map_size[0])
                )
            )

            anno_box = gt_bboxes_3d.new_zeros(
                (max_objs, num_box_dim),
                dtype=torch.float32
            )

            ind = gt_labels_3d.new_zeros(
                (max_objs,),
                dtype=torch.int64
            )

            cat = gt_labels_3d.new_zeros(
                (max_objs,),
                dtype=torch.int64
            )

            mask = gt_bboxes_3d.new_zeros(
                (max_objs,),
                dtype=torch.uint8
            )

            if self.loss_auxiliary_seg is not None:
                segmap = gt_bboxes_3d.new_zeros(
                    (
                        len(self.class_names[task_id]),
                        int(feature_map_size[1]),
                        int(feature_map_size[0])
                    )
                )
                segmap = segmap.cpu().numpy()
            else:
                segmap = None

            num_objs = min(task_boxes[task_id].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[task_id][k] - 1

                # Original box dimensions in meters.
                length = task_boxes[task_id][k][3]
                width = task_boxes[task_id][k][4]

                # Convert box size from meters to feature-map units.
                length_feat = length / voxel_size[0] / out_size_factor
                width_feat = width / voxel_size[1] / out_size_factor

                if length_feat <= 0 or width_feat <= 0:
                    continue

                x = task_boxes[task_id][k][0]
                y = task_boxes[task_id][k][1]
                z = task_boxes[task_id][k][2]

                coor_x = (x - pc_range[0]) / voxel_size[0] / out_size_factor
                coor_y = (y - pc_range[1]) / voxel_size[1] / out_size_factor

                center = torch.stack([coor_x, coor_y]).to(torch.float32)
                center_int = center.to(torch.int32)

                # Filter boxes outside feature map.
                if not (
                    0 <= center_int[0] < feature_map_size[0]
                    and 0 <= center_int[1] < feature_map_size[1]
                ):
                    continue

                # -----------------------------------------------------
                # Draw Gaussian heatmap.
                # Sparse4D / old mmdet3d environment usually uses
                # draw_heatmap_gaussian instead of yaw-aware gaussian.
                # -----------------------------------------------------
                radius = gaussian_radius(
                    (length_feat, width_feat),
                    min_overlap=self.train_cfg['gaussian_overlap']
                )
                radius = max(self.train_cfg['min_radius'], int(radius))

                draw_heatmap_gaussian(
                    heatmap[cls_id],
                    center_int,
                    radius
                )

                grid_x = center_int[0]
                grid_y = center_int[1]

                assert (
                    grid_y * feature_map_size[0] + grid_x
                    < feature_map_size[0] * feature_map_size[1]
                )

                ind[k] = grid_y * feature_map_size[0] + grid_x
                cat[k] = cls_id
                mask[k] = 1

                rot = task_boxes[task_id][k][6]
                box_dim = task_boxes[task_id][k][3:6]

                if self.loss_auxiliary_seg is not None:
                    self.draw_segmap(
                        segmap[cls_id],
                        grid_x.detach().cpu().numpy(),
                        grid_y.detach().cpu().numpy(),
                        length_feat.detach().cpu().numpy(),
                        width_feat.detach().cpu().numpy(),
                        rot.detach().cpu().numpy()
                    )

                if self.norm_bbox:
                    box_dim = box_dim.log()

                # -----------------------------------------------------
                # Basic box target:
                # [offset_x, offset_y, z, dx, dy, dz, sin_yaw, cos_yaw]
                # -----------------------------------------------------
                anno_box[k, :8] = torch.cat([
                    center - torch.stack([grid_x, grid_y]).to(
                        device=device,
                        dtype=center.dtype
                    ),
                    z.unsqueeze(0),
                    box_dim,
                    torch.sin(rot).unsqueeze(0),
                    torch.cos(rot).unsqueeze(0)
                ])

                # -----------------------------------------------------
                # nuScenes velocity target:
                # [vx, vy]
                # -----------------------------------------------------
                if has_velocity:
                    anno_box[k, 8:10] = task_boxes[task_id][k][7:9]

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            inds.append(ind)
            cats.append(cat)
            masks.append(mask)

            if self.loss_auxiliary_seg is not None:
                segmap = torch.tensor(
                    segmap,
                    device=heatmap.device,
                    dtype=heatmap.dtype
                )
                segmaps.append(segmap)

        return heatmaps, anno_boxes, inds, cats, masks, segmaps

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, file_name=None):
        """Generate targets for a batch.

        Args:
            gt_bboxes_3d (list[Tensor]): GT boxes of each sample.
            gt_labels_3d (list[Tensor]): GT labels of each sample.
            file_name (list | None): Optional file names.

        Returns:
            tuple:
                heatmaps, anno_boxes, inds, cats, masks, segmaps
        """
        num_samples = len(gt_bboxes_3d)

        if file_name is None:
            file_name = [None for _ in range(num_samples)]

        heatmaps, anno_boxes, inds, cats, masks, segmaps = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            file_name
        )

        # batch dimension -> task dimension
        # from:
        #   batch list: [[task0, task1], [task0, task1], ...]
        # to:
        #   task list:  [stack(batch_task0), stack(batch_task1)]
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms, dim=0) for hms in heatmaps]

        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(boxes, dim=0) for boxes in anno_boxes]

        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(ind, dim=0) for ind in inds]

        cats = list(map(list, zip(*cats)))
        cats = [torch.stack(cat, dim=0) for cat in cats]

        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(mask, dim=0) for mask in masks]

        if self.loss_auxiliary_seg is not None:
            segmaps = list(map(list, zip(*segmaps)))
            segmaps = [torch.stack(seg, dim=0) for seg in segmaps]
        else:
            segmaps = None

        return heatmaps, anno_boxes, inds, cats, masks, segmaps

    def calculate_motion_loss(self,
                            pred_motion_types: Tensor = None,
                            pred_vels: Tensor = None,
                            gt_motion_types: Tensor = None,
                            gt_vels: Tensor = None,
                            pos_inds: Tensor = None,
                            mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """Calculate optional motion type loss and nuScenes velocity loss.

        This function is adapted for Sparse4D v3 / old mmdet3d style with
        nuScenes official annotations.

        Args:
            pred_motion_types (Tensor | None): Predicted motion type logits,
                shape [B, C_m, H, W]. This branch is optional. It can be used
                when nuScenes attribute labels are converted to motion types.
            pred_vels (Tensor | None): Predicted velocity map, shape [B, 2, H, W].
                It corresponds to nuScenes velocity labels [vx, vy], not frame offsets.
            gt_motion_types (Tensor | None): GT motion type labels, shape [B, max_objs].
                Invalid / ignored labels should be set to -1.
            gt_vels (Tensor | None): GT velocity labels, shape [B, max_objs, 2].
            pos_inds (Tensor): Positive indices on feature map, shape [B, max_objs].
            mask (Tensor): Valid object mask, shape [B, max_objs].

        Returns:
            tuple[Tensor, Tensor]:
                loss_motion_type, loss_velocity
        """

        assert pos_inds is not None, 'pos_inds should not be None.'
        assert mask is not None, 'mask should not be None.'

        # ---------------------------------------------------------
        # Empty positive samples.
        # ---------------------------------------------------------
        if mask.sum() == 0:
            if pred_motion_types is not None:
                loss_motion_type = pred_motion_types.sum() * 0.0
            else:
                loss_motion_type = mask.float().sum() * 0.0

            if pred_vels is not None:
                loss_velocity = pred_vels.sum() * 0.0
            else:
                loss_velocity = mask.float().sum() * 0.0

            return loss_motion_type, loss_velocity

        # ---------------------------------------------------------
        # 1. Optional motion type loss.
        # Motion type can be derived from nuScenes attributes:
        # moving / stopped / parked / standing / unknown, etc.
        # If gt_motion_types is None, this branch returns zero loss.
        # ---------------------------------------------------------
        if pred_motion_types is not None and gt_motion_types is not None:
            pred_motion_types = pred_motion_types.permute(0, 2, 3, 1).contiguous()
            pred_motion_types = pred_motion_types.view(
                pred_motion_types.size(0), -1, pred_motion_types.size(-1)
            )
            pred_motion_types = self._gather_feat(pred_motion_types, pos_inds)

            valid_motion_mask = mask.bool() & (gt_motion_types >= 0)

            if valid_motion_mask.sum() == 0:
                loss_motion_type = pred_motion_types.sum() * 0.0
            else:
                loss_motion_type = self.loss_motion_type(
                    pred_motion_types[valid_motion_mask],
                    gt_motion_types[valid_motion_mask].long(),
                    avg_factor=max(1.0, float(valid_motion_mask.sum()))
                )
        else:
            if pred_motion_types is not None:
                loss_motion_type = pred_motion_types.sum() * 0.0
            else:
                loss_motion_type = mask.float().sum() * 0.0

        # ---------------------------------------------------------
        # 2. Velocity loss.
        # nuScenes official boxes usually provide velocity [vx, vy].
        # This is NOT frame offset. Do not divide it by frame_time_gap again.
        # Invalid velocity values can be NaN, so they are masked out.
        # ---------------------------------------------------------
        if pred_vels is not None and gt_vels is not None:
            pred_vels = pred_vels.permute(0, 2, 3, 1).contiguous()
            pred_vels = pred_vels.view(
                pred_vels.size(0), -1, pred_vels.size(-1)
            )
            pred_vels = self._gather_feat(pred_vels, pos_inds)

            valid_vel_mask = mask.bool() & (~torch.isnan(gt_vels).any(dim=-1))

            if valid_vel_mask.sum() == 0:
                loss_velocity = pred_vels.sum() * 0.0
            else:
                loss_velocity = self.loss_frame_offset(
                    pred_vels[valid_vel_mask],
                    gt_vels[valid_vel_mask],
                    avg_factor=max(1.0, float(valid_vel_mask.sum()))
                )
        else:
            if pred_vels is not None:
                loss_velocity = pred_vels.sum() * 0.0
            else:
                loss_velocity = mask.float().sum() * 0.0

        return loss_motion_type, loss_velocity

    def loss_by_feat(self,
                    preds_dicts: Dict,
                    gt_bboxes_3d: List,
                    gt_labels_3d: List[Tensor],
                    img_metas: List[Dict] = None) -> Dict[str, Tensor]:
        """Calculate losses from CenterHead predictions.

        This version is adapted for Sparse4D v3 / old mmdet3d style and
        nuScenes official annotations.

        Args:
            preds_dicts (list[dict] | tuple[dict]): Predictions of each task head.
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): GT 3D boxes of each sample.
                For nuScenes, boxes.tensor[:, 3:] usually contains:
                [dx, dy, dz, yaw, vx, vy].
            gt_labels_3d (list[Tensor]): GT labels of each sample.
            img_metas (list[dict], optional): NuScenes/Sparse4D meta information.

        Returns:
            dict[str, Tensor]: Loss dictionary.
        """
        if img_metas is None:
            img_metas = [dict() for _ in range(len(gt_bboxes_3d))]

        file_name = [
            meta.get('sample_idx', meta.get('filename', None))
            for meta in img_metas
        ]

        # ---------------------------------------------------------
        # LiDARInstance3DBoxes -> Tensor
        #
        # nuScenes standard box tensor:
        # boxes.gravity_center: [x, y, z]
        # boxes.tensor[:, 3:]: [dx, dy, dz, yaw, vx, vy] if velocity exists
        #
        # Final gt_bboxes_tensor:
        # [x, y, z, dx, dy, dz, yaw, vx, vy]
        # ---------------------------------------------------------
        gt_bboxes_tensor = []

        for boxes, labels in zip(gt_bboxes_3d, gt_labels_3d):
            # 当前 Sparse4D dataset + pipeline 通常走这里：
            # [x, y, z, dx, dy, dz, yaw, vx, vy]
            if torch.is_tensor(boxes):
                boxes_tensor = boxes

            # 如果 pipeline 还没转 Tensor，兼容 numpy.ndarray
            elif isinstance(boxes, np.ndarray):
                boxes_tensor = torch.from_numpy(boxes)

            # 兼容旧版 mmdet3d 的 LiDARInstance3DBoxes
            elif hasattr(boxes, "gravity_center") and hasattr(boxes, "tensor"):
                boxes_tensor = torch.cat(
                    [boxes.gravity_center, boxes.tensor[:, 3:]],
                    dim=1
                )

            else:
                raise TypeError(
                    f"Unsupported gt_bboxes_3d type: {type(boxes)}"
                )

            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels)

            boxes_tensor = boxes_tensor.to(
                device=labels.device,
                dtype=torch.float32
            )

            gt_bboxes_tensor.append(boxes_tensor)

        # ---------------------------------------------------------
        # Optional visibility filtering.
        # Standard nuScenes/Sparse4D img_metas usually do not contain
        # lidar_visible / is_cam_visible, so filter_anns should safely
        # skip these fields when unavailable.
        # ---------------------------------------------------------
        gt_bboxes_tensor, gt_labels_3d = self.filter_anns(
            gt_bboxes_tensor,
            gt_labels_3d,
            img_metas
        )

        heatmaps, anno_boxes, inds, cats, masks, segmps = self.get_targets(
            gt_bboxes_tensor,
            gt_labels_3d,
            file_name
        )

        loss_dict = dict()

        for task_id, preds_dict in enumerate(preds_dicts):
            with autocast(enabled=False):
                # -------------------------------------------------
                # Heatmap loss
                # -------------------------------------------------
                preds_dict['heatmap'] = clip_sigmoid(
                    preds_dict['heatmap'].float()
                )

                num_pos = heatmaps[task_id].eq(1).float().sum().item()
                loss_heatmap = self.loss_cls(
                    preds_dict['heatmap'],
                    heatmaps[task_id],
                    avg_factor=max(num_pos, 1)
                )

                # -------------------------------------------------
                # Optional auxiliary segmentation loss
                # -------------------------------------------------
                if self.loss_auxiliary_seg is not None:
                    preds_dict['segmap'] = clip_sigmoid(
                        preds_dict['segmap'].float()
                    )
                    num_pos = segmps[task_id].eq(1).float().sum().item()
                    loss_segmap = self.loss_auxiliary_seg(
                        preds_dict['segmap'],
                        segmps[task_id],
                        avg_factor=max(num_pos, 1)
                    )

                # -------------------------------------------------
                # BBox target
                #
                # target_box from get_targets_single:
                #   without velocity: [reg_x, reg_y, z, dx, dy, dz, sin, cos]
                #   with velocity:    [reg_x, reg_y, z, dx, dy, dz, sin, cos, vx, vy]
                # -------------------------------------------------
                target_box = anno_boxes[task_id]

                attr_name = ['reg', 'height', 'dim', 'rot']

                # If both prediction and target contain velocity, supervise vel.
                # Otherwise, fall back to 8-dim box target.
                if 'vel' in preds_dict and target_box.shape[-1] >= 10:
                    attr_name.append('vel')
                else:
                    target_box = target_box[..., :8]

                preds_dict['anno_box'] = torch.cat(
                    [preds_dict[attr] for attr in attr_name],
                    dim=1
                )

                # -------------------------------------------------
                # Gather predictions at positive locations
                # -------------------------------------------------
                ind = inds[task_id]
                num = masks[task_id].float().sum()

                pred = preds_dict['anno_box'].permute(0, 2, 3, 1).contiguous()
                pred = pred.view(pred.size(0), -1, pred.size(3))
                pred = self._gather_feat(pred, ind)

                mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
                isnotnan = (~torch.isnan(target_box)).float()
                mask *= isnotnan

                # -------------------------------------------------
                # Code weights
                #
                # Supported formats:
                #   1) [box_dim]
                #   2) [num_classes, box_dim] for current task
                #   3) [num_tasks, num_classes, box_dim]
                # -------------------------------------------------
                code_weights_cfg = self.train_cfg.get('code_weights', None)

                if code_weights_cfg is None:
                    bbox_weights = mask
                else:
                    code_weights = mask.new_tensor(code_weights_cfg)

                    if code_weights.dim() == 1:
                        # [box_dim]
                        code_weights = code_weights.view(1, 1, -1)
                        code_weights = code_weights[..., :target_box.shape[-1]]
                        code_weights = code_weights.expand_as(target_box)

                    elif code_weights.dim() == 2:
                        # [num_classes, box_dim]
                        code_weights = code_weights[:, :target_box.shape[-1]]
                        code_weights = code_weights.unsqueeze(0).expand(
                            len(gt_bboxes_tensor), -1, -1
                        )
                        code_weights = code_weights.gather(
                            1,
                            cats[task_id].unsqueeze(2).expand(
                                -1, -1, code_weights.shape[-1]
                            )
                        )

                    elif code_weights.dim() == 3:
                        # [num_tasks, num_classes, box_dim]
                        code_weights = code_weights[task_id]
                        code_weights = code_weights[:, :target_box.shape[-1]]
                        code_weights = code_weights.unsqueeze(0).expand(
                            len(gt_bboxes_tensor), -1, -1
                        )
                        code_weights = code_weights.gather(
                            1,
                            cats[task_id].unsqueeze(2).expand(
                                -1, -1, code_weights.shape[-1]
                            )
                        )

                    else:
                        raise ValueError(
                            f'Unsupported code_weights shape: {code_weights.shape}'
                        )

                    bbox_weights = mask * code_weights

                loss_bbox = self.loss_bbox(
                    pred,
                    target_box,
                    bbox_weights,
                    avg_factor=(num + 1e-4)
                )

                loss_dict[f'task{task_id}/loss_heatmap'] = loss_heatmap
                loss_dict[f'task{task_id}/loss_bbox'] = loss_bbox

                if self.loss_auxiliary_seg is not None:
                    loss_dict[f'task{task_id}/loss_segmap'] = loss_segmap * 4

        return loss_dict

    def predict_by_feat(self, preds_dicts: Dict) -> List[Dict]:
        """Decode CenterHead predictions.

        This version is adapted for Sparse4D v3 / old mmdet3d style.
        It does not use mmengine.structures.InstanceData.
        """
        rets = []

        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict['heatmap'].shape[0]

            batch_heatmap = preds_dict['heatmap'].sigmoid()
            if self.test_cfg['nms_type'] == 'local_maximum_heatmap':
                batch_heatmap = get_local_maximum(batch_heatmap)

            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict['dim'])
            else:
                batch_dim = preds_dict['dim']

            batch_rots = preds_dict['rot'][:, 0:1]
            batch_rotc = preds_dict['rot'][:, 1:2]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
            else:
                batch_vel = torch.zeros_like(preds_dict['rot'])

            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id
            )

            assert self.test_cfg['nms_type'] in [
                'circle', 'nms_bev', 'local_maximum_heatmap', 'rotate', None
            ]

            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]

            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']

                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)

                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']
                        ),
                        dtype=torch.long,
                        device=boxes.device
                    )

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                    ret_task.append(dict(
                        bboxes=boxes3d,
                        scores=scores,
                        labels=labels
                    ))

                rets.append(ret_task)

            elif self.test_cfg['nms_type'] == 'nms_bev':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']

                    boxes = boxes3d[:, [0, 1, 3, 4, 6]]
                    order = scores.sort(0, descending=True)[1]

                    boxes = boxes[order]
                    scores = scores[order]
                    labels = labels[order]
                    boxes3d = boxes3d[order]

                    keep = nms_rotated(
                        boxes,
                        scores,
                        self.test_cfg['thresh']
                    )[1]

                    if keep is None:
                        keep = torch.tensor(
                            [],
                            dtype=torch.long,
                            device=boxes.device
                        )

                    post_max_size = self.test_cfg['post_max_size']
                    if post_max_size is not None:
                        keep = keep[:post_max_size]

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                    ret_task.append(dict(
                        bboxes=boxes3d,
                        scores=scores,
                        labels=labels
                    ))

                rets.append(ret_task)

            elif self.test_cfg['nms_type'] == 'rotate':
                rets.append(
                    self.get_task_detections(
                        num_class_with_bg,
                        batch_cls_preds,
                        batch_reg_preds,
                        batch_cls_labels,
                        img_metas=None
                    )
                )

            elif (self.test_cfg['nms_type'] == 'local_maximum_heatmap'
                or self.test_cfg['nms_type'] is None):
                ret_task = [
                    dict(
                        bboxes=box['bboxes'],
                        scores=box['scores'],
                        labels=box['labels']
                    )
                    for box in temp
                ]
                rets.append(ret_task)

            else:
                raise NotImplementedError(
                    f'Unknown nms type: {self.test_cfg["nms_type"]}'
                )

        # ---------------------------------------------------------
        # Merge task results.
        # Return dict instead of InstanceData for Sparse4D v3.
        # ---------------------------------------------------------
        num_samples = len(rets[0])
        ret_list = []

        for i in range(num_samples):
            bboxes_list = [ret[i]['bboxes'] for ret in rets]
            scores_list = [ret[i]['scores'] for ret in rets]

            flag = 0
            labels_list = []
            for task_id, num_class in enumerate(self.num_classes):
                labels = rets[task_id][i]['labels'].int() + flag
                labels_list.append(labels)
                flag += num_class

            if len(bboxes_list) > 0:
                bboxes_3d = torch.cat(bboxes_list, dim=0)
                scores_3d = torch.cat(scores_list, dim=0)
                labels_3d = torch.cat(labels_list, dim=0)
            else:
                device = preds_dicts[0]['heatmap'].device
                dtype = preds_dicts[0]['heatmap'].dtype
                bboxes_3d = torch.zeros((0, self.bbox_coder.code_size), device=device, dtype=dtype)
                scores_3d = torch.zeros((0,), device=device, dtype=dtype)
                labels_3d = torch.zeros((0,), device=device, dtype=torch.long)

            if bboxes_3d.numel() > 0:
                bboxes_3d[:, 2] = bboxes_3d[:, 2] - bboxes_3d[:, 5] * 0.5

            # Convert z from gravity center to bottom center.
            bboxes_3d[:, 2] = bboxes_3d[:, 2] - bboxes_3d[:, 5] * 0.5

            bboxes_3d = LiDARInstance3DBoxes(
                bboxes_3d,
                self.bbox_coder.code_size
            )

            scores_3d = torch.cat([ret[i]['scores'] for ret in rets], dim=0)

            flag = 0
            labels_list = []
            for task_id, num_class in enumerate(self.num_classes):
                labels = rets[task_id][i]['labels'].int() + flag
                labels_list.append(labels)
                flag += num_class

            labels_3d = torch.cat(labels_list, dim=0)

            ret_list.append(dict(
                bboxes_3d=bboxes_3d,
                scores_3d=scores_3d,
                labels_3d=labels_3d
            ))

        return ret_list
    def boxes_to_anchors(self, boxes):
        """Convert CenterPoint decoded boxes to Sparse4D-style anchors.

        Args:
            boxes (Tensor): Shape [B, K, C] or [K, C].
                Expected format:
                [x, y, z, l, w, h, yaw] or
                [x, y, z, l, w, h, yaw, vx, vy]

        Returns:
            Tensor: Sparse4D-style anchors.
                [x, y, z, l, w, h, sin_yaw, cos_yaw, vx, vy]
        """
        squeeze_batch = False
        if boxes.dim() == 2:
            boxes = boxes.unsqueeze(0)
            squeeze_batch = True

        x = boxes[..., 0]
        y = boxes[..., 1]
        z = boxes[..., 2]
        l = boxes[..., 3]
        w = boxes[..., 4]
        h = boxes[..., 5]
        yaw = boxes[..., 6]

        sin_yaw = torch.sin(yaw)
        cos_yaw = torch.cos(yaw)

        if boxes.shape[-1] >= 9:
            vx = boxes[..., 7]
            vy = boxes[..., 8]
        else:
            vx = torch.zeros_like(x)
            vy = torch.zeros_like(y)

        anchors = torch.stack(
            [x, y, z, l, w, h, sin_yaw, cos_yaw, vx, vy],
            dim=-1
        )

        if squeeze_batch:
            anchors = anchors.squeeze(0)

        return anchors        
    def _pad_or_truncate_priors(self, boxes, scores, labels, max_num):
        """Pad or truncate proposals to fixed number.

        Args:
            boxes (Tensor): [N, C]
            scores (Tensor): [N]
            labels (Tensor): [N]
            max_num (int): Fixed number of proposals.

        Returns:
            tuple:
                boxes_out: [max_num, C]
                scores_out: [max_num]
                labels_out: [max_num]
        """
        device = boxes.device
        dtype = boxes.dtype
        box_dim = boxes.shape[-1]

        if boxes.shape[0] > 0:
            order = scores.sort(descending=True)[1]
            order = order[:max_num]

            boxes = boxes[order]
            scores = scores[order]
            labels = labels[order].long()
        else:
            boxes = boxes.new_zeros((0, box_dim))
            scores = scores.new_zeros((0,))
            labels = labels.new_zeros((0,), dtype=torch.long)

        num = boxes.shape[0]

        boxes_out = boxes.new_zeros((max_num, box_dim))
        scores_out = scores.new_zeros((max_num,))

        # padding label 使用 num_classes，正好对应 cls_embedding 的 padding_idx
        padding_cls_id = sum(self.num_classes)
        labels_out = labels.new_full(
            (max_num,),
            fill_value=padding_cls_id,
            dtype=torch.long
        )

        if num > 0:
            boxes_out[:num] = boxes
            scores_out[:num] = scores
            labels_out[:num] = labels

        return boxes_out, scores_out, labels_out

    def decode_lidar(self,
                    preds_dicts: List[Dict[str, Tensor]],
                    feats: List[Tensor] = None,
                    metainfo: Dict[str, Tensor] = None):
        """Decode LiDAR branch predictions into Sparse4D-style priors.

        This function uses official CenterPointBBoxCoder.decode() and then
        converts decoded boxes into Sparse4D prior anchors and instance features.

        Returns:
            dict:
                anchors: [B, N, 10]
                instance_feats: [B, N, embed_dim]
                scores: [B, N]
                labels: [B, N]
        """
        multi_task_boxes = []
        multi_task_scores = []
        multi_task_labels = []

        proposal_nums = self.test_cfg.get('proposal_nums', None)
        default_topk = self.test_cfg.get('post_max_size', 300)

        for task_id, preds_dict in enumerate(preds_dicts):
            batch_heatmap = preds_dict['heatmap'].sigmoid()

            if self.test_cfg.get('nms_type', None) == 'local_maximum_heatmap':
                batch_heatmap = get_local_maximum(batch_heatmap)

            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict['dim'])
            else:
                batch_dim = preds_dict['dim']

            batch_rots = preds_dict['rot'][:, 0:1]
            batch_rotc = preds_dict['rot'][:, 1:2]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
            else:
                batch_vel = torch.zeros_like(preds_dict['rot'])

            # -----------------------------------------------------
            # Use official CenterPointBBoxCoder interface.
            # Do NOT pass feat=... or mode='proposal'.
            # -----------------------------------------------------
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id
            )

            if proposal_nums is not None:
                task_topk = proposal_nums[task_id]
            else:
                task_topk = default_topk

            batch_boxes = []
            batch_scores = []
            batch_labels = []

            label_offset = sum(self.num_classes[:task_id])

            for sample_ret in temp:
                boxes = sample_ret['bboxes']
                scores = sample_ret['scores']
                labels = sample_ret['labels'].long() + label_offset

                boxes, scores, labels = self._pad_or_truncate_priors(
                    boxes,
                    scores,
                    labels,
                    task_topk
                )

                batch_boxes.append(boxes)
                batch_scores.append(scores)
                batch_labels.append(labels)

            batch_boxes = torch.stack(batch_boxes, dim=0)
            batch_scores = torch.stack(batch_scores, dim=0)
            batch_labels = torch.stack(batch_labels, dim=0)

            multi_task_boxes.append(batch_boxes)
            multi_task_scores.append(batch_scores)
            multi_task_labels.append(batch_labels)

        # [B, N, C]
        boxes = torch.cat(multi_task_boxes, dim=1)

        # [B, N]
        scores = torch.cat(multi_task_scores, dim=1)
        labels = torch.cat(multi_task_labels, dim=1).long()

        # ---------------------------------------------------------
        # Convert decoded boxes to Sparse4D-style anchors:
        # [x, y, z, l, w, h, sin_yaw, cos_yaw, vx, vy]
        # ---------------------------------------------------------
        anchors = self.boxes_to_anchors(boxes)

        # ---------------------------------------------------------
        # Generate instance features.
        # First version: cls_embedding only.
        # ---------------------------------------------------------
        if self.instance_feat_type == 'cls_embedding':
            instance_feats = self.cls_embedding(labels)

        elif self.instance_feat_type == 'grid_feat':
            raise NotImplementedError(
                'grid_feat requires BEV feature gathering by proposal centers. '
                'Do this after cls_embedding prior path is verified.'
            )

        elif self.instance_feat_type == 'cls_guide_grid_feat':
            raise NotImplementedError(
                'cls_guide_grid_feat requires BEV feature gathering by proposal centers. '
                'Do this after cls_embedding prior path is verified.'
            )

        else:
            raise NotImplementedError(
                f'Unknown instance_feat_type: {self.instance_feat_type}'
            )

        # ---------------------------------------------------------
        # Detach LiDAR prior in the first stage.
        # Sparse4D loss should not back-propagate to LiDAR head yet.
        # ---------------------------------------------------------
        if self.training:
            anchors = anchors.detach()
            instance_feats = instance_feats.detach()
            scores = scores.detach()
            labels = labels.detach()

        instances = dict(
            anchors=anchors,
            instance_feats=instance_feats,
            scores=scores,
            labels=labels
        )

        return instances