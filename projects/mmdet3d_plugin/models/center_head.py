import copy
import torch
from mmcv.ops import nms_rotated
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule
from torch import nn
import numpy as np
from skimage.draw import polygon
from mmdet.models.utils.gaussian_target import get_local_maximum
# from mmdet3d.models.layers import circle_nms
# from mmdet3d.structures import xywhr2xyxyr
# from mmdet3d.models.utils import (draw_heatmap_gaussian, gaussian_radius)
from torch.cuda.amp import autocast
# from multitask.registry import MMDET3D_MODELS
# from mmdet3d.models.utils import clip_sigmoid
# from multitask.ops.iou3d_nms.iou3d_nms_utils import nms_gpu
# from mmdet.models.utils import multi_apply
# from mmdet3d.models.task_modules.builder import build_bbox_coder
# from skimage.draw import polygon
# from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.core import (
    circle_nms,
    xywhr2xyxyr,
    draw_heatmap_gaussian,
    gaussian_radius,
)
from mmdet.models.builder import build_loss
# from mmdet.models.utils.gaussian_target import clip_sigmoid

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.builder import HEADS


try:
    from mmdet3d.core.post_processing import nms_bev as nms_gpu
except ImportError:
    from mmdet3d.core.post_processing.box3d_nms import nms_bev as nms_gpu

from mmdet.core import multi_apply

from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead  # noqa: F401
from mmdet3d.core.bbox.coders.centerpoint_bbox_coders import CenterPointBBoxCoder  # noqa: F401

def clip_sigmoid(x, eps=1e-4):
    """Numerically stable sigmoid used in CenterPoint-style heads."""
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


@HEADS.register_module()
class SparseFusionCenterHead(BaseModule):
    """SparseFusionCenterHead for CenterPoint-style LiDAR proposal generation."""

    def __init__(self,
                 in_channels=128,
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=None,
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='none', loss_weight=0.25),
                 loss_auxiliary_seg=None,
                 separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 head_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 near_distance=0,
                 ignore_yaw=None,
                 head_norm=False,
                 init_cfg=None):
        assert init_cfg is None, (
            'To prevent abnormal initialization behavior, '
            'init_cfg is not allowed to be set.'
        )
        super(SparseFusionCenterHead, self).__init__(init_cfg=init_cfg)

        assert tasks is not None and len(tasks) > 0, \
            'tasks should not be None for SparseFusionCenterHead.'
        assert bbox_coder is not None, \
            'bbox_coder should not be None for SparseFusionCenterHead.'

        if common_heads is None:
            common_heads = dict()
        if ignore_yaw is None:
            ignore_yaw = []

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.norm_bbox = norm_bbox
        self.near_distance = near_distance
        self.head_norm = head_norm
        self.fp16_enabled = False

        self.num_classes = [task['num_classes'] for task in tasks]
        self.class_names = [task['class_names'] for task in tasks]

        # Some configs may not provide class_weights.
        self.class_weights = [
            task.get('class_weights', [1.0] * task['num_classes'])
            for task in tasks
        ]

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_auxiliary_seg = (
            build_loss(loss_auxiliary_seg)
            if loss_auxiliary_seg is not None else None
        )

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_anchor_per_locs = [n for n in self.num_classes]

        self.ignore_yaw_cls_id = dict()
        for task_id, task in enumerate(tasks):
            self.ignore_yaw_cls_id[task_id] = []
            for cls_id, class_name in enumerate(task['class_names']):
                if class_name in ignore_yaw:
                    self.ignore_yaw_cls_id[task_id].append(cls_id)

        if share_conv_channel is not None:
            self.shared_conv = ConvModule(
                in_channels,
                share_conv_channel,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                bias=bias
            )
        else:
            share_conv_channel = in_channels

        self.task_heads = nn.ModuleList()

        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))

            if self.loss_auxiliary_seg is not None:
                heads.update(dict(segmap=(num_cls, num_heatmap_convs)))

            cur_separate_head = copy.deepcopy(separate_head)
            cur_separate_head.update(
                in_channels=share_conv_channel,
                head_conv=head_conv_channel,
                heads=heads,
                num_cls=num_cls
            )

            self.task_heads.append(HEADS.build(cur_separate_head))

        self.init_weights()

    def init_weights(self):
        """Initialize heatmap bias."""
        for task in self.task_heads:
            for head in task.heads:
                if head == 'heatmap':
                    task.__getattr__(head)[-1].bias.data.fill_(-2.19)

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[Tensor] | tuple[Tensor] | Tensor): BEV features.

        Returns:
            tuple[dict]: Predictions of each task head.
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]

        ret_dicts = []

        # 如果只有一个 BEV feature，但有多个 task head，则所有 task 共享该 feature
        if len(feats) == 1 and len(self.task_heads) > 1:
            feats = feats * len(self.task_heads)

        assert len(feats) == len(self.task_heads), (
            f'Number of feats ({len(feats)}) should match number of task heads '
            f'({len(self.task_heads)}).'
        )

        for x, task in zip(feats, self.task_heads):
            if hasattr(self, 'shared_conv'):
                x = self.shared_conv(x)
            x = task(x)
            ret_dicts.append(x)

        return tuple(ret_dicts)

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map according to positive indices.

        Args:
            feat (Tensor): Shape [B, H*W, C].
            ind (Tensor): Shape [B, max_obj].
            mask (Tensor, optional): Shape [B, max_obj].

        Returns:
            Tensor: Shape [B, max_obj, C] if mask is None,
                otherwise [num_valid, C].
        """
        dim = feat.size(2)

        ind = ind.long()
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

        feat = feat.gather(1, ind)

        if mask is not None:
            mask = mask.bool()
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)

        return feat

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, file_name=None):
        """Generate training targets for a batch.

        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes] | list[Tensor]):
                GT boxes of each sample.
            gt_labels_3d (list[Tensor]): GT labels of each sample.
            file_name (list | None): Optional file names.

        Returns:
            tuple:
                heatmaps, anno_boxes, inds, cats, masks, segmaps, box_weights
        """
        num_samples = len(gt_bboxes_3d)

        if file_name is None:
            file_name = [None for _ in range(num_samples)]

        heatmaps, anno_boxes, inds, cats, masks, segmaps, box_weights = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            file_name
        )

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

        box_weights = list(map(list, zip(*box_weights)))
        box_weights = [torch.stack(weight, dim=0) for weight in box_weights]

        if self.loss_auxiliary_seg is not None:
            segmaps = list(map(list, zip(*segmaps)))
            segmaps = [torch.stack(seg, dim=0) for seg in segmaps]
        else:
            segmaps = None

        return heatmaps, anno_boxes, inds, cats, masks, segmaps, box_weights

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, file_name=None):
        """Generate training targets for one sample.

        This version is adapted for Sparse4D v3 / old mmdet3d and nuScenes
        official annotations.

        Args:
            gt_bboxes_3d (LiDARInstance3DBoxes | Tensor):
                If LiDARInstance3DBoxes, expected tensor part is:
                    boxes.gravity_center: [x, y, z]
                    boxes.tensor[:, 3:]: [dx, dy, dz, yaw, vx, vy] if velocity exists.
                If Tensor, expected format is:
                    [x, y, z, dx, dy, dz, yaw]
                or:
                    [x, y, z, dx, dy, dz, yaw, vx, vy].
            gt_labels_3d (Tensor): Labels of boxes.

        Returns:
            tuple:
                heatmaps, anno_boxes, inds, cats, masks, segmaps, box_weights
        """
        device = gt_labels_3d.device

        # ---------------------------------------------------------
        # Convert LiDARInstance3DBoxes to Tensor.
        # nuScenes:
        #   [x, y, z, dx, dy, dz, yaw, vx, vy]
        # ---------------------------------------------------------
        if hasattr(gt_bboxes_3d, 'gravity_center') and hasattr(gt_bboxes_3d, 'tensor'):
            gt_bboxes_3d = torch.cat(
                [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
                dim=1
            ).to(device)
        else:
            gt_bboxes_3d = gt_bboxes_3d.to(device)

        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']

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

        # ---------------------------------------------------------
        # nuScenes velocity check.
        # 7 dims: [x, y, z, dx, dy, dz, yaw]
        # 9 dims: [x, y, z, dx, dy, dz, yaw, vx, vy]
        # ---------------------------------------------------------
        has_velocity = gt_bboxes_3d.shape[-1] >= 9

        # CenterHead target:
        #   8  = reg(2) + height(1) + dim(3) + rot(2)
        #   10 = reg(2) + height(1) + dim(3) + rot(2) + vel(2)
        num_box_dim = 10 if has_velocity else 8

        # ---------------------------------------------------------
        # Reorganize GT by tasks.
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
                    # Local class ids start from 1. 0 is background.
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
        box_weights = []

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

            ind = gt_labels_3d.new_zeros((max_objs,), dtype=torch.int64)
            cat = gt_labels_3d.new_zeros((max_objs,), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs,), dtype=torch.uint8)

            box_weight = gt_bboxes_3d.new_ones(
                (max_objs, num_box_dim),
                dtype=torch.float32
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

                length = task_boxes[task_id][k][3]
                width = task_boxes[task_id][k][4]

                length_feat = length / voxel_size[0] / out_size_factor
                width_feat = width / voxel_size[1] / out_size_factor

                if length_feat <= 0 or width_feat <= 0:
                    continue

                x = task_boxes[task_id][k][0]
                y = task_boxes[task_id][k][1]
                z = task_boxes[task_id][k][2]

                if abs(x) <= self.near_distance and abs(y) <= self.near_distance:
                    box_weight[k, :] = box_weight[k, :] * 2.0

                coor_x = (x - pc_range[0]) / voxel_size[0] / out_size_factor
                coor_y = (y - pc_range[1]) / voxel_size[1] / out_size_factor

                center = torch.stack([coor_x, coor_y]).to(torch.float32)
                center_int = center.to(torch.int32)

                if not (
                    0 <= center_int[0] < feature_map_size[0]
                    and 0 <= center_int[1] < feature_map_size[1]
                ):
                    continue

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

                if task_id in self.ignore_yaw_cls_id and cls_id in self.ignore_yaw_cls_id[task_id]:
                    box_weight[k, 6] = 0.0
                    box_weight[k, 7] = 0.0

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

                if self.head_norm:
                    rot = self.norm_heading(rot)

                # -------------------------------------------------
                # Basic target:
                # [offset_x, offset_y, z, dx, dy, dz, sin_yaw, cos_yaw]
                #
                # 注意：
                # 这里沿用原 CenterHead 写法，把 z 转成 box bottom/height 相关形式：
                # z + h * 0.5
                # 如果你的 bbox_coder 使用 gravity center z，则改为 z.unsqueeze(0)。
                # -------------------------------------------------
                anno_box[k, :8] = torch.cat([
                    center - torch.stack([grid_x, grid_y]).to(
                        device=device,
                        dtype=center.dtype
                    ),
                    z.unsqueeze(0) + task_boxes[task_id][k][5].unsqueeze(0) * 0.5,
                    box_dim,
                    torch.sin(rot).unsqueeze(0),
                    torch.cos(rot).unsqueeze(0)
                ])

                # -------------------------------------------------
                # nuScenes velocity target:
                # [vx, vy]
                # -------------------------------------------------
                if has_velocity:
                    vx = task_boxes[task_id][k][7]
                    vy = task_boxes[task_id][k][8]
                    anno_box[k, 8:10] = torch.stack([vx, vy])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            inds.append(ind)
            cats.append(cat)
            masks.append(mask)
            box_weights.append(box_weight)

            if self.loss_auxiliary_seg is not None:
                segmap = torch.tensor(
                    segmap,
                    device=heatmap.device,
                    dtype=heatmap.dtype
                )
                segmaps.append(segmap)

        return heatmaps, anno_boxes, inds, cats, masks, segmaps, box_weights

    @staticmethod
    def norm_heading(heading):
        """Normalize heading angle following the original half-period logic.

        The output range is approximately [-pi/2, pi/2].
        This keeps the original source-code semantics but supports Tensor input.
        """
        pi = heading.new_tensor(torch.pi)

        heading = torch.where(
            heading > pi,
            heading % pi,
            heading
        )

        heading = torch.where(
            heading < -pi,
            heading % (-pi),
            heading
        )

        heading = torch.where(
            heading > pi / 2,
            heading - pi,
            heading
        )

        heading = torch.where(
            heading < -pi / 2,
            heading + pi,
            heading
        )

        return heading

    # @force_fp32(apply_to=('preds_dicts',))
    def loss(self,
            preds_dicts=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            img_metas=None,
            teacher_preds_dicts=None,
            **kwargs):
        """Loss function for SparseFusionCenterHead.

        Adapted for Sparse4D v3 / old mmdet3d / nuScenes style.

        Args:
            preds_dicts (tuple/list[dict]): Output of forward().
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): GT boxes.
            gt_labels_3d (list[Tensor]): GT labels.
            img_metas (list[dict], optional): nuScenes/Sparse4D meta info.
            teacher_preds_dicts (tuple/list[dict], optional): Teacher predictions
                for distillation.

        Returns:
            dict[str, Tensor]: Loss dict.
        """
        assert preds_dicts is not None, 'preds_dicts is None.'
        assert gt_bboxes_3d is not None, 'gt_bboxes_3d is None.'
        assert gt_labels_3d is not None, 'gt_labels_3d is None.'

        if img_metas is None:
            img_metas = kwargs.get('img_metas', None)

        if img_metas is None:
            img_metas = [dict() for _ in range(len(gt_bboxes_3d))]

        file_name = [
            meta.get('sample_idx', meta.get('filename', None))
            for meta in img_metas
        ]

        heatmaps, anno_boxes, inds, cats, masks, segmps, box_weights = self.get_targets(
            gt_bboxes_3d,
            gt_labels_3d,
            file_name
        )

        loss_dict = dict()

        for task_id, preds_dict in enumerate(preds_dicts):
            with autocast(enabled=False):
                preds_dict['heatmap'] = clip_sigmoid(
                    preds_dict['heatmap'].float()
                )

                if teacher_preds_dicts is not None:
                    teacher_heatmap = clip_sigmoid(
                        teacher_preds_dicts[task_id]['heatmap'].float()
                    )
                    num_pos = heatmaps[task_id].eq(1).float().sum().item()
                    loss_heatmap_distillation = self.loss_cls(
                        preds_dict['heatmap'],
                        teacher_heatmap,
                        avg_factor=max(num_pos, 1)
                    )

                num_pos = heatmaps[task_id].eq(1).float().sum().item()
                loss_heatmap = self.loss_cls(
                    preds_dict['heatmap'],
                    heatmaps[task_id],
                    avg_factor=max(num_pos, 1)
                )

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

                target_box = anno_boxes[task_id]
                target_box_weight = box_weights[task_id]

                # -------------------------------------------------
                # Reconstruct bbox prediction.
                # nuScenes target:
                #   without vel: [reg(2), height(1), dim(3), rot(2)] = 8
                #   with vel:    [reg(2), height(1), dim(3), rot(2), vel(2)] = 10
                # -------------------------------------------------
                attr_name = ['reg', 'height', 'dim', 'rot']

                if 'vel' in preds_dict and target_box.shape[-1] >= 10:
                    attr_name.append('vel')
                else:
                    target_box = target_box[..., :8]
                    target_box_weight = target_box_weight[..., :8]

                if 'height' not in preds_dict:
                    # Remove height dimension:
                    # [reg_x, reg_y, z, dx, dy, dz, sin, cos] ->
                    # [reg_x, reg_y, dx, dy, sin, cos]
                    target_box = target_box[..., [0, 1, 3, 4, 6, 7]]
                    target_box_weight = target_box_weight[..., [0, 1, 3, 4, 6, 7]]
                    if 'height' in attr_name:
                        attr_name.remove('height')

                preds_dict['anno_box'] = torch.cat(
                    [preds_dict[attr] for attr in attr_name],
                    dim=1
                )

                ind = inds[task_id]
                num = masks[task_id].float().sum()

                pred = preds_dict['anno_box'].permute(0, 2, 3, 1).contiguous()
                pred = pred.view(pred.size(0), -1, pred.size(3))
                pred = self._gather_feat(pred, ind)

                mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
                isnotnan = (~torch.isnan(target_box)).float()
                mask *= isnotnan

                code_weights_cfg = self.train_cfg.get('code_weights', None)

                if code_weights_cfg is None:
                    bbox_weights = mask * target_box_weight
                else:
                    code_weights = mask.new_tensor(code_weights_cfg)

                    if code_weights.dim() == 1:
                        code_weights = code_weights.view(1, 1, -1)
                        code_weights = code_weights[..., :target_box.shape[-1]]
                        code_weights = code_weights.expand_as(target_box)

                    elif code_weights.dim() == 2:
                        code_weights = code_weights[:, :target_box.shape[-1]]
                        code_weights = code_weights.unsqueeze(0).expand(
                            len(gt_bboxes_3d), -1, -1
                        )
                        code_weights = code_weights.gather(
                            1,
                            cats[task_id].unsqueeze(2).expand(
                                -1, -1, code_weights.shape[-1]
                            )
                        )

                    elif code_weights.dim() == 3:
                        code_weights = code_weights[task_id]
                        code_weights = code_weights[:, :target_box.shape[-1]]
                        code_weights = code_weights.unsqueeze(0).expand(
                            len(gt_bboxes_3d), -1, -1
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

                    bbox_weights = mask * code_weights * target_box_weight

                loss_bbox = self.loss_bbox(
                    pred,
                    target_box,
                    bbox_weights,
                    avg_factor=(num + 1e-4)
                )

                loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
                loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox

                if self.loss_auxiliary_seg is not None:
                    loss_dict[f'task{task_id}.loss_segmap'] = loss_segmap * 4

                if teacher_preds_dicts is not None:
                    teacher_boxs = torch.cat(
                        [teacher_preds_dicts[task_id][attr] for attr in attr_name],
                        dim=1
                    )
                    teacher_boxs = teacher_boxs.permute(0, 2, 3, 1).contiguous()
                    teacher_boxs = teacher_boxs.view(
                        teacher_boxs.size(0),
                        -1,
                        teacher_boxs.size(3)
                    )
                    teacher_boxs = self._gather_feat(teacher_boxs, ind)

                    loss_bbox_distillation = self.loss_bbox(
                        pred,
                        teacher_boxs,
                        bbox_weights,
                        avg_factor=(num + 1e-4)
                    )

                    loss_dict[f'task{task_id}.loss_heatmap_distillation'] = (
                        loss_heatmap_distillation
                    )
                    loss_dict[f'task{task_id}.loss_bbox_distillation'] = (
                        loss_bbox_distillation
                    )

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Adapted for Sparse4D v3 / old mmdet3d / nuScenes style.

        Args:
            preds_dicts (tuple/list[dict]): Predictions of task heads.
            img_metas (list[dict], optional): Meta information.
            img: Unused.
            rescale (bool): Unused.

        Returns:
            list[list]:
                Each item is [LiDARInstance3DBoxes, scores_3d, labels_3d].
        """
        rets = []

        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict['heatmap'].shape[0]

            batch_heatmap = preds_dict['heatmap'].sigmoid()

            if self.test_cfg.get('nms_type', None) == 'local_maximum_heatmap':
                batch_heatmap = get_local_maximum(batch_heatmap)

            batch_reg = preds_dict['reg']

            if 'height' in preds_dict:
                batch_hei = preds_dict['height']
            else:
                batch_hei = torch.zeros_like(batch_reg[:, 0:1])

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict['dim'])
            else:
                batch_dim = preds_dict['dim']

            # Some heads only predict length/width, append a dummy height.
            if batch_dim.shape[1] == 2:
                batch_dim = torch.cat(
                    [batch_dim, torch.ones_like(batch_hei)],
                    dim=1
                )

            batch_rots = preds_dict['rot'][:, 0:1]
            batch_rotc = preds_dict['rot'][:, 1:2]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
            else:
                # Keep the decode API compatible.
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

            nms_type = self.test_cfg.get('nms_type', None)
            assert nms_type in [
                'circle',
                'rotate',
                'nms_bev',
                'local_maximum_heatmap',
                None
            ], f'Unsupported nms_type: {nms_type}'

            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]

            if nms_type == 'circle':
                ret_task = []

                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']

                    if boxes3d.numel() == 0:
                        ret_task.append(dict(
                            bboxes=boxes3d,
                            scores=scores,
                            labels=labels
                        ))
                        continue

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

                    ret_task.append(dict(
                        bboxes=boxes3d[keep],
                        scores=scores[keep],
                        labels=labels[keep]
                    ))

                rets.append(ret_task)

            elif nms_type == 'nms_bev':
                ret_task = []

                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']

                    if boxes3d.numel() == 0:
                        ret_task.append(dict(
                            bboxes=boxes3d,
                            scores=scores,
                            labels=labels
                        ))
                        continue

                    boxes = boxes3d[:, [0, 1, 3, 4, 6]]

                    order = scores.sort(0, descending=True)[1]
                    boxes = boxes[order]
                    boxes3d = boxes3d[order]
                    scores = scores[order]
                    labels = labels[order]

                    keep = nms_rotated(
                        boxes,
                        scores,
                        self.test_cfg['thresh']
                    )[1]

                    if keep is None:
                        keep = torch.empty(
                            0,
                            dtype=torch.long,
                            device=boxes.device
                        )

                    post_max_size = self.test_cfg.get('post_max_size', None)
                    if post_max_size is not None:
                        keep = keep[:post_max_size]

                    ret_task.append(dict(
                        bboxes=boxes3d[keep],
                        scores=scores[keep],
                        labels=labels[keep]
                    ))

                rets.append(ret_task)

            elif nms_type == 'rotate':
                rets.append(
                    self.get_task_detections(
                        num_class_with_bg,
                        batch_cls_preds,
                        batch_reg_preds,
                        batch_cls_labels,
                        img_metas
                    )
                )

            elif nms_type == 'local_maximum_heatmap' or nms_type is None:
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
                raise NotImplementedError(f'Unknown nms type: {nms_type}')

        # ---------------------------------------------------------
        # Merge task results.
        # Old mmdet3d format:
        #   [LiDARInstance3DBoxes, scores_3d, labels_3d]
        # ---------------------------------------------------------
        num_samples = len(rets[0])
        ret_list = []

        for i in range(num_samples):
            bboxes_list = []
            scores_list = []
            labels_list = []

            label_offset = 0

            for task_id, num_class in enumerate(self.num_classes):
                cur_bboxes = rets[task_id][i]['bboxes']
                cur_scores = rets[task_id][i]['scores']
                cur_labels = rets[task_id][i]['labels'].int() + label_offset

                bboxes_list.append(cur_bboxes)
                scores_list.append(cur_scores)
                labels_list.append(cur_labels)

                label_offset += num_class

            if len(bboxes_list) > 0:
                bboxes = torch.cat(bboxes_list, dim=0)
                scores = torch.cat(scores_list, dim=0)
                labels = torch.cat(labels_list, dim=0)
            else:
                device = preds_dicts[0]['heatmap'].device
                dtype = preds_dicts[0]['heatmap'].dtype
                bboxes = torch.zeros(
                    (0, self.bbox_coder.code_size),
                    device=device,
                    dtype=dtype
                )
                scores = torch.zeros((0,), device=device, dtype=dtype)
                labels = torch.zeros((0,), device=device, dtype=torch.long)

            if bboxes.numel() > 0:
                # Convert z from gravity center to bottom center.
                # Keep this if bbox_coder.decode outputs gravity-center z.
                bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            bboxes = LiDARInstance3DBoxes(
                bboxes,
                self.bbox_coder.code_size
            )

            ret_list.append([bboxes, scores, labels])

        return ret_list

    def heatmap_show(self, heatmaps, save_name):
        """Save gaussian heatmap visualization for debugging."""
        if len(heatmaps) == 0:
            return

        final_heatmap = torch.zeros_like(heatmaps[0].detach().cpu())

        for heatmap in heatmaps:
            final_heatmap = final_heatmap + heatmap.detach().cpu()

        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        image = to_pil(final_heatmap)
        image.save(save_name)


    def draw_segmap(self, segmap, x, y, l, w, rot):
        """Draw rotated box mask on BEV segmap.

        Args:
            segmap (np.ndarray): BEV segmentation map, shape [H, W].
            x, y: Box center on feature map.
            l, w: Box size on feature map.
            rot: Box yaw in radians.
        """
        x = float(x)
        y = float(y)
        l = float(l) + 2.0
        w = float(w) + 2.0
        rot = float(rot)

        x1, y1, x2, y2, x3, y3, x4, y4 = self.cal_vertex(x, y, l, w, rot)

        xs = np.array([x1, x2, x3, x4], dtype=np.float32)
        ys = np.array([y1, y2, y3, y4], dtype=np.float32)

        rr, cc = polygon(ys, xs, shape=segmap.shape)
        segmap[rr, cc] = 1.0


    @staticmethod
    def cal_vertex(x, y, l, w, rot):
        """Calculate four vertices of a rotated box on BEV feature map."""
        x = float(x)
        y = float(y)
        l = float(l)
        w = float(w)
        rot = float(rot)

        cos_r = np.cos(rot)
        sin_r = np.sin(rot)

        x1 = int(x + l / 2 * cos_r + w / 2 * sin_r)
        y1 = int(y + l / 2 * sin_r - w / 2 * cos_r)

        x2 = int(x - l / 2 * cos_r + w / 2 * sin_r)
        y2 = int(y - l / 2 * sin_r - w / 2 * cos_r)

        x3 = int(x - l / 2 * cos_r - w / 2 * sin_r)
        y3 = int(y - l / 2 * sin_r + w / 2 * cos_r)

        x4 = int(x + l / 2 * cos_r - w / 2 * sin_r)
        y4 = int(y + l / 2 * sin_r + w / 2 * cos_r)

        return x1, y1, x2, y2, x3, y3, x4, y4

    def anno_show(self, task_boxes, save_name):
        """Visualize annotation boxes on BEV heatmap for debugging."""
        import cv2

        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        # Use the first task's out_size_factor as default visualization stride.
        # If each task has a different stride, each box will use its own stride below.
        default_out_size_factor = self.train_cfg['out_size_factor'][0]
        feature_map_size = grid_size[:2] // default_out_size_factor

        image = np.zeros(
            (int(feature_map_size[1]), int(feature_map_size[0])),
            dtype=np.uint8
        )
        image = np.expand_dims(image, -1).repeat(3, axis=-1)

        heatmap_box = []

        for task_id, _ in enumerate(self.task_heads):
            if task_id >= len(task_boxes):
                continue

            boxes = task_boxes[task_id]

            if boxes is None or boxes.shape[0] == 0:
                continue

            out_size_factor = self.train_cfg['out_size_factor'][task_id]

            boxes = boxes.detach().cpu()
            pc_range_cpu = pc_range.cpu()
            voxel_size_cpu = voxel_size.cpu()

            num_objs = boxes.shape[0]

            for k in range(num_objs):
                x, y = boxes[k][0], boxes[k][1]
                length, width = boxes[k][3], boxes[k][4]
                rot = boxes[k][6]

                coor_x = (x - pc_range_cpu[0]) / voxel_size_cpu[0] / out_size_factor
                coor_y = (y - pc_range_cpu[1]) / voxel_size_cpu[1] / out_size_factor

                length_feat = length / voxel_size_cpu[0] / out_size_factor
                width_feat = width / voxel_size_cpu[1] / out_size_factor

                heatmap_box.append([
                    float(coor_x),
                    float(coor_y),
                    float(length_feat),
                    float(width_feat),
                    float(rot)
                ])

        if len(heatmap_box) == 0:
            cv2.imwrite(save_name, image)
            return

        heatmap_box = np.array(heatmap_box, dtype=np.float32)

        cx = heatmap_box[:, 0]
        cy = heatmap_box[:, 1]
        length = heatmap_box[:, 2]
        width = heatmap_box[:, 3]
        angle = heatmap_box[:, 4]

        height, width_img = image.shape[:2]

        for i in range(len(cx)):
            x1, y1, x2, y2, x3, y3, x4, y4 = self.cal_vertex(
                cx[i], cy[i], length[i], width[i], angle[i]
            )

            # Optional: skip boxes completely outside image.
            if (
                max(x1, x2, x3, x4) < 0 or min(x1, x2, x3, x4) >= width_img or
                max(y1, y2, y3, y4) < 0 or min(y1, y2, y3, y4) >= height
            ):
                continue

            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.line(image, (x2, y2), (x3, y3), (255, 0, 0), 1)
            cv2.line(image, (x3, y3), (x4, y4), (255, 0, 0), 1)
            cv2.line(image, (x4, y4), (x1, y1), (255, 0, 0), 1)
            cv2.circle(image, (int(cx[i]), int(cy[i])), 1, (0, 255, 0), -1)

        cv2.imwrite(save_name, image)

    def get_task_detections(self,
                            num_class_with_bg,
                            batch_cls_preds,
                            batch_reg_preds,
                            batch_cls_labels,
                            img_metas=None):
        """Apply rotated NMS for each sample of one task.

        This version is adapted for Sparse4D v3 / old mmdet3d / nuScenes style.

        Args:
            num_class_with_bg (int): Number of classes for current task.
            batch_cls_preds (list[Tensor]): Scores of each sample.
            batch_reg_preds (list[Tensor]): Boxes of each sample, shape [N, box_dim].
            batch_cls_labels (list[Tensor]): Labels of each sample.
            img_metas (list[dict], optional): Unused, kept for interface compatibility.

        Returns:
            list[dict]: Each dict contains bboxes, scores, labels.
        """
        predictions_dicts = []

        post_center_range = self.test_cfg.get('post_center_limit_range', None)
        if post_center_range is not None and len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device
            )
        else:
            post_center_range = None

        score_threshold = self.test_cfg.get('score_threshold', 0.0)
        nms_thr = self.test_cfg.get('nms_thr', self.test_cfg.get('thresh', None))
        pre_max_size = self.test_cfg.get('pre_max_size', None)
        post_max_size = self.test_cfg.get('post_max_size', None)

        assert nms_thr is not None, \
            'test_cfg should contain nms_thr or thresh for rotated NMS.'

        for box_preds, cls_preds, cls_labels in zip(
                batch_reg_preds, batch_cls_preds, batch_cls_labels):

            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long
                )
            else:
                top_scores = cls_preds.squeeze(-1)
                top_labels = cls_labels.long()

            if score_threshold > 0.0:
                score_mask = top_scores >= score_threshold
                box_preds = box_preds[score_mask]
                top_scores = top_scores[score_mask]
                top_labels = top_labels[score_mask]

            if top_scores.numel() == 0:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dicts.append(dict(
                    bboxes=torch.zeros(
                        (0, self.bbox_coder.code_size),
                        dtype=dtype,
                        device=device
                    ),
                    scores=torch.zeros((0,), dtype=dtype, device=device),
                    labels=torch.zeros((0,), dtype=torch.long, device=device)
                ))
                continue

            order = top_scores.sort(descending=True)[1]

            if pre_max_size is not None:
                order = order[:pre_max_size]

            box_preds = box_preds[order]
            top_scores = top_scores[order]
            top_labels = top_labels[order]

            # Rotated NMS boxes: [cx, cy, w, h, angle].
            # Here box_preds follows CenterPoint decoded format:
            # [x, y, z, dx, dy, dz, yaw, vx, vy] or similar.
            boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]

            keep = nms_rotated(
                boxes_for_nms,
                top_scores,
                nms_thr
            )[1]

            if keep is None:
                keep = torch.empty(
                    0,
                    dtype=torch.long,
                    device=box_preds.device
                )

            if post_max_size is not None:
                keep = keep[:post_max_size]

            final_box_preds = box_preds[keep]
            final_scores = top_scores[keep]
            final_labels = top_labels[keep]

            if post_center_range is not None and final_box_preds.numel() > 0:
                range_mask = (
                    final_box_preds[:, :3] >= post_center_range[:3]
                ).all(dim=1)
                range_mask &= (
                    final_box_preds[:, :3] <= post_center_range[3:]
                ).all(dim=1)

                final_box_preds = final_box_preds[range_mask]
                final_scores = final_scores[range_mask]
                final_labels = final_labels[range_mask]

            predictions_dicts.append(dict(
                bboxes=final_box_preds,
                scores=final_scores,
                labels=final_labels
            ))

        return predictions_dicts