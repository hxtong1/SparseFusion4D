# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature

import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from mmdet3d.models.builder import (
    build_voxel_encoder,
    build_middle_encoder,
    build_backbone as build_3d_backbone,
    build_neck as build_3d_neck,
)
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.mmcv_custom.ops.voxel import SPConvVoxelization
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseFusion4D"]


@DETECTORS.register_module()
class SparseFusion4D(MVXTwoStageDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,

        # ===== LiDAR branch from CMT / CenterPoint =====
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_backbone=None,
        pts_neck=None,
    ):
        super(SparseFusion4D, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            img_backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)

        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        else:
            self.img_neck = None
        self.head = build_head(head)

        # ===== LiDAR branch =====
        self.pts_voxel_layer = None
        self.pts_voxel_encoder = None
        self.pts_middle_encoder = None
        self.pts_backbone = None
        self.pts_neck = None

        if pts_voxel_layer is not None:
            # 你已经从 projects.mmdet3d_plugin import SPConvVoxelization
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_layer)

        if pts_voxel_encoder is not None:
            self.pts_voxel_encoder = build_voxel_encoder(
                pts_voxel_encoder
            )

        if pts_middle_encoder is not None:
            self.pts_middle_encoder = build_middle_encoder(
                pts_middle_encoder
            )

        if pts_backbone is not None:
            self.pts_backbone = build_3d_backbone(
                pts_backbone
            )

        if pts_neck is not None:
            self.pts_neck = build_3d_neck(
                pts_neck
            )

        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func

        # ===== optional depth branch =====
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None

        # ===== grid mask ===== 
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    def _format_points(self, points):
        """Convert collated LiDARPoints into list[Tensor]."""
        if points is None:
            return None

        # Debug / non-parallel case: DataContainer may still exist.
        if hasattr(points, "data"):
            points = points.data

        # Common collated forms:
        # [[LiDARPoints(...)]]
        # [LiDARPoints(...)]
        if isinstance(points, (list, tuple)) and len(points) == 1:
            if isinstance(points[0], (list, tuple)):
                points = points[0]

        formatted_points = []
        for p in points:
            if hasattr(p, "tensor"):
                p = p.tensor
            formatted_points.append(p)

        return formatted_points

    @property
    def with_pts_bbox(self):
        return (
            hasattr(self, "pts_voxel_layer")
            and self.pts_voxel_layer is not None
            and hasattr(self, "pts_voxel_encoder")
            and self.pts_voxel_encoder is not None
            and hasattr(self, "pts_middle_encoder")
            and self.pts_middle_encoder is not None
            and hasattr(self, "pts_backbone")
            and self.pts_backbone is not None
        )

    @property
    def with_pts_neck(self):
        return hasattr(self, "pts_neck") and self.pts_neck is not None

    @force_fp32(apply_to=("pts", "img_feats"))
    def extract_pts_feat(self, pts, img_feats=None, img_metas=None):
        """Extract BEV features from point clouds."""
        if not self.with_pts_bbox:
            return None

        if pts is None:
            return None

        pts = self._format_points(pts)

        if pts is None or len(pts) == 0:
            return None

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(
            voxels,
            num_points,
            coors,
        )

        batch_size = int(coors[-1, 0].item()) + 1

        x = self.pts_middle_encoder(
            voxel_features,
            coors,
            batch_size,
        )

        x = self.pts_backbone(x)

        if self.with_pts_neck:
            x = self.pts_neck(x)

        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points.

        Args:
            points (list[torch.Tensor]): points of each sample.

        Returns:
            voxels, num_points, coors_batch
        """
        voxels, coors, num_points = [], [], []

        for res in points:
            if hasattr(res, "tensor"):
                res = res.tensor

            # Make sure points are on the same device as voxel layer.
            if not res.is_cuda:
                res = res.cuda()

            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)

            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)

        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)

        coors_batch = torch.cat(coors_batch, dim=0)

        return voxels, num_points, coors_batch

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_img_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    def extract_feat(
        self,
        img=None,
        points=None,
        return_depth=False,
        metas=None,
        img_metas=None,
    ):
        """Extract image and point cloud features.

        Returns:
            img_feats: multi-view image features.
            pts_feats: LiDAR BEV features.
            depths: optional depth predictions.
        """
        if return_depth:
            img_feats, depths = self.extract_img_feat(
                img,
                return_depth=True,
                metas=metas,
            )
        else:
            img_feats = self.extract_img_feat(
                img,
                return_depth=False,
                metas=metas,
            )
            depths = None

        pts_feats = self.extract_pts_feat(
            points,
            img_feats=img_feats,
            img_metas=img_metas,
        )

        if return_depth:
            return img_feats, pts_feats, depths

        return img_feats, pts_feats

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        # 1. 从 data 中取出点云输入
        points = data.get("points", None)

        # 2. 同时提取 image feature、lidar feature 和 depth
        feature_maps, pts_feats, depths = self.extract_feat(
            img=img,
            points=points,
            return_depth=True,
            metas=data,
            img_metas=data.get("img_metas", None),
        )

        # 3. 将 LiDAR BEV feature 放入 data，供 head / fusion module 使用
        data["pts_feats"] = pts_feats
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        points = data.get("points", None)

        feature_maps, pts_feats = self.extract_feat(
            img=img,
            points=points,
            return_depth=False,
            metas=data,
            img_metas=data.get("img_metas", None),
        )

        data["pts_feats"] = pts_feats

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs)
        output = [{"img_bbox": result} for result in results]
        return output

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
