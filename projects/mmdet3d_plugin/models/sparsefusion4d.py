# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature

import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg

from mmdet.models import HEADS
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)

from mmdet3d.models.builder import HEADS as MMDET3D_HEADS
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
        
        # ===== LiDAR prior query branch =====
        lidar_prior_head=None,
        use_lidar_prior=False,
        detach_lidar_prior=True,
        lidar_prior_loss_weight=1.0,
        lidar_prior_warmup_iters=500,
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

        if getattr(self, "freeze_sparse4d_lidar_stage1", False):
            self.freeze_sparse4d_for_lidar_stage1()
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

        self.use_lidar_prior = use_lidar_prior
        self.detach_lidar_prior = detach_lidar_prior
        self.lidar_prior_loss_weight = lidar_prior_loss_weight

        self.lidar_prior_head = (
            build_from_cfg(lidar_prior_head, MMDET3D_HEADS)
            if lidar_prior_head is not None else None
        )

        if self.use_lidar_prior:
            assert self.lidar_prior_head is not None, \
                "use_lidar_prior=True requires lidar_prior_head."

        if self.detach_lidar_prior and self.lidar_prior_head is not None:
            if hasattr(self.lidar_prior_head, "cls_embedding"):
                for p in self.lidar_prior_head.cls_embedding.parameters():
                    p.requires_grad = False

        self.lidar_prior_warmup_iters = lidar_prior_warmup_iters
        self.train_iter = 0

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

    def detach_prior_info(self, prior_info):
        """Detach non-feature prior fields.

        This function fixes DDP unused-parameter errors without using dummy loss.

        For trainable prior feature generators:
            cls_embedding / grid_feat_proj / cls_guide_fusion / score_mlp

        instance_feats must NOT be detached, otherwise these parameters will not
        receive gradients from Sparse4D loss.

        Non-feature fields are detached to avoid Sparse4D loss flowing back into
        CenterHead decode / anchors / scores.
        """
        if prior_info is None:
            return None

        out = {}

        # Default True: train prior instance feature generator through Sparse4D loss.
        train_prior_instance_feats = getattr(
            self,
            "train_prior_instance_feats",
            True,
        )

        for k, v in prior_info.items():
            if not torch.is_tensor(v):
                out[k] = v
                continue

            if k == "instance_feats" and train_prior_instance_feats:
                out[k] = v
            else:
                out[k] = v.detach()

        return out

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

    # @force_fp32(apply_to=("img_feats"))
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

        # Keep the LiDAR branch in FP32. Under fp16 training,
        # SparseEncoder may output HalfTensor while SECOND/SECONDFPN
        # weights remain FloatTensor.
        voxel_features = voxel_features.float()

        batch_size = int(coors[-1, 0].item()) + 1

        x = self.pts_middle_encoder(
            voxel_features,
            coors,
            batch_size,
        )

        x = x.float()
        x = self.pts_backbone(x)

        # SECOND returns multi-scale features as tuple/list.
        # Convert every feature map to fp32 before SECONDFPN.
        if isinstance(x, (list, tuple)):
            x = [feat.float() for feat in x]
        else:
            x = x.float()

        if self.with_pts_neck:
            x = self.pts_neck(x)

        # SECONDFPN usually returns tuple/list as well.
        if isinstance(x, (list, tuple)):
            x = [feat.float() for feat in x]
        else:
            x = x.float()

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

        pts_feats = None
        if self.use_lidar_prior:
            pts_feats = self.extract_pts_feat(
                points,
                img_feats=img_feats,
                img_metas=img_metas,
            )

        if return_depth:
            return img_feats, pts_feats, depths

        return img_feats, pts_feats

    def load_instance_info(
        self,
        bs: int,
        device: torch.device,
        lidar_instance_prior_info: dict = None,
        cam_instance_prior_info: dict = None,
        metainfo: dict = None,
    ) -> dict:
        """Pack camera / LiDAR prior info into industrial-style instance_info.

        This function only packs prior_* fields.
        LAQ queries are still generated by self.head.instance_bank.get().
        """
        instance_info = dict()
        instance_info["num_prior"] = 0
        instance_info["num_cam_prior"] = 0
        instance_info["num_lidar_prior"] = 0

        anchor_dims = getattr(self.head.instance_bank, "anchor_dims", None)

        def _append_prior(prior_info: dict, prior_type: str):
            nonlocal instance_info

            if prior_info is None:
                return

            assert "anchors" in prior_info
            assert "instance_feats" in prior_info

            for k, v in prior_info.items():
                if k not in ["anchors", "instance_feats"]:
                    continue
                if not torch.is_tensor(v):
                    continue

                v = v.to(device)

                if k == "anchors" and anchor_dims is not None:
                    if v.shape[-1] < anchor_dims:
                        pad_shape = list(v.shape)
                        pad_shape[-1] = anchor_dims - v.shape[-1]
                        v = torch.cat([v, v.new_zeros(*pad_shape)], dim=-1)
                    elif v.shape[-1] > anchor_dims:
                        v = v[..., :anchor_dims]

                target_key = f"prior_{k}"
                if target_key in instance_info:
                    instance_info[target_key] = torch.cat(
                        [instance_info[target_key], v], dim=1
                    )
                else:
                    instance_info[target_key] = v

            for k in ["scores", "labels"]:
                if k not in prior_info:
                    continue
                v = prior_info[k]
                if not torch.is_tensor(v):
                    continue

                v = v.to(device)
                target_key = f"prior_{k}"
                if target_key in instance_info:
                    instance_info[target_key] = torch.cat(
                        [instance_info[target_key], v], dim=1
                    )
                else:
                    instance_info[target_key] = v

            instance_info["num_prior"] = instance_info["prior_anchors"].shape[1]

            if prior_type == "cam":
                instance_info["num_cam_prior"] = prior_info["anchors"].shape[1]
            elif prior_type == "lidar":
                instance_info["num_lidar_prior"] = prior_info["anchors"].shape[1]

        _append_prior(cam_instance_prior_info, "cam")
        _append_prior(lidar_instance_prior_info, "lidar")

        if instance_info["num_prior"] == 0:
            return None

        return instance_info

    @force_fp32(apply_to=("img",))
    def forward(self, img, points, **data):
        if self.training:
            return self.forward_train(img, points, **data)
        else:
            return self.forward_test(img, points, **data)

    def forward_train(self, img, points, **data):
        feature_maps, pts_feats, depths = self.extract_feat(
            img=img,
            points=points,
            return_depth=True,
            metas=data,
            img_metas=data.get("img_metas", None),
        )

        loss_dict = {}
        instance_info = None

        if self.use_lidar_prior:
            assert pts_feats is not None, \
                "use_lidar_prior=True requires valid pts_feats."

            lidar_preds = self.lidar_prior_head(pts_feats)

            # 1. LiDAR CenterHead independent loss.
            lidar_loss_dict = self.lidar_prior_head.loss_by_feat(
                preds_dicts=lidar_preds,
                gt_bboxes_3d=data["gt_bboxes_3d"],
                gt_labels_3d=data["gt_labels_3d"],
                img_metas=data.get("img_metas", None),
            )

            for k, v in lidar_loss_dict.items():
                loss_dict[f"lidar_prior.{k}"] = (
                    v * self.lidar_prior_loss_weight
                )

            self.train_iter += 1
            data["cur_iter"] = self.train_iter

            # 2. Warmup: before warmup, only train LiDAR prior head.
            if self.train_iter >= self.lidar_prior_warmup_iters:
                lidar_instance_prior_info = self.lidar_prior_head.decode_lidar(
                    lidar_preds,
                    feats=pts_feats,
                    metainfo=data,
                )

                if self.detach_lidar_prior:
                    lidar_instance_prior_info = self.detach_prior_info(
                        lidar_instance_prior_info
                    )

                instance_info = self.load_instance_info(
                    bs=img.shape[0],
                    device=feature_maps[0].device,
                    lidar_instance_prior_info=lidar_instance_prior_info,
                    cam_instance_prior_info=None,
                    metainfo=data,
                )

        # Important:
        # Use industrial-style instance_info, not lidar_prior_info.
        data["instance_info"] = instance_info
        data["lidar_prior_info"] = None

        # Stage-1: no LiDAR DFFA feature fusion.
        data["pts_feats"] = None

        model_outs = self.head(feature_maps, data)
        sparse4d_loss_dict = self.head.loss(model_outs, data)
        loss_dict.update(sparse4d_loss_dict)

        if depths is not None and "gt_depth" in data:
            loss_dict["loss_dense_depth"] = self.depth_branch.loss(
                depths,
                data["gt_depth"],
            )

        return loss_dict


    def forward_test(self, img, points=None, **data):
        feature_maps, pts_feats = self.extract_feat(
            img=img,
            points=points,
            return_depth=False,
            metas=data,
            img_metas=data.get("img_metas", None),
        )

        instance_info = None

        if self.use_lidar_prior:
            assert pts_feats is not None, \
                "use_lidar_prior=True requires valid pts_feats in forward_test."

            lidar_preds = self.lidar_prior_head(pts_feats)

            lidar_instance_prior_info = self.lidar_prior_head.decode_lidar(
                lidar_preds,
                feats=pts_feats,
                metainfo=data,
            )

            lidar_instance_prior_info = self.detach_prior_info(
                lidar_instance_prior_info
            )

            instance_info = self.load_instance_info(
                bs=img.shape[0],
                device=feature_maps[0].device,
                lidar_instance_prior_info=lidar_instance_prior_info,
                cam_instance_prior_info=None,
                metainfo=data,
            )

        data["instance_info"] = instance_info
        data["lidar_prior_info"] = None
        data["pts_feats"] = None

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
