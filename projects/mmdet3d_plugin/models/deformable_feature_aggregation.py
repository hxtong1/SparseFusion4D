import numpy as np

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)

from projects.mmdet3d_plugin.core.box3d import (
    X, Y, Z, W, L, H,
    SIN_YAW, COS_YAW,
    VX, VY, VZ,
    CNS, YNS,
    YAW,
)
try:
    from ..ops import deformable_aggregation_function as DAF
except:
    DAF = None

__all__ = [
    "DeformableFeatureFusionAggregation",
    "SparseBEVKptGenerator",
    "AsymmetricFFN",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

@PLUGIN_LAYERS.register_module()
class SparseBEVKptGenerator(BaseModule):
    """Generate BEV keypoints from Sparse4D anchors.

    This module is adapted from the industrial BevKptGenerator idea:
    fixed BEV sample points + learnable BEV offsets around each anchor.

    Args:
        embed_dims: query feature dim.
        num_learnable_points: number of learnable BEV offsets.
        fix_sample_scale: fixed sample offsets normalized by box size.
            Example:
                [[0.0, 0.0],
                 [0.5, 0.0],
                 [-0.5, 0.0],
                 [0.0, 0.5],
                 [0.0, -0.5]]
        add_anchor_embeds: whether to add anchor embedding before predicting offsets.
        detach_anchor: whether to detach anchors for keypoint generation.
    """

    def __init__(
        self,
        embed_dims=256,
        num_learnable_points=4,
        fix_sample_scale=None,
        add_anchor_embeds=False,
        detach_anchor=False,
    ):
        super(SparseBEVKptGenerator, self).__init__()

        self.embed_dims = embed_dims
        self.num_learnable_points = num_learnable_points
        self.add_anchor_embeds = add_anchor_embeds
        self.detach_anchor = detach_anchor

        if fix_sample_scale is None:
            fix_sample_scale = [
                [0.0, 0.0],
                [0.5, 0.0],
                [-0.5, 0.0],
                [0.0, 0.5],
                [0.0, -0.5],
            ]

        self.fix_sample_scale = nn.Parameter(
            torch.tensor(fix_sample_scale, dtype=torch.float32),
            requires_grad=False,
        )

        self.num_fix_points = len(fix_sample_scale)
        self.num_pts = self.num_fix_points + self.num_learnable_points

        if self.num_learnable_points > 0:
            self.sample_scale_fc = Linear(
                self.embed_dims,
                self.num_learnable_points * 2,
            )

    def init_weights(self):
        if self.num_learnable_points > 0:
            xavier_init(self.sample_scale_fc, distribution="uniform", bias=0.0)

    def forward(self, anchors, instance_feats, anchor_embeds=None):
        """Generate BEV keypoints.

        Args:
            anchors: [B, N, >=8]
            instance_feats: [B, N, C]
            anchor_embeds: [B, N, C]

        Returns:
            kpts: [B, N, P, 3], in LiDAR coordinates.
                  z is copied from anchor center for interface compatibility.
        """
        if self.detach_anchor:
            anchors = anchors.detach()

        B, N = anchors.shape[:2]

        # Anchor size is stored as log(l/w/h)
        sizes = anchors[..., [L, W]].exp().view(B, N, 1, 2)

        fixed_offsets = self.fix_sample_scale.to(
            device=anchors.device,
            dtype=anchors.dtype,
        )
        fixed_offsets = fixed_offsets.view(1, 1, self.num_fix_points, 2)
        fixed_offsets = fixed_offsets * sizes

        offsets = fixed_offsets

        if self.num_learnable_points > 0:
            feat = instance_feats
            if self.add_anchor_embeds:
                assert anchor_embeds is not None
                feat = feat + anchor_embeds

            learnable_offsets = self.sample_scale_fc(feat)
            learnable_offsets = learnable_offsets.view(
                B,
                N,
                self.num_learnable_points,
                2,
            )
            learnable_offsets = learnable_offsets.sigmoid() - 0.5
            learnable_offsets = learnable_offsets * sizes

            offsets = torch.cat([offsets, learnable_offsets], dim=2)

        # Rotate BEV offsets by anchor yaw
        cos_yaw = anchors[..., COS_YAW]
        sin_yaw = anchors[..., SIN_YAW]

        ox = offsets[..., 0]
        oy = offsets[..., 1]

        rot_x = ox * cos_yaw.unsqueeze(-1) - oy * sin_yaw.unsqueeze(-1)
        rot_y = ox * sin_yaw.unsqueeze(-1) + oy * cos_yaw.unsqueeze(-1)

        cx = anchors[..., X].unsqueeze(-1)
        cy = anchors[..., Y].unsqueeze(-1)
        cz = anchors[..., Z].unsqueeze(-1).expand_as(rot_x)

        kpt_x = cx + rot_x
        kpt_y = cy + rot_y

        kpts = torch.stack([kpt_x, kpt_y, cz], dim=-1)
        return kpts


@ATTENTION.register_module()
class DeformableFeatureFusionAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",

        pc_range=None,
        use_lidar_feat=True,
        lidar_gated=True,
        lidar_kps_generator=None,
        lidar_feat_scale=0.1,
        lidar_ramp_start=0,
        lidar_ramp_end=1000,
        lidar_align_corners=True,
        enhance_laq_only=False,
    ):
        super(DeformableFeatureFusionAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        self.pc_range = pc_range
        self.use_lidar_feat = use_lidar_feat
        self.pc_range = pc_range
        self.use_lidar_feat = use_lidar_feat
        self.lidar_gated = lidar_gated
        self.lidar_feat_scale = lidar_feat_scale
        self.lidar_ramp_start = lidar_ramp_start
        self.lidar_ramp_end = lidar_ramp_end
        self.lidar_align_corners = lidar_align_corners
        self.enhance_laq_only = enhance_laq_only

        if self.use_lidar_feat:
            if lidar_kps_generator is None:
                lidar_kps_generator = dict(
                    type="SparseBEVKptGenerator",
                    embed_dims=embed_dims,
                    num_learnable_points=4,
                    fix_sample_scale=[
                        [0.0, 0.0],
                        [0.5, 0.0],
                        [-0.5, 0.0],
                        [0.0, 0.5],
                        [0.0, -0.5],
                    ],
                    add_anchor_embeds=True,
                    detach_anchor=True,
                )
            else:
                lidar_kps_generator = lidar_kps_generator.copy()
                lidar_kps_generator.setdefault("embed_dims", embed_dims)

            self.lidar_kps_generator = build_from_cfg(
                lidar_kps_generator,
                PLUGIN_LAYERS,
            )
            self.lidar_num_pts = self.lidar_kps_generator.num_pts

            # Point-wise / group-wise LiDAR sampling weights.
            # Shape after reshape:
            # [B, N, lidar_num_pts, num_groups]
            self.lidar_kpts_weight_fc = Linear(
                embed_dims,
                self.lidar_num_pts * num_groups,
            )

            self.lidar_output_proj = Linear(embed_dims, embed_dims)

            if self.lidar_gated:
                self.lidar_gate = nn.Sequential(
                    nn.Linear(embed_dims * 2, embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dims, embed_dims),
                    nn.Sigmoid(),
                )
                # Conservative initialization: LiDAR branch starts weak.
                nn.init.constant_(self.lidar_gate[-2].bias, -4.0)
        
        if self.use_lidar_feat:
            self.lidar_output_proj = Linear(embed_dims, embed_dims)

            if self.lidar_gated:
                self.lidar_gate = nn.Sequential(
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dims, embed_dims),
                    nn.Sigmoid(),
                )
                nn.init.constant_(self.lidar_gate[-2].bias, -8.0)

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

        if getattr(self, "use_lidar_feat", False):
            constant_init(self.lidar_kpts_weight_fc, val=0.0, bias=0.0)
            xavier_init(self.lidar_output_proj, distribution="uniform", bias=0.0)

        for m in self.modules():
            if m != self and hasattr(m, "init_weights"):
                m.init_weights()

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        pts_feats=None,
        **kwargs: dict,
    ):

        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # fuse multi-point features

        # pts_feats = kwargs.get("pts_feats", None)
        if self.use_lidar_feat and pts_feats is not None:
            lidar_scale = self._get_lidar_scale(metas)

            if lidar_scale > 0:
                # Optionally only enhance LAQ queries.
                # This is safer because final output/cache currently use LAQ only.
                if self.enhance_laq_only and metas is not None and "num_laq" in metas:
                    num_laq = int(metas["num_laq"])
                else:
                    num_laq = None

                if num_laq is not None and num_laq > 0:
                    lidar_instance_feats = instance_feature[:, :num_laq]
                    lidar_anchors = anchor[:, :num_laq]
                    lidar_anchor_embeds = anchor_embed[:, :num_laq]
                else:
                    lidar_instance_feats = instance_feature
                    lidar_anchors = anchor
                    lidar_anchor_embeds = anchor_embed

                lidar_key_points = self.lidar_kps_generator(
                    lidar_anchors,
                    lidar_instance_feats,
                    anchor_embeds=lidar_anchor_embeds
                    if self.lidar_kps_generator.add_anchor_embeds else None,
                )

                lidar_features = self.lidar_feature_sampling(
                    pts_feats=pts_feats,
                    key_points=lidar_key_points,
                    instance_feats=lidar_instance_feats,
                    anchor_embeds=lidar_anchor_embeds,
                )

                lidar_features = self.lidar_output_proj(lidar_features.float())
                lidar_features = lidar_features.to(features.dtype)

                if self.lidar_gated:
                    base_features = (
                        features[:, :num_laq]
                        if num_laq is not None and num_laq > 0
                        else features
                    )
                    lidar_gate = self.lidar_gate(
                        torch.cat(
                            [base_features.float(), lidar_features.float()],
                            dim=-1,
                        )
                    )
                    lidar_gate = lidar_gate.to(features.dtype)
                    lidar_delta = lidar_scale * lidar_gate * lidar_features
                else:
                    lidar_delta = lidar_scale * lidar_features

                if num_laq is not None and num_laq > 0:
                    features = features.clone()
                    features[:, :num_laq] = features[:, :num_laq] + lidar_delta
                else:
                    features = features + lidar_delta

        output = self.proj_drop(self.output_proj(features))        
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    def bev_project(self, key_points: torch.Tensor) -> torch.Tensor:
        """Project 3D key points to normalized BEV coordinates.

        Args:
            key_points: [B, N, P, 3] in LiDAR coordinates.

        Returns:
            bev_points: [B, N, P, 2], normalized to [-1, 1] for grid_sample.
        """
        if self.pc_range is None:
            raise ValueError("pc_range must be set when using LiDAR BEV sampling.")

        x_min, y_min, _, x_max, y_max, _ = self.pc_range

        x = key_points[..., 0]
        y = key_points[..., 1]

        x = (x - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)

        # grid_sample uses [-1, 1]
        x = x * 2.0 - 1.0
        y = y * 2.0 - 1.0

        return torch.stack([x, y], dim=-1)

    def _get_lidar_scale(self, metas=None):
        """Ramp-up LiDAR feature scale."""
        scale = float(self.lidar_feat_scale)

        if metas is None:
            return scale

        cur_iter = metas.get("cur_iter", None)
        if cur_iter is None:
            return scale

        try:
            cur_iter = int(cur_iter)
        except Exception:
            return scale

        if cur_iter < self.lidar_ramp_start:
            return 0.0

        if cur_iter < self.lidar_ramp_end:
            ratio = float(cur_iter - self.lidar_ramp_start) / float(
                max(self.lidar_ramp_end - self.lidar_ramp_start, 1)
            )
            return scale * ratio

        return scale

    def _get_lidar_kpt_weights(self, instance_feats, anchor_embeds):
        """Generate learned BEV point weights.

        Returns:
            weights: [B, N, P, G]
        """
        B, N = instance_feats.shape[:2]

        feats = instance_feats + anchor_embeds
        weights = self.lidar_kpts_weight_fc(feats)
        weights = weights.view(
            B,
            N,
            self.lidar_num_pts,
            self.num_groups,
        )
        weights = weights.softmax(dim=2)

        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                B,
                N,
                self.lidar_num_pts,
                1,
                device=weights.device,
                dtype=weights.dtype,
            )
            weights = (mask > self.attn_drop) * weights / (1.0 - self.attn_drop)

        return weights

    def bev_project(self, key_points):
        """Project LiDAR keypoints to normalized BEV grid coords.

        Args:
            key_points: [B, N, P, 3], LiDAR coordinates.

        Returns:
            grid: [B, N, P, 2], normalized to [-1, 1].
        """
        if self.pc_range is None:
            raise ValueError(
                "pc_range must be set when using LiDAR BEV sampling."
            )

        x_min, y_min, _, x_max, y_max, _ = self.pc_range

        x = key_points[..., 0]
        y = key_points[..., 1]

        x = (x - x_min) / max(x_max - x_min, 1e-5)
        y = (y - y_min) / max(y_max - y_min, 1e-5)

        x = x * 2.0 - 1.0
        y = y * 2.0 - 1.0

        return torch.stack([x, y], dim=-1)

    def lidar_feature_sampling(
        self,
        pts_feats,
        key_points,
        instance_feats,
        anchor_embeds,
    ):
        """Sample LiDAR BEV features with learned point weights.

        Args:
            pts_feats:
                Tensor [B, C, H, W] or list[Tensor].
            key_points:
                [B, N, P, 3].
            instance_feats:
                [B, N, C].
            anchor_embeds:
                [B, N, C].

        Returns:
            lidar_features:
                [B, N, embed_dims].
        """
        if pts_feats is None:
            return None

        if isinstance(pts_feats, torch.Tensor):
            pts_feats = [pts_feats]

        # v1: single-level BEV feature.
        bev_feat = pts_feats[0].float()

        B, C, H, W = bev_feat.shape
        _, N, P, _ = key_points.shape

        if C != self.embed_dims:
            raise RuntimeError(
                f"[DFFA-LiDAR] expected pts_feats channel={self.embed_dims}, "
                f"but got {C}. Please check pts_adapter / pts_out_channels."
            )

        grid = self.bev_project(key_points)  # [B, N, P, 2]
        grid = grid.reshape(B, N * P, 1, 2)

        sampled = torch.nn.functional.grid_sample(
            bev_feat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.lidar_align_corners,
        )  # [B, C, N*P, 1]

        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
        sampled = sampled.reshape(B, N, P, C)  # [B, N, P, C]

        weights = self._get_lidar_kpt_weights(
            instance_feats,
            anchor_embeds,
        )  # [B, N, P, G]

        group_dims = self.embed_dims // self.num_groups

        sampled = sampled.view(
            B,
            N,
            P,
            self.num_groups,
            group_dims,
        )

        lidar_features = (sampled * weights.unsqueeze(-1)).sum(dim=2)
        lidar_features = lidar_features.reshape(B, N, self.embed_dims)

        lidar_features = torch.nan_to_num(
            lidar_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return lidar_features

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features
    @staticmethod
    def bev_feature_sampling(
        pts_feats: List[torch.Tensor],
        key_points: torch.Tensor,
        point_cloud_range,
    ) -> torch.Tensor:
        """Sample LiDAR BEV features according to 3D key points.

        Args:
            pts_feats: list of BEV features, each [B, C, H, W].
            key_points: [B, N, P, 3], in LiDAR coordinates.
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].

        Returns:
            bev_features: [B, N, P, C].
        """
        bev_feat = pts_feats[0]
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range

        x = key_points[..., 0]
        y = key_points[..., 1]

        # normalize to [0, 1]
        grid_x = (x - x_min) / (x_max - x_min)
        grid_y = (y - y_min) / (y_max - y_min)

        # normalize to [-1, 1] for grid_sample
        grid_x = grid_x * 2 - 1
        grid_y = grid_y * 2 - 1

        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, N, P, 2]

        # grid_sample expects [B, H_out, W_out, 2]
        B, N, P = grid.shape[:3]
        grid = grid.reshape(B, N * P, 1, 2)

        sampled = torch.nn.functional.grid_sample(
            bev_feat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [B, C, N*P, 1]

        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        sampled = sampled.reshape(B, N, P, bev_feat.shape[1])

        return sampled