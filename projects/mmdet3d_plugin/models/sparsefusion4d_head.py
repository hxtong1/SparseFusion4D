# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple, Union
from collections import defaultdict
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from .blocks import DeformableFeatureAggregation as DFG

__all__ = ["SparseFusion4DHead"]


@HEADS.register_module()
class SparseFusion4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,

        # ===== LiDAR BEV feature adapter =====
        use_pts_feat: bool = False,
        use_lidar_prior_query: bool = True,
        lidar_prior_score_thr: float = 0.01,
        lidar_prior_feat_scale: float = 0.2,
        cache_lidar_prior: bool = False,
        output_lidar_prior: bool = False,
        pts_in_channels: int = 512,
        pts_out_channels: int = 256,
        **kwargs,
    ):
        super(SparseFusion4DHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = self.instance_bank.embed_dims

        self.use_pts_feat = use_pts_feat
        self.use_lidar_prior_query = use_lidar_prior_query
        self.lidar_prior_score_thr = lidar_prior_score_thr
        self.lidar_prior_feat_scale = lidar_prior_feat_scale
        self.cache_lidar_prior = cache_lidar_prior
        self.output_lidar_prior = output_lidar_prior

        self.pts_adapter = None
        if self.use_pts_feat:
            self.pts_adapter = nn.Conv2d(
                pts_in_channels,
                pts_out_channels,
                kernel_size=1,
                bias=False,
            )
            
        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def prepare_instance_info(
        self,
        laq_info: dict = None,
        prior_info: dict = None,
        prop_info: dict = None,
    ) -> dict:
        """Prepare instance queries in industrial-style order.

        Query order:
            [LAQ, prior, prop]

        Important:
            Bool masks are NOT used in the differentiable feature path.
            They are only stored for bookkeeping. This avoids PyTorch
            inplace-version errors on CUDABoolTensor during backward.
        """
        if laq_info is None:
            laq_info = dict()
        if prior_info is None:
            prior_info = dict()
        if prop_info is None:
            prop_info = dict()

        assert any(len(info) > 0 for info in [laq_info, prior_info, prop_info])

        query_anchors = []
        query_feats = []
        valid_masks = []
        query_sources = []

        num_laq = 0
        num_prior = 0
        num_prop = 0
        num_cam_prior = 0
        num_lidar_prior = 0

        # ---------------------------------------------------------
        # 1. LAQ first
        # ---------------------------------------------------------
        if len(laq_info) > 0:
            laq_anchor = laq_info["anchors"]
            laq_feature = laq_info["instance_feats"]

            query_anchors.append(laq_anchor)
            query_feats.append(laq_feature)

            bs, num_laq = laq_anchor.shape[:2]

            laq_valid_mask = torch.ones(
                (bs, num_laq),
                dtype=torch.bool,
                device=laq_anchor.device,
            )
            laq_query_source = torch.zeros(
                (bs, num_laq),
                dtype=torch.long,
                device=laq_anchor.device,
            )

            valid_masks.append(laq_valid_mask)
            query_sources.append(laq_query_source)

        assert len(query_anchors) > 0, \
            "Current Sparse4D adaptation requires LAQ / bank queries."

        ref_anchor = query_anchors[0]
        ref_feature = query_feats[0]

        # ---------------------------------------------------------
        # 2. Prior second
        # ---------------------------------------------------------
        if len(prior_info) > 0:
            prior_anchor = prior_info["prior_anchors"]
            prior_feature = prior_info["prior_instance_feats"]
            prior_scores = prior_info.get("prior_scores", None)
            prior_valid_mask = prior_info.get("prior_valid_mask", None)

            prior_anchor = prior_anchor.to(
                device=ref_anchor.device,
                dtype=ref_anchor.dtype,
            )
            prior_feature = prior_feature.to(
                device=ref_feature.device,
                dtype=ref_feature.dtype,
            )

            anchor_dim = ref_anchor.shape[-1]
            feat_dim = ref_feature.shape[-1]

            if prior_anchor.shape[-1] < anchor_dim:
                pad_shape = list(prior_anchor.shape)
                pad_shape[-1] = anchor_dim - prior_anchor.shape[-1]
                prior_anchor = torch.cat(
                    [prior_anchor, prior_anchor.new_zeros(*pad_shape)],
                    dim=-1,
                )
            elif prior_anchor.shape[-1] > anchor_dim:
                prior_anchor = prior_anchor[..., :anchor_dim]

            if prior_feature.shape[-1] != feat_dim:
                raise RuntimeError(
                    f"prior feature dim {prior_feature.shape[-1]} != "
                    f"LAQ feature dim {feat_dim}"
                )

            prior_anchor = torch.nan_to_num(
                prior_anchor,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            prior_feature = torch.nan_to_num(
                prior_feature,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            bs, num_prior = prior_anchor.shape[:2]

            # -----------------------------------------------------
            # Bool mask only for bookkeeping, NOT for feature graph.
            # -----------------------------------------------------
            if prior_valid_mask is not None:
                prior_valid_mask = prior_valid_mask.to(
                    device=prior_anchor.device,
                    dtype=torch.bool,
                ).clone().detach()
            elif prior_scores is not None:
                prior_scores_detached = prior_scores.detach().to(
                    device=prior_anchor.device
                )
                prior_scores_detached = torch.nan_to_num(
                    prior_scores_detached,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                prior_valid_mask = (
                    prior_scores_detached > self.lidar_prior_score_thr
                ).clone().detach()
            else:
                prior_scores_detached = None
                prior_valid_mask = torch.ones(
                    (bs, num_prior),
                    dtype=torch.bool,
                    device=prior_anchor.device,
                )

            # -----------------------------------------------------
            # Use detached FLOAT score weight instead of Bool mask.
            # This prevents CUDABoolTensor inplace-version errors.
            # Padded proposals usually have score 0, so this also
            # suppresses padded prior tokens.
            # -----------------------------------------------------
            if prior_scores is not None:
                score_weight = prior_scores.detach().to(
                    device=prior_feature.device,
                    dtype=prior_feature.dtype,
                )
                score_weight = torch.nan_to_num(
                    score_weight,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(min=0.0, max=1.0)
                prior_feature = prior_feature * score_weight.unsqueeze(-1)

            # -----------------------------------------------------
            # Dynamic prior feature scale.
            # -----------------------------------------------------
            scale = self.lidar_prior_feat_scale

            cur_iter = getattr(self, "_cur_iter", None)
            if cur_iter is not None:
                try:
                    cur_iter_int = int(cur_iter)
                except Exception:
                    cur_iter_int = None

                if cur_iter_int is not None:
                    ramp_start = getattr(self, "lidar_prior_ramp_start", 0)
                    ramp_end = getattr(self, "lidar_prior_ramp_end", 1000)

                    if cur_iter_int < ramp_start:
                        scale = 0.0
                    elif cur_iter_int < ramp_end:
                        ratio = float(cur_iter_int - ramp_start) / float(
                            max(ramp_end - ramp_start, 1)
                        )
                        scale = scale * ratio

            prior_feature = prior_feature * scale

            prior_query_source = torch.ones(
                (bs, num_prior),
                dtype=torch.long,
                device=prior_anchor.device,
            )

            query_anchors.append(prior_anchor)
            query_feats.append(prior_feature)
            valid_masks.append(prior_valid_mask)
            query_sources.append(prior_query_source)

            num_cam_prior = prior_info.get("num_cam_prior", 0)
            num_lidar_prior = prior_info.get("num_lidar_prior", 0)

        # ---------------------------------------------------------
        # 3. Prop last
        # ---------------------------------------------------------
        if len(prop_info) > 0:
            prop_anchor = prop_info["anchors"].to(
                device=ref_anchor.device,
                dtype=ref_anchor.dtype,
            )
            prop_feature = prop_info["instance_feats"].to(
                device=ref_feature.device,
                dtype=ref_feature.dtype,
            )

            query_anchors.append(prop_anchor)
            query_feats.append(prop_feature)

            bs, num_prop = prop_anchor.shape[:2]

            if "confidences" in prop_info:
                prop_conf = prop_info["confidences"].detach().to(
                    device=prop_anchor.device,
                    dtype=prop_anchor.dtype,
                )
                prop_valid_mask = (prop_conf > 0.0).clone().detach()
            else:
                prop_conf = None
                prop_valid_mask = torch.ones(
                    (bs, num_prop),
                    dtype=torch.bool,
                    device=prop_anchor.device,
                )

            prop_query_source = prop_anchor.new_full(
                (bs, num_prop),
                fill_value=2,
                dtype=torch.long,
            )

            valid_masks.append(prop_valid_mask)
            query_sources.append(prop_query_source)

        # ---------------------------------------------------------
        # 4. Concatenate
        # ---------------------------------------------------------
        anchors = torch.cat(query_anchors, dim=1)
        instance_feats = torch.cat(query_feats, dim=1)

        bs, num_anchors = anchors.shape[:2]

        # Clone after cat. Do not modify this tensor in-place later.
        valid_mask = torch.cat(valid_masks, dim=1).clone().detach()
        query_source = torch.cat(query_sources, dim=1)

        query_info = dict(
            anchors=anchors,
            instance_feats=instance_feats,
            num_anchors=num_anchors,
            num_laq=num_laq,
            num_prior=num_prior,
            num_prop=num_prop,
            num_cam_prior=num_cam_prior,
            num_lidar_prior=num_lidar_prior,
            valid_mask=valid_mask,
            query_source=query_source,
        )

        # ---------------------------------------------------------
        # 5. Prop attention mask.
        # No in-place bool indexing.
        # ---------------------------------------------------------
        if len(prop_info) > 0 and "confidences" in prop_info:
            prop_conf = prop_info["confidences"].detach().to(
                device=anchors.device,
                dtype=anchors.dtype,
            )
            prop_valid = (prop_conf > 0.0).clone().detach()

            prop_attn_values = torch.where(
                prop_valid,
                torch.zeros_like(prop_conf),
                prop_conf - 1.0,
            )

            prop_attn_mask = prop_attn_values.unsqueeze(1).expand(
                -1,
                num_anchors,
                -1,
            )

            prefix_zeros = anchors.new_zeros(
                (bs, num_anchors, num_anchors - num_prop)
            )
            query_info["attn_mask"] = torch.cat(
                [prefix_zeros, prop_attn_mask],
                dim=-1,
            )

            # Rebuild instead of in-place assignment.
            query_info["valid_mask"] = torch.cat(
                [valid_mask[:, :num_anchors - num_prop], prop_valid],
                dim=1,
            ).clone().detach()

        return query_info

    def merge_lidar_prior_query(
        self,
        anchor: torch.Tensor,
        instance_feature: torch.Tensor,
        lidar_prior_info,
    ):
        """Merge Sparse4D bank query and LiDAR prior query with source states.

        Query order:
            [bank_query, lidar_prior_query]

        query_source:
            0 = bank / learnable / temporal query
            1 = LiDAR prior query
        """
        B, N_bank = anchor.shape[:2]

        bank_valid_mask = torch.ones(
            (B, N_bank),
            dtype=torch.bool,
            device=anchor.device,
        )
        bank_query_source = torch.zeros(
            (B, N_bank),
            dtype=torch.long,
            device=anchor.device,
        )

        if lidar_prior_info is None:
            return (
                anchor,
                instance_feature,
                0,
                bank_valid_mask,
                bank_query_source,
            )

        prior_anchor = lidar_prior_info["anchors"]
        prior_feature = lidar_prior_info["instance_feats"]
        prior_scores = lidar_prior_info.get("scores", None)

        prior_anchor = prior_anchor.to(device=anchor.device, dtype=anchor.dtype)
        prior_feature = prior_feature.to(
            device=instance_feature.device,
            dtype=instance_feature.dtype,
        )

        anchor_dim = anchor.shape[-1]
        feat_dim = instance_feature.shape[-1]

        if prior_anchor.shape[-1] < anchor_dim:
            pad = prior_anchor.new_zeros(
                *prior_anchor.shape[:-1],
                anchor_dim - prior_anchor.shape[-1],
            )
            prior_anchor = torch.cat([prior_anchor, pad], dim=-1)
        elif prior_anchor.shape[-1] > anchor_dim:
            prior_anchor = prior_anchor[..., :anchor_dim]

        if prior_feature.shape[-1] != feat_dim:
            raise RuntimeError(
                f"LiDAR prior feature dim {prior_feature.shape[-1]} "
                f"!= instance feature dim {feat_dim}"
            )

        prior_anchor = torch.nan_to_num(
            prior_anchor,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        prior_feature = torch.nan_to_num(
            prior_feature,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        B, N_prior = prior_anchor.shape[:2]

        if prior_scores is not None:
            prior_scores = prior_scores.to(device=anchor.device)
            prior_scores = torch.nan_to_num(
                prior_scores,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            prior_valid_mask = prior_scores > self.lidar_prior_score_thr
        else:
            prior_valid_mask = torch.ones(
                (B, N_prior),
                dtype=torch.bool,
                device=anchor.device,
            )

        # 工业代码里 prior 有 valid_mask；当前 Sparse4D attention mask 不完全兼容 B,N,N，
        # 所以先把 invalid prior feature 置零，并用 feature scale 降低扰动。
        prior_feature = prior_feature * prior_valid_mask.unsqueeze(-1).to(
            prior_feature.dtype
        )
        prior_feature = prior_feature * self.lidar_prior_feat_scale

        prior_query_source = torch.ones(
            (B, N_prior),
            dtype=torch.long,
            device=anchor.device,
        )

        anchor = torch.cat([anchor, prior_anchor], dim=1)
        instance_feature = torch.cat([instance_feature, prior_feature], dim=1)
        valid_mask = torch.cat([bank_valid_mask, prior_valid_mask], dim=1)
        query_source = torch.cat([bank_query_source, prior_query_source], dim=1)

        return (
            anchor,
            instance_feature,
            N_prior,
            valid_mask,
            query_source,
        )


    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # Current training iteration passed from detector.
        # Used for prior feature scale ramp-up.
        self._cur_iter = metas.get("cur_iter", None)




        pts_feats = metas.get("pts_feats", None)
        if self.use_pts_feat and pts_feats is not None:
            if isinstance(pts_feats, torch.Tensor):
                pts_feats = [pts_feats]

            # Keep LiDAR BEV features in fp32 before Conv2d adapter.
            pts_feats = [self.pts_adapter(x.float()) for x in pts_feats]
            pts_feats = [x.float() for x in pts_feats]
            metas["pts_feats"] = pts_feats

        # =========================================================
        # 1. Get LAQ / bank queries from Sparse4D instance bank.
        # =========================================================
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None

        (
            laq_instance_feature,
            laq_anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size,
            metas,
            dn_metas=self.sampler.dn_metas,
        )

        laq_info = dict(
            anchors=laq_anchor,
            instance_feats=laq_instance_feature,
        )

        # =========================================================
        # 2. Industrial-style query preparation.
        #    Query order is strictly: [LAQ, prior, prop].
        #    Prior comes from detector-level data["instance_info"].
        #    Do NOT call merge_lidar_prior_query() here again.
        # =========================================================
        query_info = self.prepare_instance_info(
            laq_info=laq_info,
            prior_info=metas.get("instance_info", None),
            prop_info=None,
        )

        anchor = query_info["anchors"]
        instance_feature = query_info["instance_feats"]
        query_valid_mask = query_info["valid_mask"]
        query_source = query_info["query_source"]

        num_laq = query_info["num_laq"]
        num_prior = query_info["num_prior"]
        num_prop = query_info["num_prop"]
        num_lidar_prior = query_info["num_lidar_prior"]
        num_cam_prior = query_info["num_cam_prior"]
        num_anchors = query_info["num_anchors"]

        # ========= prepare for denoising training ============
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        temp_dn_cls_target = None
        temp_valid_mask = None
        dn_id_target = None
        valid_mask = None
        dn_reg_target = None
        dn_cls_target = None
        num_free_instance = anchor.shape[1]

        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x["instance_id"]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None

            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )

        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        cls = None

        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)

                # Update only LAQ / bank queries. Prior and prop are not written
                # into Sparse4D temporal instance_bank.
                if len(prediction) == self.num_single_frame_decoder:
                    if num_prior > 0 or num_prop > 0:
                        laq_instance_feature = instance_feature[:, :num_laq]
                        laq_anchor = anchor[:, :num_laq]
                        laq_cls = cls[:, :num_laq] if cls is not None else None

                        other_instance_feature = instance_feature[:, num_laq:num_free_instance]
                        other_anchor = anchor[:, num_laq:num_free_instance]
                        other_cls = (
                            cls[:, num_laq:num_free_instance]
                            if cls is not None else None
                        )

                        laq_instance_feature, laq_anchor = self.instance_bank.update(
                            laq_instance_feature,
                            laq_anchor,
                            laq_cls,
                        )

                        instance_feature = torch.cat(
                            [laq_instance_feature, other_instance_feature],
                            dim=1,
                        )
                        anchor = torch.cat([laq_anchor, other_anchor], dim=1)
                        if cls is not None:
                            cls = torch.cat([laq_cls, other_cls], dim=1)
                    else:
                        instance_feature, anchor = self.instance_bank.update(
                            instance_feature,
                            anchor,
                            cls,
                        )

                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask.clone().detach()
                            if valid_mask is not None else None,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )

                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # =========================================================
        # Split DN predictions if denoising is enabled.
        # After this block, classification / prediction / quality
        # correspond to normal queries only: [LAQ, prior, prop].
        # =========================================================
        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] if x is not None else None
                for x in classification
            ]
            classification = [
                x[:, :num_free_instance] if x is not None else None
                for x in classification
            ]

            dn_prediction = [
                x[:, num_free_instance:] if x is not None else None
                for x in prediction
            ]
            prediction = [
                x[:, :num_free_instance] if x is not None else None
                for x in prediction
            ]

            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]

            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask.clone().detach()
                    if valid_mask is not None else None,
                }
            )

            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask.clone().detach()
                        if temp_valid_mask is not None else None,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask

            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]

            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            if cls is not None:
                cls = cls[:, :num_free_instance]

            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask.clone().detach()
                if valid_mask is not None else None,
                dn_id_target,
            )

        # =========================================================
        # Output policy.
        # Query order: [LAQ, prior, prop].
        # Current stage: prior participates in decoder interaction,
        # but final loss/post_process/cache/instance_id only use LAQ.
        # =========================================================
        output_lidar_prior = getattr(self, "output_lidar_prior", False)
        if (num_prior > 0 or num_prop > 0) and not output_lidar_prior:
            classification_for_output = [
                x[:, :num_laq] if x is not None else None
                for x in classification
            ]
            prediction_for_output = [
                x[:, :num_laq] if x is not None else None
                for x in prediction
            ]
            quality_for_output = [
                x[:, :num_laq] if x is not None else None
                for x in quality
            ]
        else:
            classification_for_output = classification
            prediction_for_output = prediction
            quality_for_output = quality

        output.update(
            {
                "classification": classification_for_output,
                "prediction": prediction_for_output,
                "quality": quality_for_output,
            }
        )

        # Cache only LAQ queries by default.
        cache_lidar_prior = getattr(self, "cache_lidar_prior", False)
        if (num_prior > 0 or num_prop > 0) and not cache_lidar_prior:
            cache_instance_feature = instance_feature[:, :num_laq]
            cache_anchor = anchor[:, :num_laq]
            cache_cls = cls[:, :num_laq] if cls is not None else None
        else:
            cache_instance_feature = instance_feature
            cache_anchor = anchor
            cache_cls = cls

        self.instance_bank.cache(
            cache_instance_feature,
            cache_anchor,
            cache_cls,
            metas,
            feature_maps,
        )

        output["instance_info"] = dict(
            anchors=anchor,
            instance_feats=instance_feature,
            valid_mask=query_valid_mask,
            query_source=query_source,
            num_laq=num_laq,
            num_prior=num_prior,
            num_prop=num_prop,
            num_lidar_prior=num_lidar_prior,
            num_cam_prior=num_cam_prior,
            num_anchors=num_anchors,
        )

        if not self.training:
            if (num_prior > 0 or num_prop > 0) and not output_lidar_prior:
                id_cls = cls[:, :num_laq] if cls is not None else None
                id_anchor = anchor[:, :num_laq]
            else:
                id_cls = cls
                id_anchor = anchor

            instance_id = self.instance_bank.get_instance_id(
                id_cls,
                id_anchor,
                self.decoder.score_threshold,
            )
            output["instance_id"] = instance_id

        assert "classification" in output, output.keys()
        assert "prediction" in output, output.keys()
        assert "quality" in output, output.keys()

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if "dn_prediction" not in model_outs:
            return output

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )
