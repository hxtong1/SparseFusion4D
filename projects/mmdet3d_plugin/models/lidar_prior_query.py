import torch
import torch.nn as nn


class LiDARPriorQueryGenerator(nn.Module):
    """Geometry-aware LiDAR Prior Query Generator.

    This module generates a small set of LiDAR-guided prior queries from BEV
    features. It keeps the total Sparse4D query number unchanged by later
    merging these prior queries with the last K learnable queries.

    Assumed Sparse4D anchor format:
        [x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy, ...]
    If your anchor format is different, only modify `update_prior_anchor()`.
    """

    def __init__(
        self,
        embed_dims=256,
        num_prior=100,
        in_channels=None,
        pc_range=None,
        voxel_size=None,
        out_size_factor=8,
        use_geometry_prior=True,
        score_gate_scale=0.5,
        min_gate=0.05,
        max_gate=0.70,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_prior = num_prior
        self.in_channels = in_channels if in_channels is not None else embed_dims

        if self.in_channels != self.embed_dims:
            self.input_proj = nn.Conv2d(
                self.in_channels,
                self.embed_dims,
                kernel_size=1,
                bias=False,
            )
        else:
            self.input_proj = nn.Identity()
        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if voxel_size is None:
            voxel_size = [0.1, 0.1, 0.2]

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.out_size_factor = out_size_factor

        self.use_geometry_prior = use_geometry_prior
        self.score_gate_scale = score_gate_scale
        self.min_gate = min_gate
        self.max_gate = max_gate
        
        if self.use_geometry_prior:
            self.z_head = nn.Conv2d(embed_dims, 1, kernel_size=1)
            self.dim_head = nn.Conv2d(embed_dims, 3, kernel_size=1)
            self.yaw_head = nn.Conv2d(embed_dims, 2, kernel_size=1)
            self.vel_head = nn.Conv2d(embed_dims, 2, kernel_size=1)
        else:
            self.z_head = None
            self.dim_head = None
            self.yaw_head = None
            self.vel_head = None

        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        # Objectness heatmap. It is used for TopK selection and also participates
        # in feature gating, so it receives gradients.
        self.objectness = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, 1, kernel_size=1),
        )

        # Feature projection for prior query feature.
        self.feat_proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

        if self.use_geometry_prior:
            # Lightweight geometry prediction heads.
            self.z_head = nn.Conv2d(embed_dims, 1, kernel_size=1)
            self.dim_head = nn.Conv2d(embed_dims, 3, kernel_size=1)
            self.yaw_head = nn.Conv2d(embed_dims, 2, kernel_size=1)
            self.vel_head = nn.Conv2d(embed_dims, 2, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        # Conservative objectness bias. Avoid producing overly strong prior
        # queries at the beginning.
        last_conv = self.objectness[-1]
        nn.init.constant_(last_conv.bias, -2.0)

        if self.use_geometry_prior:
            nn.init.constant_(self.z_head.bias, 0.0)
            nn.init.constant_(self.dim_head.bias, 0.0)
            nn.init.constant_(self.yaw_head.bias, 0.0)
            nn.init.constant_(self.vel_head.bias, 0.0)

    def gather_feat(self, feat_map, topk_inds):
        """Gather BEV map features by TopK indices.

        Args:
            feat_map: [B, C, H, W]
            topk_inds: [B, K]

        Returns:
            gathered: [B, K, C]
        """
        B, C, H, W = feat_map.shape
        feat_flat = feat_map.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
        gather_index = topk_inds.unsqueeze(-1).expand(-1, -1, C)
        return torch.gather(feat_flat, 1, gather_index)

    def update_prior_anchor(
        self,
        prior_anchor,
        xs,
        ys,
        z_pred=None,
        dim_pred=None,
        yaw_pred=None,
        vel_pred=None,
    ):
        """Update anchor parameters using LiDAR BEV prior predictions.

        Assumed anchor format:
            [x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy, ...]
        """
        pc_range = self.pc_range
        voxel_size = self.voxel_size
        out_size_factor = self.out_size_factor

        x_metric = (
            xs.float() + 0.5
        ) * voxel_size[0] * out_size_factor + pc_range[0]

        y_metric = (
            ys.float() + 0.5
        ) * voxel_size[1] * out_size_factor + pc_range[1]

        prior_anchor[..., 0] = x_metric
        prior_anchor[..., 1] = y_metric

        if self.use_geometry_prior:
            if z_pred is not None and prior_anchor.shape[-1] > 2:
                # z prediction is residual-style. Keep it conservative.
                prior_anchor[..., 2] = prior_anchor[..., 2] + 0.5 * z_pred.squeeze(-1)

            if dim_pred is not None and prior_anchor.shape[-1] > 5:
                # Predict residual on dimensions. Clamp to avoid unstable boxes.
                dim_delta = dim_pred.clamp(min=-1.0, max=1.0)
                prior_anchor[..., 3:6] = prior_anchor[..., 3:6] + 0.2 * dim_delta

            if yaw_pred is not None and prior_anchor.shape[-1] > 7:
                yaw = torch.tanh(yaw_pred)
                norm = torch.norm(yaw, dim=-1, keepdim=True).clamp(min=1e-6)
                yaw = yaw / norm

                # Conservative yaw update.
                prior_anchor[..., 6:8] = (
                    0.7 * prior_anchor[..., 6:8] + 0.3 * yaw
                )

            if vel_pred is not None and prior_anchor.shape[-1] > 9:
                vel_delta = vel_pred.clamp(min=-2.0, max=2.0)
                prior_anchor[..., 8:10] = prior_anchor[..., 8:10] + 0.2 * vel_delta

        return prior_anchor

    def forward(self, pts_feats, template_anchor):
        """
        Args:
            pts_feats: list[[B, C, H, W]]
            template_anchor: [B, N, D]

        Returns:
            prior_feature: [B, K, C]
            prior_anchor: [B, K, D]
            prior_score: [B, K]
        """
        if pts_feats is None:
            return None, None, None

        if isinstance(pts_feats, torch.Tensor):
            pts_feats = [pts_feats]

        if len(pts_feats) == 0 or pts_feats[0] is None:
            return None, None, None

        bev_feat = pts_feats[0].float()  # [B, C, H, W]
        B, C, H, W = bev_feat.shape

        # ========= channel-adaptive input =========
        # Case 1: raw LiDAR neck feature, e.g. [B, 512, H, W]
        if C == self.in_channels:
            bev_feat = self.input_proj(bev_feat)

        # Case 2: already adapted LiDAR feature, e.g. [B, 256, H, W]
        elif C == self.embed_dims:
            pass

        else:
            raise RuntimeError(
                f"[LiDARPriorQuery] pts_feats channel mismatch: "
                f"got C={C}, expected {self.in_channels} or {self.embed_dims}"
            )

        # After this, bev_feat must be [B, embed_dims, H, W]
        feat = self.shared_conv(bev_feat)

        heatmap = self.objectness(feat).sigmoid()  # [B, 1, H, W]
        score = heatmap.flatten(1)  # [B, H*W]

        K = min(self.num_prior, H * W, template_anchor.shape[1])
        topk_score, topk_inds = torch.topk(score, k=K, dim=1)

        ys = topk_inds // W
        xs = topk_inds % W

        # Prior feature from BEV features.
        prior_feature = self.gather_feat(feat, topk_inds)  # [B, K, embed_dims]
        prior_feature = self.feat_proj(prior_feature)

        # Let objectness participate in feature construction to avoid DDP unused params.
        score_gate = topk_score.unsqueeze(-1).to(prior_feature.dtype)
        prior_feature = prior_feature * (1.0 + score_gate)

        # Prior anchor from template anchor.
        prior_anchor = template_anchor[:, :K, :].clone()

        if self.use_geometry_prior:
            z_pred = self.gather_feat(self.z_head(feat), topk_inds)
            dim_pred = self.gather_feat(self.dim_head(feat), topk_inds)
            yaw_pred = self.gather_feat(self.yaw_head(feat), topk_inds)
            vel_pred = self.gather_feat(self.vel_head(feat), topk_inds)
        else:
            z_pred = dim_pred = yaw_pred = vel_pred = None

        prior_anchor = self.update_prior_anchor(
            prior_anchor=prior_anchor,
            xs=xs,
            ys=ys,
            z_pred=z_pred,
            dim_pred=dim_pred,
            yaw_pred=yaw_pred,
            vel_pred=vel_pred,
        )

        return prior_feature, prior_anchor, topk_score