import torch
import torch.nn as nn
import torch.nn.functional as F


class LiDARTemporalFusion(nn.Module):
    """Temporal LiDAR Fusion for Sparse4D-style tracking.

    This module refines propagated temporal instance features using current
    frame LiDAR BEV features before temporal graph attention.

    It does not change:
        - query number
        - instance bank logic
        - tracking ID assignment
        - target assignment

    Args:
        temp_instance_feature: [B, Nt, C] or [B, Nt, 2C]
        temp_anchor: [B, Nt, D]
        pts_feats: list[[B, C_lidar, H, W]]
    """

    def __init__(
        self,
        embed_dims=256,
        query_dims=None,
        kv_dims=None,
        num_heads=8,
        dropout=0.1,
        pc_range=None,
        bev_downsample=4,
        num_bev_tokens=512,
        use_anchor_pos=True,
        use_pos_embed=True,
        use_gate=True,
        gate_init_bias=-2.0,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.query_dims = query_dims if query_dims is not None else embed_dims
        self.kv_dims = kv_dims if kv_dims is not None else embed_dims

        self.num_heads = num_heads
        self.dropout = dropout
        self.bev_downsample = bev_downsample
        self.num_bev_tokens = num_bev_tokens

        self.use_anchor_pos = use_anchor_pos
        self.use_pos_embed = use_pos_embed
        self.use_gate = use_gate

        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.pc_range = pc_range

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.query_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=self.kv_dims,
            vdim=self.kv_dims,
        )

        self.query_norm = nn.LayerNorm(self.query_dims)
        self.token_norm = nn.LayerNorm(self.kv_dims)
        self.out_norm = nn.LayerNorm(self.query_dims)
        self.out_proj = nn.Linear(self.query_dims, self.query_dims)
        self.gate_init_bias = gate_init_bias

        if self.use_pos_embed:
            self.bev_pos_mlp = nn.Sequential(
                nn.Linear(2, self.kv_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.kv_dims, self.kv_dims),
            )

        if self.use_anchor_pos:
            self.anchor_pos_mlp = nn.Sequential(
                nn.Linear(2, self.query_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.query_dims, self.query_dims),
            )

        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(self.query_dims, self.query_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.query_dims, self.query_dims),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.gate[-2].bias, gate_init_bias)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        if self.use_gate:
            nn.init.constant_(self.gate[-2].bias, self.gate_init_bias)
    
    def build_anchor_pos(self, anchor):
        pc_range = self.pc_range
        xy = anchor[..., 0:2].float()

        x = (xy[..., 0] - pc_range[0]) / max(pc_range[3] - pc_range[0], 1e-5)
        y = (xy[..., 1] - pc_range[1]) / max(pc_range[4] - pc_range[1], 1e-5)
        xy_norm = torch.stack([x, y], dim=-1).clamp(0.0, 1.0)

        return self.anchor_pos_mlp(xy_norm)

    def select_topk_tokens(self, bev_feat):
        """Select informative LiDAR BEV tokens.

        Args:
            bev_feat: [B, C, H, W]

        Returns:
            lidar_tokens: [B, K, C]
            token_pos: [B, K, 2]
        """
        B, C, H, W = bev_feat.shape

        score_map = bev_feat.float().pow(2).mean(dim=1)  # [B, H, W]
        score = score_map.flatten(1)  # [B, H*W]

        K = min(self.num_bev_tokens, H * W)
        _, topk_inds = torch.topk(score, k=K, dim=1)

        tokens = bev_feat.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
        gather_index = topk_inds.unsqueeze(-1).expand(-1, -1, C)
        lidar_tokens = torch.gather(tokens, 1, gather_index)

        ys = topk_inds // W
        xs = topk_inds % W
        token_x = (xs.float() + 0.5) / max(W, 1)
        token_y = (ys.float() + 0.5) / max(H, 1)
        token_pos = torch.stack([token_x, token_y], dim=-1)

        return lidar_tokens, token_pos

    def forward(self, temp_instance_feature, temp_anchor, pts_feats):
        """
        Args:
            temp_instance_feature: [B, Nt, query_dims]
            temp_anchor: [B, Nt, D]
            pts_feats: list[[B, kv_dims, H, W]]

        Returns:
            refined temp_instance_feature: [B, Nt, query_dims]
        """
        if temp_instance_feature is None or temp_anchor is None:
            return temp_instance_feature

        if pts_feats is None:
            return temp_instance_feature

        if isinstance(pts_feats, torch.Tensor):
            pts_feats = [pts_feats]

        if len(pts_feats) == 0 or pts_feats[0] is None:
            return temp_instance_feature

        if temp_instance_feature.shape[-1] != self.query_dims:
            raise RuntimeError(
                f"[LiDARTemporalFusion] temp_instance_feature dim mismatch: "
                f"got {temp_instance_feature.shape[-1]}, expected {self.query_dims}"
            )

        bev_feat = pts_feats[0].float()

        if self.bev_downsample is not None and self.bev_downsample > 1:
            bev_feat = F.avg_pool2d(
                bev_feat,
                kernel_size=self.bev_downsample,
                stride=self.bev_downsample,
            )

        B, C, H, W = bev_feat.shape

        if C != self.kv_dims:
            raise RuntimeError(
                f"[LiDARTemporalFusion] pts_feats channel mismatch: "
                f"got C={C}, expected kv_dims={self.kv_dims}"
            )

        lidar_tokens, token_pos = self.select_topk_tokens(bev_feat)

        if self.use_pos_embed:
            lidar_tokens = lidar_tokens + self.bev_pos_mlp(
                token_pos.to(lidar_tokens.dtype)
            )

        query_input = temp_instance_feature.float()

        if self.use_anchor_pos:
            query_input = query_input + self.build_anchor_pos(temp_anchor.detach())

        query = self.query_norm(query_input)
        key_value = self.token_norm(lidar_tokens.float())

        lidar_context, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )

        lidar_context = self.out_proj(lidar_context)
        lidar_context = self.out_norm(lidar_context)

        if self.use_gate:
            gate = self.gate(query_input)
            out = temp_instance_feature.float() + gate * lidar_context
        else:
            out = temp_instance_feature.float() + lidar_context

        return out.to(temp_instance_feature.dtype)