import torch
import torch.nn as nn
import torch.nn.functional as F


class LiDARBEVCrossAttention(nn.Module):
    """Anchor-aware LiDAR BEV Token Cross-Attention.

    This module is designed for Sparse4D-style decoder:
    - query feature after deformable aggregation may be 512-dim
      when residual_mode="cat";
    - LiDAR BEV feature is usually 256-dim after pts_adapter;
    - anchor xy is used as query-side positional prior;
    - TopK BEV tokens are selected from LiDAR BEV feature to reduce
      memory cost and suppress background tokens.
    """

    def __init__(
        self,
        embed_dims=256,
        query_dims=None,
        kv_dims=None,
        kv_in_channels=None,
        num_heads=8,
        dropout=0.1,
        pc_range=None,
        bev_downsample=4,
        num_bev_tokens=512,
        use_pos_embed=True,
        use_anchor_pos=True,
        use_gate=True,
        gate_init_bias=-2.0,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.query_dims = query_dims if query_dims is not None else embed_dims
        self.kv_dims = kv_dims if kv_dims is not None else embed_dims
        self.kv_in_channels = kv_in_channels if kv_in_channels is not None else self.kv_dims

        if self.kv_in_channels != self.kv_dims:
            self.kv_input_proj = nn.Conv2d(
                self.kv_in_channels,
                self.kv_dims,
                kernel_size=1,
                bias=False,
            )
        else:
            self.kv_input_proj = nn.Identity()
        self.num_heads = num_heads
        self.dropout = dropout
        self.bev_downsample = bev_downsample
        self.num_bev_tokens = num_bev_tokens

        self.use_pos_embed = use_pos_embed
        self.use_anchor_pos = use_anchor_pos
        self.use_gate = use_gate
        self.gate_init_bias = gate_init_bias

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
            # Conservative initialization: avoid disturbing original Sparse4D
            # decoder too much at the beginning.
            nn.init.constant_(self.gate[-2].bias, gate_init_bias)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        if self.use_gate:
            nn.init.constant_(self.gate[-2].bias, self.gate_init_bias)
    
    def build_dense_bev_pos(self, B, H, W, device, dtype):
        ys = torch.linspace(0, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(0, 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        grid = grid.reshape(1, H * W, 2).repeat(B, 1, 1)
        return self.bev_pos_mlp(grid)

    def build_anchor_pos(self, anchor):
        """Build anchor xy positional embedding.

        Args:
            anchor: [B, N, D], assumed anchor[..., 0:2] = x, y.
        """
        pc_range = self.pc_range
        xy = anchor[..., 0:2].float()

        x = (xy[..., 0] - pc_range[0]) / max(pc_range[3] - pc_range[0], 1e-5)
        y = (xy[..., 1] - pc_range[1]) / max(pc_range[4] - pc_range[1], 1e-5)
        xy_norm = torch.stack([x, y], dim=-1).clamp(0.0, 1.0)

        return self.anchor_pos_mlp(xy_norm)

    def select_topk_tokens(self, bev_feat):
        """Select informative BEV tokens.

        Args:
            bev_feat: [B, C, H, W]

        Returns:
            lidar_tokens: [B, K, C]
            token_pos: [B, K, 2], normalized xy in [0, 1]
        """
        B, C, H, W = bev_feat.shape

        # Feature norm works as a simple objectness proxy.
        # It avoids introducing an additional unsupervised objectness branch.
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
        token_pos = torch.stack([token_x, token_y], dim=-1)  # [B, K, 2]

        return lidar_tokens, token_pos

    def forward(self, instance_feature, pts_feats, anchor=None):
        """
        Args:
            instance_feature: [B, N, query_dims], e.g. [B, 1220, 512]
            pts_feats: list[[B, C, H, W]], C can be 512 or 256
            anchor: [B, N, D], optional.

        Returns:
            out: [B, N, query_dims]
        """
        if pts_feats is None:
            return instance_feature

        if isinstance(pts_feats, torch.Tensor):
            pts_feats = [pts_feats]

        if len(pts_feats) == 0 or pts_feats[0] is None:
            return instance_feature

        bev_feat = pts_feats[0].float()  # [B, C, H, W]
        C_in = bev_feat.shape[1]

        # ========= channel-adaptive input =========
        # raw pts_neck feature: [B, 512, H, W] -> [B, 256, H, W]
        if C_in == self.kv_in_channels:
            bev_feat = self.kv_input_proj(bev_feat)

        # already adapted feature: [B, 256, H, W]
        elif C_in == self.kv_dims:
            pass

        else:
            raise RuntimeError(
                f"[LiDARBEVCrossAttention] pts_feats channel mismatch: "
                f"got C={C_in}, expected {self.kv_in_channels} or {self.kv_dims}"
            )

        if self.bev_downsample is not None and self.bev_downsample > 1:
            bev_feat = F.avg_pool2d(
                bev_feat,
                kernel_size=self.bev_downsample,
                stride=self.bev_downsample,
            )

        B, C, H, W = bev_feat.shape

        if C != self.kv_dims:
            raise RuntimeError(
                f"[LiDARBEVCrossAttention] pts_feats channel mismatch after proj: "
                f"got C={C}, expected kv_dims={self.kv_dims}"
            )

        if instance_feature.shape[-1] != self.query_dims:
            raise RuntimeError(
                f"[LiDARBEVCrossAttention] instance_feature dim mismatch: "
                f"got {instance_feature.shape[-1]}, expected query_dims={self.query_dims}"
            )

        lidar_tokens, token_pos = self.select_topk_tokens(bev_feat)

        if self.use_pos_embed:
            token_pos_embed = self.bev_pos_mlp(token_pos.to(lidar_tokens.dtype))
            lidar_tokens = lidar_tokens + token_pos_embed

        query_input = instance_feature.float()

        if self.use_anchor_pos and anchor is not None:
            query_input = query_input + self.build_anchor_pos(anchor.detach())

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

        # ========= residual adapter =========
        # Important: residual base must be original instance_feature,
        # not normalized query.
        if self.use_gate:
            gate = self.gate(query_input)
            gate = gate.to(lidar_context.dtype)
            out = instance_feature.float() + gate * lidar_context
        else:
            out = instance_feature.float() + lidar_context

        return out.to(instance_feature.dtype)