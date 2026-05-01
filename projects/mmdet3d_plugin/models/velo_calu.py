from typing import Dict, List
import torch
from torch import Tensor
# from mmengine.model import BaseModule
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class VelocityCalculator(BaseModule):
    def __init__(self,
                 ignore_initial: bool = False,
                 invalid_motion_cls: List[int] = None):
        super().__init__()
        self.invalid_motion_cls = invalid_motion_cls
        self.ignore_initial = ignore_initial
        self.prev_seq_ids = None
        self.test_prev_seq_ids = None

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.test_prev_seq_ids = None

    def offset2vel(self,
                   frame_offsets: Tensor,
                   metainfo: Dict[str, Tensor],
                   is_dynamic: Tensor = None):

        # time_gaps = metainfo['frame_time_gaps'].view(-1, 1, 1).clamp(min=1e-6)
        vels = frame_offsets / metainfo['frame_time_gaps'].view(-1, 1, 1)
        if is_dynamic is not None:
            vels = vels * is_dynamic.float()
        return vels

    def process(self,
                vels: Tensor,
                is_dynamic: Tensor = None,
                labels: Tensor = None,
                metainfo: Dict[str, Tensor] = None) -> Tensor:
        if is_dynamic is not None:
            vels = vels * is_dynamic.float()

        if self.ignore_initial and not torch.onnx.is_in_onnx_export() and metainfo is not None:
            prev_seq_ids = self.prev_seq_ids if self.training else self.test_prev_seq_ids
            curr_seq_ids = metainfo['sequence_ids']
            if prev_seq_ids is None:
                vels = vels * 0.0
            else:
                mask = prev_seq_ids == curr_seq_ids
                vels = vels * mask.float().view(-1, 1, 1)

        if len(self.invalid_motion_cls) > 0 and labels is not None:
            invalid_cls = labels.new_tensor(self.invalid_motion_cls)  # [M]
            labels = labels.unsqueeze(-1)
            invalid_cls = invalid_cls.view(1, 1, -1)
            valid = (labels != invalid_cls).all(dim=-1)
            vels = vels * valid.float().unsqueeze(-1)

        return vels

    def update(self, metainfo: Dict[str, Tensor]):
        if torch.onnx.is_in_onnx_export():
            return
        seq_ids = metainfo['sequence_ids']
        if self.training:
            self.prev_seq_ids = seq_ids.clone()
        else:
            self.test_prev_seq_ids = seq_ids.clone()