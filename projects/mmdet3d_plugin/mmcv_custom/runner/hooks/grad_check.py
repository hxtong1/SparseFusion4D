"""Gradient check hook for learnable reference_points in track training."""
from mmcv.runner import HOOKS, Hook
import os


@HOOKS.register_module()
class TrackLearnableGradCheckHook(Hook):
    """Check that reference_points.weight and bev_embedding receive gradients.

    Enable via env TRACK_GRAD_DEBUG=1. Logs every `interval` iters.

    Note: This does NOT check det-head pollution by hist_loss. Use
    tools/verify_det_grad_isolation.py for that.
    """

    def __init__(self, interval=500):
        self.interval = interval

    def after_train_iter(self, runner):
        if os.environ.get('TRACK_GRAD_DEBUG', '0') != '1':
            return
        if runner.iter % self.interval != 0 or runner.iter == 0:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        head = None
        for m in model.modules():
            if m.__class__.__name__ == 'CmtLidarHead':
                head = m
                break
        if head is None:
            runner.logger.info(
                f'[GRAD_CHECK] iter={runner.iter} CmtLidarHead not found')
            return

        ref_w = getattr(head, 'reference_points', None)
        bev_emb = getattr(head, 'bev_embedding', None)
        ref_grad = float('nan')
        bev_grad = float('nan')
        if ref_w is not None and ref_w.weight.grad is not None:
            ref_grad = ref_w.weight.grad.mean().item()
        if bev_emb is not None:
            for p in bev_emb.parameters():
                if p.grad is not None:
                    bev_grad = p.grad.mean().item()
                    break

        runner.logger.info(
            f'[GRAD_CHECK] iter={runner.iter} '
            f'ref_points.grad.mean={ref_grad:.6f} '
            f'bev_embed.grad.mean={bev_grad:.6f}')
