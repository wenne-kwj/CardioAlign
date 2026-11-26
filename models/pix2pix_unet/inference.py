"""Explicit BatchNorm inference policy, with Dropout off and no BN mutation."""
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def inference_policy(model, batchnorm_mode='running', *, allow_tf32=None):
    """Use frozen running stats or each layer's current-window spatial stats.

    In ``current`` mode batch size is one. Each BN layer normalizes over the
    current tile's spatial activations. Running buffers and batch counters are
    neither read for normalization nor updated. All module flags are restored,
    including when evaluation fails. This does not enable gradients.
    """
    if batchnorm_mode not in ('running', 'current'):
        raise ValueError('batchnorm_mode must be running or current')
    modules = list(model.modules())
    training_flags = [(module, module.training) for module in modules]
    batchnorms = [m for m in modules if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    tracking_flags = [(module, module.track_running_stats) for module in batchnorms]
    previous_tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    try:
        if allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        model.eval()
        if batchnorm_mode == 'current':
            for module in batchnorms:
                module.training = True
                module.track_running_stats = False
        elif any(m.running_mean is None or m.running_var is None for m in batchnorms):
            raise ValueError('Running inference requires saved BN statistics')
        yield {'batchnorm_mode': batchnorm_mode,
               'batchnorm_layers': len(batchnorms),
               'dropout_enabled': False,
               'running_buffers_updated': False,
               'allow_tf32': torch.backends.cudnn.allow_tf32,
               'statistics_scope': ('current tile, per layer, B=1 over spatial axes'
                                    if batchnorm_mode == 'current' else 'saved training EMA')}
    finally:
        if allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        for module, flag in tracking_flags:
            module.track_running_stats = flag
        for module, flag in training_flags:
            module.training = flag
