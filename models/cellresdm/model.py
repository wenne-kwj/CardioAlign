"""Single-BF, three-fluorescence CellResDM denoiser."""
import torch
from torch import nn
from .vendor.unet import UNetModelSwin


class CellResDM(nn.Module):
    def __init__(self, image_size=256, model_channels=160, channel_mult=(1,2,2,4),
                 num_res_blocks=(2,2,2,2), attention_resolutions=(32,16,8,4),
                 swin_depth=2, swin_embed_dim=192, window_size=8, num_head_channels=32,
                 mlp_ratio=4., dropout=0.):
        super().__init__()
        if image_size <= 0 or image_size % (2**(len(channel_mult)-1)):
            raise ValueError('image_size must be divisible by the encoder downsampling factor')
        if model_channels <= 0 or model_channels % 32 or swin_embed_dim % 32:
            raise ValueError('GroupNorm feature widths must be positive multiples of 32')
        if window_size <= 0 or num_head_channels <= 0 or swin_embed_dim % num_head_channels:
            raise ValueError('Invalid Swin window/head dimensions')
        if not 0 <= dropout < 1:raise ValueError('Dropout must lie in [0,1)')
        self.image_size = image_size
        self.network = UNetModelSwin(image_size=image_size,in_channels=3,out_channels=3,
            model_channels=model_channels,channel_mult=channel_mult,num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,dropout=dropout,num_head_channels=num_head_channels,
            use_scale_shift_norm=True,swin_depth=swin_depth,swin_embed_dim=swin_embed_dim,
            window_size=window_size,mlp_ratio=mlp_ratio,cond_lq=True,lq_size=image_size)

    def forward(self, state, timesteps, condition):
        if state.ndim != 4 or state.shape[1:] != (3,self.image_size,self.image_size):
            raise ValueError('Diffusion state must be Bx3ximage_sizeximage_size')
        if condition.shape != (state.shape[0],1,self.image_size,self.image_size):
            raise ValueError('Condition must be one BF channel with matching shape')
        if timesteps.shape != (state.shape[0],):
            raise ValueError('One timestep per sample is required')
        if getattr(self,'_channels_last',False):
            state=state.contiguous(memory_format=torch.channels_last)
            condition=condition.contiguous(memory_format=torch.channels_last)
        compiled=getattr(self,'_compiled_train_forward' if self.training and torch.is_grad_enabled() else '_compiled_eval_forward',None)
        if compiled is not None:return compiled(state,timesteps,condition)
        return self._forward_impl(state,timesteps,condition)

    def _forward_impl(self,state,timesteps,condition):
        return self.network(state, timesteps, lq=condition.repeat(1,3,1,1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
