"""FP32 MSE and frozen, per-fluorescence-channel VGG-LPIPS."""
from pathlib import Path
import torch
from torch import nn


class DiffusionLoss(nn.Module):
    def __init__(self, lpips_weight=1., torch_hub_dir=None, perceptual=None):
        super().__init__()
        if lpips_weight < 0: raise ValueError('LPIPS weight must be nonnegative')
        self.lpips_weight=lpips_weight
        self.perceptual=None
        if lpips_weight:
            if perceptual is None:
                if torch_hub_dir is None: raise ValueError('An explicit pretrained weight cache is required')
                directory=Path(torch_hub_dir)
                if not (directory/'checkpoints/vgg16-397923af.pth').is_file():
                    raise FileNotFoundError('Prepare VGG weights before training: '+str(directory))
                import lpips
                torch.hub.set_dir(str(directory))
                perceptual=lpips.LPIPS(net='vgg',verbose=False)
            self.perceptual=perceptual.eval()
            self.perceptual.requires_grad_(False)

    def train(self,mode=True):
        super().train(mode)
        if self.perceptual is not None: self.perceptual.eval()
        return self

    def forward(self,prediction,target):
        if prediction.shape!=target.shape or prediction.ndim!=4 or prediction.shape[1]!=3:
            raise ValueError('Expected matching three-channel image batches')
        prediction,target=prediction.float(),target.float()
        with torch.autocast(device_type=prediction.device.type,enabled=False):
            mse=(prediction-target).square().mean((1,2,3))
            perceptual=torch.zeros_like(mse)
            if self.perceptual is not None:
                for channel in range(3):
                    a=prediction[:,channel:channel+1].repeat(1,3,1,1).clamp(-1,1)
                    b=target[:,channel:channel+1].repeat(1,3,1,1)
                    perceptual=perceptual+self.perceptual(a,b).reshape(prediction.shape[0])/3
            values=mse+self.lpips_weight*perceptual
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Non-finite MSE/LPIPS; refusing to hide invalid gradients')
        return values.mean(),{'mse':mse.detach().mean(),'lpips':perceptual.detach().mean()}
