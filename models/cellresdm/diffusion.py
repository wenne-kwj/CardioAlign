"""Residual bridge diffusion. Coefficients match CellResDM/ResShift."""
import math
import torch
from torch import nn


class ResidualDiffusion(nn.Module):
    def __init__(self, steps=15, power=.3, kappa=2., sqrt_eta_end=.99,
                 min_noise_level=.2, input_scaling='variance', clip_xstart=False):
        super().__init__()
        if steps < 2 or kappa <= 0 or power <= 0:
            raise ValueError('Invalid diffusion schedule')
        start=min(min_noise_level/kappa,min_noise_level)
        if not 0 < start < sqrt_eta_end <= 1:
            raise ValueError('Schedule endpoints must satisfy 0 < start < end <= 1')
        if input_scaling not in ('variance','none'):
            raise ValueError('Unknown input scaling')
        self.steps,self.kappa,self.input_scaling,self.clip_xstart=steps,kappa,input_scaling,clip_xstart
        fraction=torch.linspace(0,1,steps,dtype=torch.float64).pow(power)
        sqrt_eta=start*torch.exp(fraction*math.log(sqrt_eta_end/start))
        eta=sqrt_eta.square(); previous=torch.cat((torch.zeros(1,dtype=torch.float64),eta[:-1]))
        alpha=eta-previous
        for name,value in {'sqrt_eta':sqrt_eta,'eta':eta,'mean_xt':previous/eta,
                           'mean_x0':alpha/eta,'variance':kappa*kappa*previous/eta*alpha}.items():
            self.register_buffer(name,value)

    @staticmethod
    def anchor(condition):
        if condition.ndim!=4 or condition.shape[1]!=1:
            raise ValueError('Expected Bx1xHxW brightfield')
        return condition.float().repeat(1,3,1,1)

    def coefficient(self,name,t,reference):
        if t.shape!=(reference.shape[0],) or t.dtype!=torch.long or bool(((t<0)|(t>=self.steps)).any()):
            raise ValueError('Invalid timestep indices')
        return getattr(self,name).to(reference.device)[t].float().view(-1,1,1,1)

    def q_sample(self,x0,condition,t,noise):
        anchor=self.anchor(condition)
        if x0.shape!=anchor.shape or noise.shape!=x0.shape:
            raise ValueError('Noisy state and target must have three matching channels')
        return x0.float()+self.coefficient('eta',t,x0)*(anchor-x0.float())+self.kappa*self.coefficient('sqrt_eta',t,x0)*noise.float()

    def predict_x0(self,model,state,condition,t,precision='float32'):
        scaled=state.float()
        if self.input_scaling=='variance':
            scaled=scaled/torch.sqrt(1+self.kappa**2*self.coefficient('eta',t,state))
        with torch.autocast(device_type=state.device.type,dtype=torch.bfloat16,
                            enabled=state.device.type=='cuda' and precision=='bfloat16'):
            result=model(scaled,t,condition)
        if result.shape!=state.shape:
            raise ValueError('Denoiser output shape differs from diffusion state')
        return result.float()

    def posterior(self,state,x0,t):
        return (self.coefficient('mean_xt',t,state)*state+self.coefficient('mean_x0',t,state)*x0,
                self.coefficient('variance',t,state))

    @torch.inference_mode()
    def sample(self,model,condition,noise_source,precision='float32'):
        anchor=self.anchor(condition)
        # noise_source(step) is stateless and keyed by sample/FOV identity.
        initial=noise_source(self.steps).to(anchor)
        if initial.shape!=anchor.shape: raise ValueError('Initial noise shape mismatch')
        state=anchor+self.kappa*self.sqrt_eta[-1].float()*initial
        for index in reversed(range(self.steps)):
            t=torch.full((state.shape[0],),index,device=state.device,dtype=torch.long)
            predicted=self.predict_x0(model,state,condition,t,precision)
            if self.clip_xstart: predicted=predicted.clamp(-1,1)
            mean,variance=self.posterior(state,predicted,t)
            if index:
                noise=noise_source(index).to(state)
                if noise.shape!=state.shape: raise ValueError('Reverse noise shape mismatch')
                state=mean+variance.sqrt()*noise
            else:
                state=mean
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError(f'Non-finite reverse state at step {index}')
        return state
