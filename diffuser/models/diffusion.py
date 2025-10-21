import numpy as np
import torch
from torch import nn

from diffuser.models.helpers import get_schedule_jump

from diffuser.models.temporal_film import ConditionalUnet1D

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class GaussianDiffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # If we learn the global trajectory, it outputs 128 future steps otherwise local trajectory with 16 future steps
        self.horizon = 128 if "global" in cfg.planner_type else 16
        self.observation_dim = cfg.observation_dim
        self.action_dim = cfg.action_dim
        self.transition_dim = self.observation_dim + self.action_dim
        self.diffusion_feature_dim = cfg.diffusion_feat_dim
        self.model = ConditionalUnet1D(cfg, transition_dim=self.diffusion_feature_dim, horizon=self.horizon, global_cond_dim=[32, 128, 32], lstm_dim=6, output_dim=5, global_feature_num=4)

        betas = cosine_beta_schedule(cfg.n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(cfg.n_timesteps)
        self.clip_denoised = cfg.clip_denoised
        self.predict_epsilon = cfg.predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(cfg.action_weight, cfg.loss_discount, cfg.loss_weights)
        self.loss_fn = Losses[cfg.loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.action_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, global_cond):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, global_cond=global_cond)) # if not self.predict_epsilon, then just output noise=model()

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    @torch.no_grad()
    def p_sample(self, x, global_cond, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, global_cond=global_cond)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop_original(self, shape, global_cond, cond, verbose=True, return_diffusion=False):
        device = self.betas.device # device - CPU or GPU?

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x] # if we return the full diffusion denoising process

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()

        # INFO: inverse play the denoising process
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, global_cond, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)
                
        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        
    # @torch.no_grad()
    def p_sample_loop(self, shape, global_cond, cond, estimator, sample_type, return_diffusion=False, verbose=False):
        if sample_type == 'original':
            return self.p_sample_loop_original(shape, global_cond, cond, verbose, return_diffusion)
        else:
            raise NotImplementedError

    # @torch.no_grad()
    def conditional_sample(self, seg_egoMotion_tgtPose, data, horizon=None, estimator=None, return_diffusion=False):
        '''
            conditions : [ (time, state), ... ]
        '''

        batch_size = len(data)
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.diffusion_feature_dim)
        sample_type = "original"

        return self.p_sample_loop(shape, seg_egoMotion_tgtPose, data, estimator, sample_type, return_diffusion)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, global_cond, cond, t):

        # Sample a Gaussian
        noise = torch.randn_like(x_start)

        # Forward process - we can sample from the start timestep to any timestep described by t
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        # Reconstruct sample from denoising
        x_recon = self.model(x_noisy, t, global_cond=global_cond)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon: # reconstruct by predicting the noise
            loss, info = self.loss_fn(x_recon, noise)
        else: # reconstruct by directly predicting the previous step value
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, global_cond, cond):
        x = x.to(self.betas.device) # ground truth future trajectory
        batch_size = len(x) # batch size
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long() # return the random diffusion timesteps of each batch 
        return self.p_losses(x, global_cond, cond, t)

    def forward(self, seg_egoMotion_tgtPose, data):
        return self.conditional_sample(seg_egoMotion_tgtPose, data)

