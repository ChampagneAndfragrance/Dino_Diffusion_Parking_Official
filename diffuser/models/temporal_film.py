from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from torchsummary import summary

from diffuser.models.temporal import TemporalUnet
from diffuser.models.fc_encoder import EncoderFC

import numpy as np

from .cnn_encoder import EncoderCNN

from diffuser.models.helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

logger = logging.getLogger(__name__)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups), # Conv1dBlock: Conv1d --> GroupNorm --> Mish
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups), # Conv1dBlock: Conv1d --> GroupNorm --> Mish
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        # self.blocks are basically convolution layers (for input)
        out = self.blocks[0](x)

        # linear layer for condition
        cond = cond.float()
        embed = self.cond_encoder(cond)

        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias # This is what happens in the paper
        else:
            out = out + embed

        # self.blocks are basically convolution layers (for input)
        out = self.blocks[1](out)

        # Residual Network
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        cfg,
        dim = 32,
        transition_dim = 32, 
        horizon = None,
        cond_dim = None,
        local_cond_dim=None,
        global_cond_dim=None,
        output_dim = 32,
        motion_feature_num = 3,
        target_feature_num = 3,
        hidden_dim = 32,

        dim_mults=(1, 2, 4, 8),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        lstm_dim = 3, 
        global_feature_num = 5,
        ):
        super().__init__()

        # Input dimensions, 3 for SE2 Cartesian space
        input_dim = transition_dim
        # This horizon specifies how long the predicted trajectory is into the future
        self.horizon = horizon

        # Down_dims for residual block - 32 * (1, 2, 4, 8)
        down_dims = [dim * mult for mult in dim_mults]
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Define the diffusion step encoder
        dsed = global_cond_dim[0]
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # Add global condition dimensions in 
        cond_dim = np.sum(global_cond_dim)
        
        # Condition encoder for the DINO embeddings and the target
        self.segmentation_encoder = EncoderCNN(config=cfg, in_channels = 267 if cfg.motion_head == "embedding" else 1, d_model=cfg.tf_de_dim, height=200, width=200, output_dim=global_cond_dim[1])
        self.target_encoder = EncoderFC(input_dim=target_feature_num, hidden_dim = hidden_dim, output_dim = global_cond_dim[2])

        # Define the input and output channel dimensions for the components in local_cond_encoder, down_modules, and up_modules; [(2, 32), (32, 64), (64, 128), (128, 256)]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),)

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )



    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None, **kwargs):
        """
        sample: (B,horizon,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """

        # batch, horizon, input_dim -> batch, input_dim, horizon
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # Encode diffusion timesteps into the global features
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # Broadcast to batch dimension
        timesteps = timesteps.expand(sample.shape[0])

        # Sinusoidal encoding of the diffusion steps
        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            if 'pred_segmentation' in global_cond.keys():
                # Encode the DINO embeddings
                segmentation_encoded = self.segmentation_encoder(global_cond['pred_segmentation'])
                global_feature = torch.cat([segmentation_encoded, global_feature], axis=-1)
                # Encode the target slot
            if 'target_point' in global_cond.keys():
                targetPoint_encoded = self.target_encoder(global_cond['target_point'])
                global_feature = torch.cat([targetPoint_encoded, global_feature], axis=-1)            

        # no local features
        h_local = list()
        
        x = sample
        h = []
        # self.down_modules: [ConditionalResidualBlock1D*2, Downsample1d]*N
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # self.mid_modules: [ConditionalResidualBlock1D, ConditionalResidualBlock1D]
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # self.up_modules [ConditionalResidualBlock1D*2, Upsample1d]*N
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        # conv block + conv
        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transition_dim = 2
    horizon = 64
    cond_dim = 0
    
    
    model = ConditionalUnet1D(
        horizon = horizon,
        transition_dim = transition_dim,
        cond_dim = cond_dim,
        global_cond_dim = 0,
        dim = 32,
        dim_mults=(1, 2, 4, 8),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=True
    ).to(device)


    model_orig = TemporalUnet(
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ).to(device)

    # test forward
    x = torch.randn(2, 64, 2).to(device)
    timestep = torch.tensor([32, 32]).to(device)
    global_cond = torch.randn((2, 8)).to(device)

    out_orig = model_orig(x, None, timestep)
    print(out_orig.shape)

    out_new = model(x, timestep, global_cond = None)
    print(out_new.shape)

    model = ConditionalUnet1D(
        horizon = horizon,
        transition_dim = transition_dim,
        cond_dim = cond_dim,
        global_cond_dim = 8,
        dim = 32,
        dim_mults=(1, 2, 4, 8),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=True
    ).to(device)

    out_global_cond = model(x, timestep, global_cond = global_cond)
    print(out_global_cond.shape)