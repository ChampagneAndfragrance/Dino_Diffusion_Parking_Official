import torch.nn as nn

class FiLMModulator(nn.Module):
    """ We don't use GradCam now so we further filter the bev feature with the FiLM conditioned on the target slot pose """
    def __init__(self, cfg):
        super().__init__()
        target_dim = 3
        feature_dim = cfg.bev_encoder_out_channel
        self.film = nn.Sequential(
            nn.Linear(target_dim, 2 * feature_dim),  # outputs gamma and beta
            nn.ReLU()
        )

    def forward(self, features, target_point):
        """
        features: [B, T, C]
        target_point: [B, D]
        """
        film_params = self.film(target_point)  # [B, 2*C]
        gamma, beta = film_params.chunk(2, dim=-1)  # [B, C] each
        # gamma = gamma.unsqueeze(1)  # [B, 1, C]
        # beta = beta.unsqueeze(1)
        return gamma * features + beta  # [B, T, C]