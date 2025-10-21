import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, config, in_channels=3, d_model=256, height=200, width=200, output_dim=32):
        super().__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        self.cfg = config

        # Project from segmentation logits (C channels) to d_model feature size
        if self.cfg.motion_head == "segmentation":
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # [B, 64, 100, 100]
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2), # [B, 128, 50, 50]
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=5, stride=2, padding=2),  # [B, d_model, 25, 25]
            )
            self.proj = nn.Sequential(
                nn.Linear(in_features=3*25*25, out_features=256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)  # For example, output size 10
            )
        elif self.cfg.motion_head == "embedding":
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=2),  # [B, 64, 100, 100]
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=2), # [B, 128, 50, 50]
                nn.ReLU(),
                nn.Conv2d(128, 32, kernel_size=2, stride=2, padding=2),  # [B, d_model, 25, 25]
            )
                # project to a feature vector
            self.proj = nn.Sequential(
                nn.Linear(in_features=32*5*5, out_features=256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)  # For example, output size 10
            )
        else:
            pass
        
    def forward(self, seg_logits):
        """
        seg_logits: [B, C, H, W]  - raw segmentation logits (not normalized)
        Returns:
            memory: [B, L, d_model] - flattened encoder memory for decoder
        """
        if seg_logits.dim() == 4:
            B, C, H, W = seg_logits.shape
            assert H == self.height and W == self.width, "Unexpected input size"
        elif seg_logits.dim() == 3:
            B, N, C = seg_logits.shape
            H = W = int(N ** 0.5)  # assumes square
            seg_logits = seg_logits.reshape(B, C, H, W)

        if self.cfg.motion_head == "segmentation":
            pred_segmentation = torch.argmax(seg_logits, dim=1, keepdim=True).to(dtype=torch.float32)
            pred_segmentation[pred_segmentation == 1] = 0.5
            pred_segmentation[pred_segmentation == 2] = 1.0
            seg_logits = pred_segmentation.flip(dims=[2])

        # pred_seg_img = pred_segmentation[0, :, :][::-1]

        # Normalize segmentation logits → probabilities
        # seg_probs = F.softmax(seg_logits, dim=1)           # [B, C, H, W]

        # Project to d_model channels
        feat = self.conv(seg_logits)                        # [B, d_model, H, W]

        # Project to a feature vector
        feat = self.proj(feat.view(B, -1))

        return feat