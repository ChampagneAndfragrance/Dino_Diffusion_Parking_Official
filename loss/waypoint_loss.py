import torch
import numpy as np

from torch import nn
from tool.config import Configuration


class WaypointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(WaypointLoss, self).__init__()
        self.cfg = cfg
        self.pad_idx = self.cfg.token_nums - 1
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, pred, data):
        pred_waypoint = pred.reshape(-1, pred.shape[-1])
        gt_waypoint = data['gt_waypoint'][:, 1:].reshape(-1).cuda()
        waypoint_loss = self.ce_loss(pred_waypoint, gt_waypoint)
        return waypoint_loss

