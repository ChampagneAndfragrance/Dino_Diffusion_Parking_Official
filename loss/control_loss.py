import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from tool.config import Configuration


class ControlLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlLoss, self).__init__()
        self.cfg = cfg
        self.pad_idx = self.cfg.token_nums - 1
        self.valid_token = self.cfg.token_nums - 4
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, pred, data):
        gt = data['gt_control'][:, 1:]
        # pred_control = pred.reshape(-1, pred.shape[-1])
        # gt_control = data['gt_control'][:, 1:].reshape(-1).cuda()

        ## INFO: New control loss
        # Step 1: Convert logits to probabilities
        # probs = F.softmax(pred_control, dim=-1)  # [B, T, 200]

        # Step 2: Create bin indices [0, 1, ..., 199]
        # bins = torch.arange(pred_control.shape[-1], device=pred.device).float()  # [200]

        # Step 3: Compute expected value over bins
        # expected = (probs * bins).sum(dim=-1)  # [B, T]

        # Step 4: Compute L1 or L2 loss against ground-truth labels
        # control_loss = (0.5 / self.valid_token) * self.l1_loss(expected, gt_control.float())  # or use F.mse_loss(expected, gt.float())

        ## INFO: Old control loss
        # control_loss = self.ce_loss(pred_control, gt_control)

        # Select throttle/brake positions
        tb_indices = [0, 3, 6, 9]
        pred_tb = pred[:, tb_indices, :].reshape(-1, pred.shape[-1])      # [16, 4, 204]
        gt_tb = gt[:, tb_indices].reshape(-1)            # [16, 4]
        loss_tb = self.ce_loss(pred_tb, gt_tb)

        # Select steer positions
        steer_indices = [1, 4, 7, 10]
        pred_steer = pred[:, steer_indices, :].reshape(-1, pred.shape[-1])
        gt_steer = gt[:, steer_indices].reshape(-1)
        loss_steer = self.ce_loss(pred_steer, gt_steer)

        # Select gear positions
        gear_indices = [2, 5, 8, 11]
        pred_gear = pred[:, gear_indices, :].reshape(-1, pred.shape[-1])  # [16, 4, 204]
        gt_gear = gt[:, gear_indices].reshape(-1)        # [16, 4]
        loss_gear = self.ce_loss(pred_gear, gt_gear)

        # Select end tokens
        end_indices = [12, 13]
        pred_end = pred[:, end_indices, :].reshape(-1, pred.shape[-1])  # [16, 4, 204]
        gt_end = gt[:, end_indices].reshape(-1)        # [16, 4]
        loss_end = self.ce_loss(pred_end, gt_end)

        control_loss = loss_tb + loss_steer + 2.0 * loss_gear + loss_end

        return control_loss


class ControlValLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlValLoss, self).__init__()
        self.cfg = cfg
        self.pad_idx = self.cfg.token_nums - 1

        self.valid_token = self.cfg.token_nums - 4
        self.half_token = float(self.valid_token / 2)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.l1_loss = nn.SmoothL1Loss()

    def detokenize_acc(self, acc_token):
        if acc_token > self.half_token:
            acc = (acc_token / self.half_token - 1)
        else:
            acc = -(acc_token / self.half_token - 1)
        return acc

    def detokenize_steer(self, steer_token):
        steer = (steer_token / self.half_token) - 1
        return steer

    def forward(self, pred, data):
        pred_control = pred[:, :-2, :]
        pred_acc_token = pred_control[:, 0::3, :]
        pred_steer_token = pred_control[:, 1::3, :]
        pred_reverse_token = pred_control[:, 2::3, :]

        pred_acc_token = torch.softmax(pred_acc_token, dim=-1)
        pred_acc_token = pred_acc_token.argmax(dim=-1)
        pred_acc_token = pred_acc_token.reshape(-1).tolist()
        pred_acc = [self.detokenize_acc(x) for x in pred_acc_token]
        pred_acc = torch.from_numpy(np.array(pred_acc, dtype=np.float32)).cuda()
        gt_acc = data['gt_acc'].reshape(-1).cuda()
        acc_val_loss = self.l1_loss(pred_acc, gt_acc)

        pred_steer_token = torch.softmax(pred_steer_token, dim=-1)
        pred_steer_token = pred_steer_token.argmax(dim=-1)
        pred_steer_token = pred_steer_token.reshape(-1)
        pred_steer = self.detokenize_steer(pred_steer_token)
        gt_steer = data['gt_steer'].reshape(-1).cuda()
        steer_val_loss = self.l1_loss(pred_steer, gt_steer)

        acc_steer_val_loss = (acc_val_loss + steer_val_loss)

        pred_reverse_token = torch.softmax(pred_reverse_token, dim=-1)
        p_no_reverse = torch.sum(pred_reverse_token[:, :, :101], dim=-1).reshape(-1)
        p_reverse = torch.sum(pred_reverse_token[:, :, 101:], dim=-1).reshape(-1)
        pred_reverse = torch.stack((p_no_reverse, p_reverse), dim=0).T
        gt_reverse = data['gt_reverse'].reshape(-1).cuda()
        reverse_val_loss = self.ce_loss(pred_reverse, gt_reverse)

        return acc_steer_val_loss, reverse_val_loss
