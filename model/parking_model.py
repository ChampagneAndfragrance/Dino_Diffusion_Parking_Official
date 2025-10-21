import torch
from torch import nn

from tool.config import Configuration
from model.bev_model import BevModel
from model.bev_encoder import BevEncoder
from model.feature_fusion import FeatureFusion
from diffuser.models.diffusion import GaussianDiffusion
from model.segmentation_head import SegmentationHead
from model.film_modulator import FiLMModulator

import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

import numpy as np
from diffuser.utils.visualizer import plot_trajectory_with_yaw, invert_trajectory_2D, deg2rad_trajectory_2D

class ParkingModelDiffusion(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        self.bev_model = BevModel(self.cfg)

        self.bev_encoder = BevEncoder(self.cfg.bev_encoder_in_channel)

        self.feature_fusion = FeatureFusion(self.cfg)

        self.trajectory_predict = GaussianDiffusion(self.cfg)

        self.film_modulate = FiLMModulator(self.cfg)

        self.segmentation_head = SegmentationHead(self.cfg)
        
    def adjust_target_bev(self, bev_feature, target_point):
        b, c, h, w = bev_feature.shape
        bev_target = torch.zeros((b, 1, h, w), dtype=torch.float).to(self.cfg.device, non_blocking=True)

        x_pixel = (h / 2 + target_point[:, 0] / self.cfg.bev_x_bound[2]).unsqueeze(0).T.int()
        y_pixel = (w / 2 + target_point[:, 1] / self.cfg.bev_y_bound[2]).unsqueeze(0).T.int()
        target_point = torch.cat([x_pixel, y_pixel], dim=1)

        noise = (torch.rand_like(target_point, dtype=torch.float) * 10 - 5).int()
        target_point += noise
        for batch in range(b):
            bev_target_batch = bev_target[batch][0]
            target_point_batch = target_point[batch]
            bev_target_batch[target_point_batch[0] - 4:target_point_batch[0] + 4,
            target_point_batch[1] - 4:target_point_batch[1] + 4] = 1.0
        return bev_feature, bev_target

    def encoder(self, data):

        # INFO: Extract sensor data - images, camera parameters, tracked target
        images = data['image'].to(self.cfg.device, non_blocking=True)
        intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
        extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
        target_point = data['target_point'].to(self.cfg.device, non_blocking=True) # in car ego frame

        # INFO: Get the BEV feature and the estimated depth from the BEV model
        bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics)

        # INFO: Downsample the BEV
        bev_target = self.adjust_target_bev(bev_feature, target_point)
        bev_down_sample = self.bev_encoder(bev_feature)

        # INFO: Fuse the target information into the BEV
        target_point = target_point.unsqueeze(1)
        fuse_feature = self.feature_fusion(bev_down_sample, target_point)

        # INFO: Filter the BEV with the FiLM layer conditioned on the target point
        filmed_fuse_feature = self.film_modulate(fuse_feature, target_point)

        # INFO: Use BEV+Seg feature or BEV feature only for the downstreaming tasks
        pred_segmentation = self.segmentation_head(filmed_fuse_feature)
        # Step 1: Downsample segmentation to 16x16
        seg_down = F.interpolate(pred_segmentation, size=(16, 16), mode='bilinear', align_corners=False)  # [1, 3, 16, 16]
        # Step 2: Rearrange segmentation to [1, 256, 3]
        seg_down_flat = seg_down.permute(0, 2, 3, 1).reshape(pred_segmentation.shape[0], filmed_fuse_feature.shape[1], 3)  # [1, 256, 3]
        # Step 3: Concatenate along the feature dimension
        concat_feature = torch.cat([filmed_fuse_feature, seg_down_flat], dim=-1)  # [1, 256, 267]

        return concat_feature, pred_segmentation, pred_depth, bev_target

    def forward(self, data):
        """ This function is used in the inference """

        # INFO: Generate the fused BEV, segmentation and depth from the perception modules
        fuse_feature, pred_segmentation, pred_depth, _ = self.encoder(data)

        # INFO: Use ego centric trajectory or trajectory in the fixed frame?
        if self.cfg.ego_centric_traj:
            gt_target_point_traj = self.world_to_ego0(data["ego_trans_traj"])
        else:
            gt_target_point_traj = data["gt_target_point_traj"]

        # INFO: Do we use the normalized coordinates for the trajectory or not?
        if self.cfg.normalize_traj:
            gt_target_point_traj = self.normalize_trajectories(gt_target_point_traj, device = "cuda")
            target_point = self.normalize_trajectories(data["target_point"], device = "cuda")
        else:
            gt_target_point_traj = gt_target_point_traj
            target_point = target_point

        fuse_feature.requires_grad_(True)

        # INFO: Obtain the start (current pose) and the end (target pose) of the parking trajectory
        if "global" in self.cfg.planner_type:
            start_end_relative_point = torch.cat((gt_target_point_traj[:,0:1,:], gt_target_point_traj[:,-1:,:]), dim=1)
        else:
            start_end_relative_point = gt_target_point_traj[:,0:1,:]

        # INFO: Do we use the fused bev + segmentation (embedding) or segmentation itself (segmentation)?
        if self.cfg.motion_head == "embedding":
            seg_egoMotion_tgtPose = {"pred_segmentation": fuse_feature, "target_point": target_point}
        elif self.cfg.motion_head == "segmentation":
            seg_egoMotion_tgtPose = {"pred_segmentation": pred_segmentation, "target_point": target_point}
        else:
            pass

        # INFO: Derive the parking trajectory in the future steps
        pred_control = self.trajectory_predict(seg_egoMotion_tgtPose, start_end_relative_point)

        return pred_control, pred_segmentation, pred_depth, fuse_feature

    def diffusion_loss(self, data):
        """ This loss is used to train the diffusion trajectory generator """

        # INFO: Generate the fused BEV, segmentation and depth from the perception modules
        fuse_feature, pred_segmentation, pred_depth, _ = self.encoder(data)

        # INFO: Use ego centric trajectory or trajectory in the fixed frame?
        if self.cfg.ego_centric_traj:
            gt_target_point_traj = self.world_to_ego0(data["ego_trans_traj"])
        else:
            gt_target_point_traj = data["gt_target_point_traj"]

        # INFO: Do we use the normalized coordinates for the trajectory or not?
        if self.cfg.normalize_traj:
            gt_target_point_traj = self.normalize_trajectories(gt_target_point_traj, device = "cuda")
            target_point = self.normalize_trajectories(data["target_point"], device = "cuda")
        else:
            gt_target_point_traj = gt_target_point_traj
            target_point = target_point

        # INFO: Obtain the start (current pose) and the end (target pose) of the parking trajectory
        if "global" in self.cfg.planner_type:
            start_end_relative_point = torch.cat((gt_target_point_traj[:,0:1,:], gt_target_point_traj[:,-1:,:]), dim=1)
        else:
            start_end_relative_point = gt_target_point_traj[:,0:1,:]
            
        # INFO: Do we use the fused bev + segmentation (embedding) or segmentation itself (segmentation)?
        if self.cfg.motion_head == "embedding":
            seg_egoMotion_tgtPose = {"pred_segmentation": fuse_feature, "target_point": target_point}
        elif self.cfg.motion_head == "segmentation":
            seg_egoMotion_tgtPose = {"pred_segmentation": pred_segmentation, "target_point": target_point}
        else:
            pass

        # INFO: Calculate the diffusion loss from the ground truth trajectory and predicted trajectory 
        loss = self.trajectory_predict.loss(gt_target_point_traj, seg_egoMotion_tgtPose, start_end_relative_point)[0]
        return loss
        
    def predict(self, data, final_steps = 0):

        # INFO: Generate the fused BEV, segmentation and depth from the perception modules
        fuse_feature, pred_segmentation, pred_depth, bev_target = self.encoder(data)

        # INFO: Do we use the normalized coordinates for the trajectory or not?
        if self.cfg.normalize_traj:
            target_point = self.normalize_trajectories(data["target_point"], device = "cpu")
        else:
            target_point = target_point

        fuse_feature.requires_grad_(True)

        # INFO: Use ego centric trajectory or trajectory in the fixed frame?
        if self.cfg.ego_centric_traj:
            end_relative_point = target_point.unsqueeze(1)
            if self.cfg.normalize_traj:
                end_relative_point = (end_relative_point)
            start_relative_point = torch.zeros_like(end_relative_point)
        else:
            start_relative_point = target_point.unsqueeze(1)
            if self.cfg.normalize_traj:
                start_relative_point = (start_relative_point)
            end_relative_point = torch.zeros_like(start_relative_point)

        # INFO: Obtain the start (current pose) and the end (target pose) of the parking trajectory
        if "global" in self.cfg.planner_type:
            start_end_relative_point = torch.cat((start_relative_point, end_relative_point), dim=1)
        else:
            start_end_relative_point = start_relative_point

        # INFO: Do we use the fused bev + segmentation (embedding) or segmentation itself (segmentation)?
        if self.cfg.motion_head == "embedding":
            seg_egoMotion_tgtPose = {"pred_segmentation": fuse_feature, "target_point": target_point}
        elif self.cfg.motion_head == "segmentation":
            seg_egoMotion_tgtPose = {"pred_segmentation": pred_segmentation, "target_point": target_point}
        else:
            pass        
        
        # INFO: Derive the (normalized) parking trajectory in the future steps
        pred_traj = self.trajectory_predict(seg_egoMotion_tgtPose, start_end_relative_point)

        # INFO: Denormalize the trajectory if it has been normalized before
        if self.cfg.normalize_traj:
            pred_traj = self.denormalize_target_point(pred_traj, device="cuda")

        # INFO: Make sure the trajectory is ego-centric as output
        if self.cfg.ego_centric_traj:
            pred_traj = pred_traj.squeeze(0)
        else:
            pred_traj = self.world_to_ego0(pred_traj)

        return pred_traj, pred_segmentation, pred_depth, bev_target

    def normalize_trajectories(self, traj, device = "cpu"):
        """ Normalize the trajectory in case the neural network cannot learn large values """
        normalized_traj = traj / torch.tensor([[10.0, 10.0, 180.0]], device=device)
        return normalized_traj

    def denormalize_target_point(self, traj, device):
        """ Denormalize the trajectories """
        if device == "cuda":
            traj = traj * torch.Tensor([[10.0, 10.0, 180.0]]).cuda()
        elif device == "cpu":
            traj = traj * torch.Tensor([[10.0, 10.0, 180.0]])
        else:
            pass
        return traj

    def world_to_ego0(self, ego_trans_world):
        """
        ego_trans_world: [B, T, 3] -> (x_w, y_w, yaw_deg_w)
        returns:         [B, T, 3] -> (x_0, y_0, yaw_deg_rel) in ego_0 frame
        """
        def wrap_deg(d):
            # wrap to (-180, 180]
            return (d + 180.0) % 360.0 - 180.0

        xw  = ego_trans_world[..., 0]
        yw  = ego_trans_world[..., 1]
        yaw = ego_trans_world[..., 2]          # degrees

        x0   = xw[..., :1]                      # [B, 1]
        y0   = yw[..., :1]                      # [B, 1]
        yaw0 = yaw[..., :1]                     # [B, 1]

        th0 = torch.deg2rad(yaw0)
        c0, s0 = torch.cos(th0), torch.sin(th0)

        # translate then rotate by -yaw0
        dx = xw - x0
        dy = yw - y0
        x_rel =  c0 * dx + s0 * dy
        y_rel = -s0 * dx + c0 * dy

        yaw_rel = wrap_deg(yaw - yaw0)

        return torch.stack([x_rel, y_rel, yaw_rel], dim=-1)





