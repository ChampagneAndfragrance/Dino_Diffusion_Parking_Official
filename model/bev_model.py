import torch

from torch import nn
from model.cam_encoder import CamEncoder, DinoCamEncoder
from tool.config import Configuration
from tool.geometry import VoxelsSumming, calculate_birds_eye_view_parameters
from torchvision.transforms import Resize

class BevModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        # INFO: Lift-Splat-Shoot
        bev_res, bev_start_pos, bev_dim = calculate_birds_eye_view_parameters(self.cfg.bev_x_bound,
                                                                              self.cfg.bev_y_bound,
                                                                              self.cfg.bev_z_bound)
        self.bev_res = nn.Parameter(bev_res, requires_grad=False)
        self.bev_start_pos = nn.Parameter(bev_start_pos, requires_grad=False)
        self.bev_dim = nn.Parameter(bev_dim, requires_grad=False)

        self.down_sample = self.cfg.bev_down_sample

        self.frustum = self.create_frustum()
        self.depth_channel, _, _, _ = self.frustum.shape
        
        # INFO: Now we are using DINO V2 as our image processing backbone
        if "efficient" in cfg.backbone:
            self.cam_encoder = CamEncoder(self.cfg, self.depth_channel)
        elif "dino" in cfg.backbone:
            self.cam_encoder = DinoCamEncoder(self.cfg, self.depth_channel)
        else:
            raise NotImplementedError

    def create_frustum(self):

        # INFO: Get the final image height and width from configuration
        h, w = self.cfg.final_dim

        # INFO: Compute the downsampled height and width based on the down_sample factor
        down_sample_h, down_sample_w = h // self.down_sample, w // self.down_sample

        # INFO: Create a depth grid using the specified depth bounds (start, end, step)
        # Shape: [D, 1, 1] -> broadcast to [D, H, W]
        depth_grid = torch.arange(*self.cfg.d_bound, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, down_sample_h, down_sample_w)
        depth_slice = depth_grid.shape[0]  # Number of depth slices (D)

        # INFO: Create an x-coordinate grid ranging from 0 to w-1 and broadcast to [D, H, W]
        x_grid = torch.linspace(0, w - 1, down_sample_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, down_sample_w).expand(depth_slice, down_sample_h, down_sample_w)

        # INFO: Create a y-coordinate grid ranging from 0 to h-1 and broadcast to [D, H, W]
        y_grid = torch.linspace(0, h - 1, down_sample_h, dtype=torch.float)
        y_grid = y_grid.view(1, down_sample_h, 1).expand(depth_slice, down_sample_h, down_sample_w)

        # INFO: Stack x, y, and depth to create a frustum grid of shape [D, H, W, 3]
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)

        # INFO: Return the frustum as a non-trainable parameter
        return nn.Parameter(frustum, requires_grad=False)


    def get_geometry(self, intrinsics, extrinsics):

        # self.frustum dimension is a stack of x_grid, y_grid, depth_grid

        # INFO: Invert extrinsics to convert from world-to-camera to camera-to-world transformation
        extrinsics = torch.inverse(extrinsics).cuda()

        # INFO: Extract rotation matrix (3x3) and translation vector (3,) from the extrinsics
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]

        # INFO: Get batch size (b) and number of cameras (n)
        b, n, _ = translation.shape

        # INFO: Expand frustum to match the batch and camera dimensions
        # Shape becomes [1, 1, D, H, W, 3, 1] assuming frustum is [D, H, W, 3]
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # INFO: Convert normalized frustum coordinates into image plane coordinates:
        # - x and y are scaled by depth (z) to get coordinates in the image plane
        # - we keep z as is to maintain 3D coordinates in the camera frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]), 5)

        # INFO: Compute combined camera-to-world transform: R * K^-1
        # - K: camera intrinsics
        # - R: rotation matrix from extrinsics
        combine_transform = rotation.matmul(torch.inverse(intrinsics)).cuda()

        # INFO: Apply the transformation to get points in the camera frame
        # Shape: [b, n, D, H, W, 3, 1] → squeeze to [b, n, D, H, W, 3]
        points = combine_transform.view(b, n, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)

        # INFO: Add the translation to get points in the world frame
        points += translation.view(b, n, 1, 1, 1, 3)

        # INFO: Return the 3D points in world coordinates for each frustum grid location
        return points


    def encoder_forward(self, images):
        """ Use DINO features in the depth estimation and downstreaming tasks """

        b, n, c, h, w = images.shape
        images = images.view(b * n, c, h, w)
        # INFO: Obtain the DINO embeddings and the depth embeddings (also from DINO)
        x, depth = self.cam_encoder(images)

        # INFO: Obtain the depth distribution
        depth_prob = depth.softmax(dim=1)
        if self.cfg.use_depth_distribution:
            x = depth_prob.unsqueeze(1) * x.unsqueeze(2) 
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.depth_channel, 1, 1)

        # INFO: Rearrange the dimensions
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2) # x dimensions: [b, n, depth_bin, h, w, feature_channel]
        return x, depth_prob

    def proj_bev_feature(self, geom, image_feature):
        """ This function is used to generate the BEV features with the classical Splat-Shoot technique """

        # INFO: Unpack the dimensions
        batch, n, d, h, w, c = image_feature.shape  # batch size, number of views, depth, height, width, channels

        # INFO: Initialize the output BEV feature map (shape: B x C x H_bev x W_bev)
        output = torch.zeros((batch, c, self.bev_dim[0], self.bev_dim[1]), dtype=torch.float, device=image_feature.device)

        # INFO: Total number of 3D points per sample = n * d * h * w
        N = n * d * h * w

        # INFO: Process each sample in the batch
        for b in range(batch):
            image_feature_b = image_feature[b]  # (n, d, h, w, c)
            geom_b = geom[b]                    # (n, d, h, w, 3)

            # INFO: Flatten the image features to a 2D tensor: (N, c)
            x_b = image_feature_b.reshape(N, c)

            # INFO: Normalize the geometry to BEV grid coordinates
            # geom_b becomes (N, 3) representing x, y, z in the BEV grid
            geom_b = ((geom_b - (self.bev_start_pos - self.bev_res / 2.0)) / self.bev_res)
            geom_b = geom_b.view(N, 3).long()  # Convert to integer grid positions

            # INFO: Mask out points that fall outside the BEV grid dimensions
            mask = ((geom_b[:, 0] >= 0) & (geom_b[:, 0] < self.bev_dim[0])
                    & (geom_b[:, 1] >= 0) & (geom_b[:, 1] < self.bev_dim[1])
                    & (geom_b[:, 2] >= 0) & (geom_b[:, 2] < self.bev_dim[2]))
            x_b = x_b[mask]          # Filter valid image features
            geom_b = geom_b[mask]    # Filter corresponding grid locations

            # INFO: Compute a linear "rank" index for voxel sorting (for later summing)
            ranks = ((geom_b[:, 0] * (self.bev_dim[1] * self.bev_dim[2])
                    + geom_b[:, 1] * self.bev_dim[2]) + geom_b[:, 2])
            sorts = ranks.argsort()  # Sort ranks to prepare for grouped summing
            x_b, geom_b, ranks = x_b[sorts], geom_b[sorts], ranks[sorts]

            # INFO: Custom operation: sum features with the same voxel coordinates
            x_b, geom_b = VoxelsSumming.apply(x_b, geom_b, ranks)

            # INFO: Create empty BEV volume (z, x, y, c) to hold projected features
            bev_feature = torch.zeros((self.bev_dim[2], self.bev_dim[0], self.bev_dim[1], c),
                                    device=image_feature_b.device)

            # INFO: Fill BEV volume at projected 3D coordinates
            bev_feature[geom_b[:, 2], geom_b[:, 0], geom_b[:, 1]] = x_b

            # INFO: Rearrange to standard BEV layout: (c, x, y)
            tmp_bev_feature = bev_feature.permute((0, 3, 1, 2)).squeeze(0)

            # INFO: Store in output tensor
            output[b] = tmp_bev_feature

        return output


    def calc_bev_feature(self, images, intrinsics, extrinsics):
        # INFO: Get the vehicle surrounding geometries from intrinsic and extrinsic parameters
        geom = self.get_geometry(intrinsics, extrinsics)
        # INFO: Get the DINO embeddings
        x, pred_depth = self.encoder_forward(images)
        # INFO: Project DINO embeddings to BEV
        bev_feature = self.proj_bev_feature(geom, x)
        return bev_feature, pred_depth

    def forward(self, images, intrinsics, extrinsics):
        bev_feature, pred_depth = self.calc_bev_feature(images, intrinsics, extrinsics)
        return bev_feature.squeeze(1), pred_depth

    def get_intermidiate_layers(self, images, intrinsics, extrinsics):
        """ Fetch the intermidiate layers for visualization """

        geom = self.get_geometry(intrinsics, extrinsics)

        b, n, c, h, w = images.shape
        images = images.view(b * n, c, h, w)
        backbone_feature = self.cam_encoder.get_backbone_feature(images)
        cams_feature, depth = self.cam_encoder(images)

        depth_prob = depth.softmax(dim=1)
        if self.cfg.use_depth_distribution:
            depth_bin_feature = depth_prob.unsqueeze(1) * cams_feature.unsqueeze(2)
        else:
            depth_bin_feature = cams_feature.unsqueeze(2).repeat(1, 1, self.depth_channel, 1, 1)

        depth_bin_feature = depth_bin_feature.view(b, n, *depth_bin_feature.shape[1:])
        depth_bin_feature = depth_bin_feature.permute(0, 1, 3, 4, 5, 2) # x dimensions: [b, n, depth_bin, h, w, feature_channel]

        bev_feature = self.proj_bev_feature(geom, depth_bin_feature)

        return backbone_feature, cams_feature, depth, depth_bin_feature, bev_feature





