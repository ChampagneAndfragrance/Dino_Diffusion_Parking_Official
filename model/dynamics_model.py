import torch
from torch import nn
import numpy as np
import math

class DynamicsModel(nn.Module):  # Fixed typo in nn.module -> nn.Module

    def __init__(self, hidden_dim=128, output_dim=2):
        '''
        This module is used to predict ego_motion
        Input: speed_x, speed_y, acc_x, acc_y, throttle, brake, steer, reverse, cos(yaw), sin(yaw))
        Output: delta_mean, log_var, displacement_x_world, displacement_y_world
        '''
        super().__init__()
        
        # Define the MLP model with LayerNorm
        self.mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Replace BatchNorm1d with LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.std_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, data, dt=0.1):
        # Extract x, y, and yaw
        ego_pos = data['ego_pos']  # [x, y, yaw]
        ego_motion = data['ego_motion'] # vel_x, vel_y, accel_x, accel_y
        x, y, yaw = ego_pos[:, 0], ego_pos[:, 1], ego_pos[:, 2]

        # Convert yaw to radians
        yaw = torch.deg2rad(yaw)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # Transform accelerations to the world frame
        accel_x_world = ego_motion[:, 3] * cos_yaw + ego_motion[:, 4] * sin_yaw
        accel_y_world = -ego_motion[:, 3] * sin_yaw + ego_motion[:, 4] * cos_yaw

        # Convert speed from km/h to m/s
        reverse = data['raw_control'][:, 3]

        # Recover vehicle velocity components in the world frame
        vehicle_velocity_x = data['ego_motion'][:,0] / 3.6
        vehicle_velocity_y = data['ego_motion'][:,1] / 3.6
        vehicle_velocity_z = data['ego_motion'][:,2] / 3.6
        speed = torch.sqrt(vehicle_velocity_x ** 2 + vehicle_velocity_y ** 2 + vehicle_velocity_z ** 2)

        # Compute displacements for reference only km/h -> m/s
        displacement_x_world_track = vehicle_velocity_x * dt + 0.5 * accel_x_world * dt**2
        displacement_y_world_track = vehicle_velocity_y * dt + 0.5 * accel_y_world * dt**2

        # torch.szie[B]
        throttle, brake, steer, reverse = data['raw_control'][:,0], data['raw_control'][:,1], data['raw_control'][:,2], data['raw_control'][:,3]
        # torch.size[B,3]
        cur_ego_pos = data['ego_pos']  # [x, y, yaw]
        dt_torch = torch.tensor(dt).to(throttle.device).unsqueeze(0).unsqueeze(0).expand(throttle.shape[0], 1)
        # Concatenate all inputs into a single tensor
        # inputs.shape = torch.size[26,3]
        inputs = torch.cat((speed.reshape(-1,1), 
                            accel_x_world.reshape(-1,1), 
                            accel_y_world.reshape(-1,1),
                            throttle.reshape(-1,1), 
                            brake.reshape(-1,1), 
                            steer.reshape(-1,1), 
                            reverse.reshape(-1,1), 
                            cos_yaw.reshape(-1,1), 
                            sin_yaw.reshape(-1,1),
                            dt_torch.reshape(-1,1)), dim=1).float()
        # Pass through the MLP
        feat = self.mlp(inputs)
        delta_mean = self.mean_head(feat)
        logvar = torch.exp(self.std_head(feat))  # std

        return delta_mean, logvar, displacement_x_world_track, displacement_y_world_track