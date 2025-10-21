import torch
import random
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset.carla_dataset import CarlaDataset
from tool.config import Configuration

from functools import partial


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def diffusion_collate_fn(batch, seq_len, interval, aggregate, collate_option="global"):
    """
    Batch: list of dicts from Dataset __getitem__
    """
    batch_dict = {}

    # Keys you want to resize
    traj_keys = [
        'gt_target_point_traj',
        'ego_trans_traj',
    ]

    unneccessary_keys = [        
        'gt_control_traj',
        'gt_acc_traj',
        'gt_steer_traj',
        'gt_reverse_traj'
        ]

    for key in traj_keys:
        resized = [resize_trajectory(item[key], seq_len, mode=collate_option, interval=interval, aggregate=aggregate, angle_col=2, angle_unit="deg") for item in batch]  # List of [fixed_steps, D]
        batch_dict[key] = torch.stack(resized, dim=0)  # → [B, fixed_steps, D]

    # Copy over all other keys directly (not resized)
    for key in batch[0]:
        if key not in traj_keys and key not in unneccessary_keys:
            batch_dict[key] = torch.utils.data.default_collate([item[key] for item in batch])

    return batch_dict

def _block_reduce(tensor, interval, aggregate="last", angle_col=None, angle_unit="deg", keep_endpoints=True):
    """Collapse every `interval` steps into one row, optionally preserving first and last."""
    if interval <= 1:
        return tensor

    T, D = tensor.shape
    if keep_endpoints and T > 2:
        first = tensor[0:1]
        last = tensor[-1:]
        middle = tensor[1:-1]
        # Reduce middle part
        reduced_middle = []
        for start in range(0, middle.shape[0], interval):
            end = min(start + interval, middle.shape[0])
            block = middle[start:end]
            if aggregate == "last":
                reduced_middle.append(block[-1])
            elif aggregate == "mean":
                if angle_col is None:
                    reduced_middle.append(block.mean(dim=0))
                else:
                    m = block.mean(dim=0)
                    ang = block[:, angle_col]
                    if angle_unit == "deg":
                        ang_rad = torch.deg2rad(ang)
                    else:
                        ang_rad = ang
                    c = torch.cos(ang_rad).mean()
                    s = torch.sin(ang_rad).mean()
                    ang_mean = torch.atan2(s, c)
                    if angle_unit == "deg":
                        ang_mean = torch.rad2deg(ang_mean)
                    m[angle_col] = ang_mean
                    reduced_middle.append(m)
            else:
                raise ValueError(f"Unknown aggregate: {aggregate}")
        reduced_middle = torch.stack(reduced_middle, dim=0) if reduced_middle else torch.empty((0, D), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([first, reduced_middle, last], dim=0)
    else:
        # Original reduce without keeping endpoints
        out = []
        for start in range(0, T, interval):
            end = min(start + interval, T)
            block = tensor[start:end]
            if aggregate == "last":
                out.append(block[-1])
            elif aggregate == "mean":
                if angle_col is None:
                    out.append(block.mean(dim=0))
                else:
                    m = block.mean(dim=0)
                    ang = block[:, angle_col]
                    if angle_unit == "deg":
                        ang_rad = torch.deg2rad(ang)
                    else:
                        ang_rad = ang
                    c = torch.cos(ang_rad).mean()
                    s = torch.sin(ang_rad).mean()
                    ang_mean = torch.atan2(s, c)
                    if angle_unit == "deg":
                        ang_mean = torch.rad2deg(ang_mean)
                    m[angle_col] = ang_mean
                    out.append(m)
            else:
                raise ValueError(f"Unknown aggregate: {aggregate}")
        return torch.stack(out, dim=0)


def resize_trajectory(tensor, fixed_steps, mode="global",
                      interval=1, aggregate="last", angle_col=None, angle_unit="deg"):
    """
    Resize a (T, D) tensor to (fixed_steps, D).

    interval: collapse every `interval` timesteps into one before resizing.
              aggregate='last' (default) or 'mean' (circular mean for angle_col).

    Modes:
    1. "global":
        - T == fixed_steps → return as is
        - T > fixed_steps  → uniform sample along FULL traj
        - T < fixed_steps  → pad with last
    2. "global_interp":
        - T >= fixed_steps → interpolate fixed_steps along FULL traj
        - T < fixed_steps  → interpolate T along FULL traj, then pad with last
    3. "local":
        - T == fixed_steps → return as is
        - T > fixed_steps  → take the NEXT fixed_steps (slice)
        - T < fixed_steps  → pad with last
    4. "local_interp":
        - T >= fixed_steps → interpolate fixed_steps along NEXT fixed_steps segment
        - T < fixed_steps  → interpolate T along FULL traj, then pad with last
    """
    # 1) collapse blocks first
    tensor = _block_reduce(tensor, interval, aggregate, angle_col, angle_unit)
    T, D = tensor.shape

    if mode == "global":
        if T == fixed_steps:
            return tensor
        elif T > fixed_steps:
            idx = np.linspace(0, T - 1, fixed_steps, dtype=int)
            return tensor[idx]
        else:
            padding = tensor[-1:].repeat(fixed_steps - T, 1)
            return torch.cat([tensor, padding], dim=0)

    elif mode == "global_interp":
        x_old = np.linspace(0, 1, T)
        if T >= fixed_steps:
            x_new = np.linspace(0, 1, fixed_steps)
            interp = np.vstack([np.interp(x_new, x_old, tensor[:, d].detach().cpu().numpy()) for d in range(D)]).T
            return torch.tensor(interp, dtype=tensor.dtype, device=tensor.device)
        else:
            x_new = np.linspace(0, 1, T)
            interp = np.vstack([np.interp(x_new, x_old, tensor[:, d].detach().cpu().numpy()) for d in range(D)]).T
            interp_t = torch.tensor(interp, dtype=tensor.dtype, device=tensor.device)
            padding = interp_t[-1:].repeat(fixed_steps - T, 1)
            return torch.cat([interp_t, padding], dim=0)

    elif mode == "local":
        if T == fixed_steps:
            return tensor
        elif T > fixed_steps:
            return tensor[:fixed_steps]
        else:
            padding = tensor[-1:].repeat(fixed_steps - T, 1)
            return torch.cat([tensor, padding], dim=0)

    elif mode == "local_interp":
        if T >= fixed_steps:
            segment_len = min(fixed_steps, T)
            x_old = np.linspace(0, 1, segment_len)
            x_new = np.linspace(0, 1, fixed_steps)
            interp = np.vstack([np.interp(x_new, x_old, tensor[:segment_len, d].detach().cpu().numpy()) for d in range(D)]).T
            return torch.tensor(interp, dtype=tensor.dtype, device=tensor.device)
        else:
            x_old = np.linspace(0, 1, T)
            x_new = np.linspace(0, 1, T)
            interp = np.vstack([np.interp(x_new, x_old, tensor[:, d].detach().cpu().numpy()) for d in range(D)]).T
            interp_t = torch.tensor(interp, dtype=tensor.dtype, device=tensor.device)
            padding = interp_t[-1:].repeat(fixed_steps - T, 1)
            return torch.cat([interp_t, padding], dim=0)

    else:
        raise ValueError(f"Unknown mode: {mode}")
        

class ParkingDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.data_dir = self.cfg.data_dir
        self.train_loader = None
        self.val_loader = None

    def setup(self, stage: str):
        data_root = self.data_dir
        train_set = CarlaDataset(data_root, 1, self.cfg)
        val_set = CarlaDataset(data_root, 0, self.cfg)
        diffusion_collate = partial(diffusion_collate_fn, seq_len=128 if "global" in self.cfg.planner_type else 16, interval=self.cfg.data_interval, aggregate=self.cfg.data_aggregate_type, collate_option=self.cfg.planner_type)
        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=self.cfg.batch_size,
                                       shuffle=True,
                                       num_workers=16,
                                       pin_memory=True,
                                       worker_init_fn=seed_worker,
                                       drop_last=True,
                                       collate_fn=diffusion_collate)
        self.val_loader = DataLoader(dataset=val_set,
                                     batch_size=self.cfg.batch_size,
                                     shuffle=False,
                                     num_workers=16,
                                     pin_memory=True,
                                     worker_init_fn=seed_worker,
                                     drop_last=True,
                                     collate_fn=diffusion_collate)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
