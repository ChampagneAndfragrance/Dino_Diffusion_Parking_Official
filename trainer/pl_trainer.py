import torch
import pytorch_lightning as pl
import matplotlib as mpl
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor, ModelSummary
from tool.config import Configuration
from loss.control_loss import ControlLoss, ControlValLoss
from loss.waypoint_loss import WaypointLoss
from loss.depth_loss import DepthLoss
from loss.seg_loss import SegmentationLoss
from model.parking_model import ParkingModelDiffusion
import torch.nn.functional as F
import shutil
import os


def setup_callbacks(cfg):
    callbacks = []

    ckpt_callback = ModelCheckpoint(dirpath=cfg.checkpoint_dir,
                                    monitor='val_loss',
                                    save_top_k=40,
                                    mode='min',
                                    filename='E2EParking-{epoch:02d}-{val_loss:.2f}',
                                    save_last=True)
    callbacks.append(ckpt_callback)

    progress_bar = TQDMProgressBar()
    callbacks.append(progress_bar)

    model_summary = ModelSummary(max_depth=2)
    callbacks.append(model_summary)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    return callbacks

def freeze_module(module):
    """ Freeze module parameters """
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module(module):
    """ Unfreeze module parameters """
    for param in module.parameters():
        param.requires_grad = True

class ParkingTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Configuration, model_path=None):
        super(ParkingTrainingModule, self).__init__()

        self.save_hyperparameters(ignore=['model_path'])

        self.cfg = cfg

        self.control_loss_func = ControlLoss(self.cfg)

        self.waypoint_loss_func = WaypointLoss(self.cfg)

        self.control_val_loss_func = ControlValLoss(self.cfg)

        self.segmentation_loss_func = SegmentationLoss(
            class_weights=torch.Tensor(self.cfg.seg_vehicle_weights)
        )

        self.depth_loss_func = DepthLoss(self.cfg)

        self.parking_model = ParkingModelDiffusion(self.cfg)

        self.perception_training_steps = 15

    def on_train_start(self):
        """ Save the config files to the log folder at the begining of the training"""

        config_path = "./config/dino_training.yaml"
        checkpoint_dir = self.cfg.checkpoint_dir

        if self.trainer.is_global_zero and os.path.isfile(config_path):
            os.makedirs(checkpoint_dir, exist_ok=True)  # create dir if not exists

            dst_path = os.path.join(checkpoint_dir, os.path.basename(config_path))
            shutil.copyfile(config_path, dst_path)  # overwrite if exists

    def on_train_epoch_start(self):
        """ Decide if we are using the augmented data at the begining of each training epoch """

        dataloader = self.trainer.datamodule.train_dataloader()
        dataset = dataloader.dataset

        # INFO: Train the perception module until epoch self.perception_training_steps
        if self.current_epoch < self.perception_training_steps:
            if self.current_epoch % 3 != 0: # augment with target relabeling
                if hasattr(dataset, 'relabel_goals'):
                    dataset.relabel_goals(self.current_epoch)
                    print("Training with relabeled target.")
            else: # keep the original target
                dataset.keep_goals(self.current_epoch)
                print("Training with original target.")

        # INFO: Train the parking trajectory generator after epoch self.perception_training_steps - freeze the perception module
        else:
            dataset.keep_goals(self.current_epoch)
            print("Train the trajectory generator.")
            if self.cfg.motion_head == "segmentation":
                freeze_module(self.parking_model.bev_model)
                freeze_module(self.parking_model.bev_encoder)
                freeze_module(self.parking_model.feature_fusion)
                freeze_module(self.parking_model.film_modulate)
                freeze_module(self.parking_model.segmentation_head)
            else:
                freeze_module(self.parking_model.bev_model)
                freeze_module(self.parking_model.bev_encoder)
                freeze_module(self.parking_model.feature_fusion)
                freeze_module(self.parking_model.film_modulate)
                freeze_module(self.parking_model.segmentation_head)

    def training_step(self, batch, batch_idx):

        loss_dict = {}

        # INFO: Stage 1 - Train the perception moduels
        # pred_segmentation, pred_depth, fuse_feature = self.parking_model(batch)

        # segmentation_loss = self.segmentation_loss_func(pred_segmentation.unsqueeze(1), batch['segmentation'])
        # loss_dict.update({
        #     "segmentation_loss": segmentation_loss
        # })

        # depth_loss = self.depth_loss_func(pred_depth, batch['depth'])
        # loss_dict.update({
        #     "depth_loss": depth_loss
        # }) 

        # INFO: Stage 2 - Train the parking diffusion      
        diffusion_loss = self.parking_model.diffusion_loss(batch)
        loss_dict.update({
            "diffusion_loss": diffusion_loss*0.0 if self.current_epoch < self.perception_training_steps else diffusion_loss
        })

        # INFO: Sum losses as the training loss - segmentation_loss+depth_loss for stage 1 and diffusion_loss for stage 2
        train_loss = sum(loss_dict.values())
        loss_dict.update({
            "train_loss": train_loss
        })

        # INFO: Add the loss you want to visualize here
        self.log_dict(loss_dict)

        return train_loss

    def validation_step(self, batch, batch_idx):

        val_loss_dict = {}

        # INFO: Stage 1 - Validate the perception moduels
        # with torch.enable_grad():
        #     pred_segmentation, pred_depth, fuse_feature = self.parking_model(batch)
        #     segmentation_val_loss = self.segmentation_loss_func(pred_segmentation.unsqueeze(1), batch['segmentation'])
        #     val_loss_dict.update({
        #         "segmentation_val_loss": segmentation_val_loss
        #     })

        #     depth_val_loss = self.depth_loss_func(pred_depth, batch['depth'])
        #     val_loss_dict.update({
        #         "depth_val_loss": depth_val_loss
        #     })

        # INFO: Stage 2 - Validate the parking diffusion
        with torch.enable_grad():
            diffusion_loss = self.parking_model.diffusion_loss(batch)
            val_loss_dict.update({
                "diffusion_loss": diffusion_loss
            })

        # INFO: Sum losses as the training loss - segmentation_loss+depth_loss for stage 1 and control_loss for stage 2
        val_loss = sum(val_loss_dict.values())
        val_loss_dict.update({
            "val_loss": val_loss
        })

        # INFO: Add the loss you want to visualize here
        self.log_dict(val_loss_dict)

        return val_loss

    def configure_optimizers(self):
    
        # INFO: Set learning rates and weight decays for the perception and policy learning
        base_lr = self.cfg.learning_rate
        perception_lr = self.cfg.learning_rate
        weight_decay = self.cfg.weight_decay

        # INFO: Set the parameter groups - perception parameters and non-perception parameters (parking policy paras)
        named_params = list(self.named_parameters())

        perception_params = (p for n, p in named_params 
        if n.startswith("parking_model.bev_model") or n.startswith("parking_model.bev_encoder") 
        or n.startswith("parking_model.feature_fusion") or n.startswith("parking_model.film_modulate") 
        or n.startswith("parking_model.segmentation_head"))

        other_params = (
            p for n, p in named_params
            if not (n.startswith("parking_model.bev_model") or n.startswith("parking_model.bev_encoder") 
            or n.startswith("parking_model.feature_fusion") or n.startswith("parking_model.film_modulate") 
            or n.startswith("parking_model.segmentation_head"))
        )

        # INFO: Set learning rate for parameter groups
        param_groups = [
            {"params": perception_params, "lr": perception_lr, "weight_decay": weight_decay},
            {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
        ]

        # INFO: Optimizer
        optimizer = torch.optim.Adam(param_groups)

        # INFO: Scheduler (applies to both groups uniformly)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.cfg.epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
