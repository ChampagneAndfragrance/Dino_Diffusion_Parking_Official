import os
import sys
import argparse
import yaml
import sys
print(sys.path)
from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from trainer.pl_trainer import ParkingTrainingModule, setup_callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.dataloader import ParkingDataModule
from tool.config import get_cfg
import torch

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# INFO: Train with multiple GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def train():
    arg_parser = argparse.ArgumentParser(description='ParkingModel')
    arg_parser.add_argument(
        '--config',
        default='./config/training.yaml',
        type=str,
        help='path to training.yaml (default: ./config/training.yaml)')
    arg_parser.add_argument(
        '--model_path',
        default=None,
        help='path to model.ckpt')
    args = arg_parser.parse_args()

    with open(args.config, 'r') as yaml_file:
        try:
            cfg_yaml = yaml.safe_load(yaml_file)
        except yaml.YAMLError:
            logger.exception("Open {} failed!", args.config)
    cfg = get_cfg(cfg_yaml)
    cfg.model_path = args.model_path
    
    logger.remove()
    logger.add(cfg.log_dir + '/training_{time}.log', enqueue=True, backtrace=True, diagnose=True)
    logger.add(sys.stderr, enqueue=True)
    logger.info("Config Yaml File: {}", args.config)

    seed_everything(42)

    parking_callbacks = setup_callbacks(cfg)
    tensor_logger = TensorBoardLogger(save_dir=cfg.log_dir, default_hp_metric=False)
    num_gpus = 1

    torch.set_float32_matmul_precision('medium')

    parking_model = ParkingTrainingModule(cfg,model_path=cfg.model_path)
    parking_datamodule = ParkingDataModule(cfg)
    parking_trainer = Trainer(callbacks=parking_callbacks,
                              logger=tensor_logger,
                              accelerator='gpu',
                              strategy='ddp' if num_gpus > 1 else None,
                              devices=num_gpus,
                              max_epochs=cfg.epochs,
                              log_every_n_steps=cfg.log_every_n_steps,
                              check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                              profiler='simple')

    # INFO: Set the start training epochs: The first stage is to train the perception model which should be loaded from the second stage which trains parking policy
    parking_trainer.fit_loop.epoch_progress.current.completed = 15
    
    # INFO: Load the checkpoint parameters
    ckpt = torch.load(cfg.model_path, map_location='cpu')

    # INFO: Get only the model weights
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    # INFO: Filter out the trajectory_predict model and only obtain the parameters of the perception modules
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith('parking_model.trajectory_predict')
    }

    # INFO: Load the filtered parameters to the perception modules
    missing, unexpected = parking_model.load_state_dict(filtered_state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # INFO: Start training
    parking_trainer.fit(
        parking_model, 
        datamodule=parking_datamodule,
        ckpt_path= None 
    )

if __name__ == '__main__':
    train()