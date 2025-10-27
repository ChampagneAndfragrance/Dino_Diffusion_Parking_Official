# Dino-Diffusion Modular Designs Bridge the Cross-Domain Gap in Autonomous Parking

## Author

Zixuan Wu, Hengyuan Zhang, Ting-Hsuan Chen, Yuliang Guo, David Paz, Xinyu Huang, Liu Ren 

(Work done in [Bosch AI Silicon Valley](https://www.bosch-ai.com/))

## Introduction
This repository contains the code for our submitted conference paper.

This work proposes a learning modular based pipeline of the autonomous parking that can be trained with only a specific environment setting in CARLA but applied to multiple different domain settings and even the Gaussian-Splatting (GS) rendered real-world in a zero-shot fashion without additional data collection. The inputs are the images captured by surrounding cameras, while the output is an on-the-fly updated trajectory. We test the performance of our policies on both CARLA and real-world 3DGS settings.

## Setup

Clone the repo, setup CARLA 0.9.11, and build the conda environment:

```Shell
git clone https://github.com/qintonguav/e2e-parking-carla.git
cd e2e-parking-carla/
conda env create -f environment.yml
conda activate E2EParking
chmod +x setup_carla.sh
./setup_carla.sh
```
CUDA 11.7 is used as default. We also validate the compatibility of CUDA 10.2 and 11.3.

The python virtual environment packages we are using and the versions could be found at [requirements.txt](./requirements.txt).

## Evaluation (Inference with pre-trained model)
For inference, we prepare a [pre-trained dino-diffusion model](https://huggingface.co/ChampagneAndfragrance/Dino_Diffusion_Parking/resolve/main/last.ckpt?download=true) and a [dynamic model](https://huggingface.co/ChampagneAndfragrance/Dino_Diffusion_Parking/resolve/main/milestone/dynamic_control_speed.ckpt?download=true). The success rate of our model in the same-domain and cross-domain validation in CARLA can reach >90%.

The first step is to launch a CARLA server:

```Shell
./carla/CarlaUE4.sh -opengl
```

In a separate terminal, use the script below for trained model evaluation:
```Shell
python3 carla_parking_eva.py
```

The main variables to set for this script:
```
--model_path        -> path to model.ckpt
--eva_epochs        -> number of eva epochs (default: 4')
--eva_task_nums     -> number of evaluation task (default: 16')
--eva_parking_nums  -> number of parking nums for every slot (default: 6')
--eva_result_path   -> path to save evaluation result csv file
--shuffle_veh       -> shuffle static vehicles between tasks (default: True)
--shuffle_weather   -> shuffle weather between tasks (default: False)
--random_seed       -> random seed to initialize env (default: 0)
```

For example, we can use "args": 
```
"--model_config_path", "./config/dino_training.yaml"
"--model_path", "./ckpt/local_embedding/last.ckpt"
"--model_path_dynamic", "./milestone/dynamic_control_speed.ckpt"
"--show_eva_imgs", "True"
"--eva_parking_nums", "6"
"--eva_task_nums", "16"
"--eva_epochs", "1"
```

When the evaluation is completed, metrics will be saved to csv files located at '--eva_result_path'. Notice that we do not need to shuffle the weather with the parameter '--shuffle_weather', the weather will be changed in the file [carla_parking_eva.py](./carla_parking_eva.py) such that we can quantitavely tell what the domain it is now.

## Dataset and Training

### Training Data
We use the same dataset with the state-of-the-art work [End-to-End Visual Autonomous Parking via Control-Aided Attention](https://www.arxiv.org/abs/2509.11090?context=cs) and show better performance over it in the benchmarking. Since the whole dataset is extremely large (> 2 TB) so we upload part of our training data to [HuggingFace](https://huggingface.co/datasets/ChampagneAndfragrance/Dino_Diffusion_Parking/tree/main).

### Training script

The code for training is provided in [pl_train.py](./pl_train.py) \
A minimal example of running the training script on a single GPU:
```Shell
python pl_train.py 
```
To configure the training parameters, please refer to [dino_training.yaml](./config/dino_training.yaml), including training data path, number of epoch and checkpoint path.

For parallel training, modify the settings in [pl_train.py](./pl_train.py).

For instance, 8 GPU parallel training:
```
line 14: os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
line 42: num_gpus = 8
```

Notice: We first train the perception modules (freezing the DINO part) and then train the diffusion planning model (freezing the perception parts). You can check [pl_trainer.py](./trainer/pl_trainer.py) function 'on_train_epoch_start' for the training details.

## Video

We intuitively compare our dino-diffusion pipeline with the concurrent state-of-art method in the following video. It is clear that our method outperform baselines in the cross-domain tasks.

<p align="center">
  <img src="./resource/icra26_cross_domain_compressed.gif" alt="Dino_Diffusion Parking Demo" width="85%">
</p>


## Citation

If you think this work helps, please consider cite it with:
```
@misc{wu2025dinodiffusionmodulardesignsbridge,
      title={Dino-Diffusion Modular Designs Bridge the Cross-Domain Gap in Autonomous Parking}, 
      author={Zixuan Wu and Hengyuan Zhang and Ting-Hsuan Chen and Yuliang Guo and David Paz and Xinyu Huang and Liu Ren},
      year={2025},
      eprint={2510.20335},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.20335}, 
}
```

## Acknowledgement

We build our project upon the paper [E2E Parking: Autonomous Parking by the End-to-end Neural Network on the CARLA Simulator](resource/E2E_APA_IV24_final.pdf):
```
@inproceedings{E2EAPA,
	title={E2E Parking: Autonomous Parking by the End-to-end Neural Network on the CARLA Simulator},
	author={Yang, Yunfan and Chen, Denglong and Qin, Tong and Mu, Xiangru and Xu, Chunjing and Yang, Ming},
	booktitle={Conference on IEEE Intelligent Vehicles Symposium},
	year={2024}
}
```
and [End-to-End Visual Autonomous Parking via Control-Aided Attention](https://www.arxiv.org/abs/2509.11090?context=cs):
```
@misc{chen2025endtoendvisualautonomousparking,
      title={End-to-End Visual Autonomous Parking via Control-Aided Attention}, 
      author={Chao Chen and Shunyu Yao and Yuanwu He and Tao Feng and Ruojing Song and Yuliang Guo and Xinyu Huang and Chenxu Wu and Ren Liu and Chen Feng},
      year={2025},
      eprint={2509.11090},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.11090}, 
}
```

Our diffusion code is adapted from the paper [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/abs/2205.09991):
```
@misc{janner2022planningdiffusionflexiblebehavior,
      title={Planning with Diffusion for Flexible Behavior Synthesis}, 
      author={Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
      year={2022},
      eprint={2205.09991},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2205.09991}, 
}
```
and [Learning Wheelchair Tennis Navigation from Broadcast Videos with Domain Knowledge Transfer and Diffusion Motion Planning](https://arxiv.org/abs/2409.19771):
```
@misc{wu2025learningwheelchairtennisnavigation,
      title={Learning Wheelchair Tennis Navigation from Broadcast Videos with Domain Knowledge Transfer and Diffusion Motion Planning}, 
      author={Zixuan Wu and Zulfiqar Zaidi and Adithya Patil and Qingyu Xiao and Matthew Gombolay},
      year={2025},
      eprint={2409.19771},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.19771}, 
}
```
