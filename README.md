# Dino-Diffusion Modular Designs Bridge the Cross-Domain Gap in Autonomous Parking

## Statement

We are sorry but we have to temporarily remove the code to follow the Bosch source code opening process. We will immediately open the code again after the period is done. Feel free to email Zixuan Wu (zwu380@gatech.edu) for any questions!

## Author

Zixuan Wu, Hengyuan Zhang, Ting-Hsuan Chen, Yuliang Guo, David Paz, Xinyu Huang, Liu Ren 

(Work done in [Bosch AI Silicon Valley](https://www.bosch-ai.com/))

## Introduction
This repository will contain the code for our submitted conference paper.

This work proposes a learning modular based pipeline of the autonomous parking that can be trained with only a specific environment setting in CARLA but applied to multiple different domain settings and even the Gaussian-Splatting (GS) rendered real-world in a zero-shot fashion without additional data collection. The inputs are the images captured by surrounding cameras, while the output is an on-the-fly updated trajectory. We test the performance of our policies on both CARLA and real-world 3DGS settings.

## Video

We intuitively compare our dino-diffusion pipeline with the concurrent state-of-art method in the following video. It is clear that our method outperform baselines in the cross-domain tasks.

<p align="center">
  <img src="./cross_domain_compressed.gif" alt="Dino_Diffusion Parking Demo" width="85%">
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
