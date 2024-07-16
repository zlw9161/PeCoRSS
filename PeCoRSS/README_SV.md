# PeCoRSS: Awsome Radar Semantic Segmentation Library developed by PeCoLab of ISTA@CASIC

- Contributers of PeCoLab
  - Liwen Zhang [Liwen Git.](https://github.com/zlw9161)
  - Youcheng Zhang
  - Xinyan Zhang
  - Teng Li
  - Wen Chen
  - Zijun Hu.
- Recommend Environment (Main part)
  - Python 3.9.19
  - Torch 2.1.0
  - Numpy 1.26.4
  - Torch Vision 0.16.0

## Illustrations of RSS-Nets

### TARSS-Net

![illustration-1](./images/tarss-net-teaser.png "TARSS-Net")

![illustration-1-1](./images/tarss-principle.png "TARSS-Net Principle")

> TARSS-Net: Temporal-Aware Radar Semantic Segmentation Network, under submission of NeurIPS 2024.

### PKCIn-Net

![illustration-2](./images/pkcin.png "PKCIn-Net")

![illustration-2-1](./images/pkc-vanilla.png "Vanilla-PKC")

![illustration-2-2](./images/pkc-reda.png "ReDA-PKC")

> PeakConv: Learning Peak Receptive Field for Radar Semantic Segmentation, CVPR2023.
> [PeakConv Code Link](https://github.com/zlw9161/PKC)

## Illustrations of Radar DataSets

### CARRADA-RAC V.S. CARRADA

![illustration-3](./images/carrada-rac-vis.png "CARRADA-RAC Vis.")
[CARRADA_RAC Link](https://github.com/zlw9161/CARRADA-RAC)

[CARRADA Link](https://github.com/valeoai/carrada_dataset)

![illustration-3-1](./images/carrada-rac-pipeline.png "CARRADA-RAC Pipeline")

> CARRADA_RAC Pipline is used for correct the annotations of CARRADA

## Model-Data Support

### Single-View Support

| DataSet    | Model          | Description              | Task Type    |
| ---------- | -------------- | ------------------------ | ------------ |
| KuRALS-CWR | CWRNet         | Default baseline model.  | sv (RD only) |
|            | PKCIn4CWR      | PKCIn ReDA version       |              |
|            | PKCIn4CWR_Dns  | PKCIn Vanilla version    |              |
|            | TARSS4CWR_V1   | TARSS-Net w/ Depth-TRAP  |              |
|            | TARSS4CWR_V2   | TARSS-Net w/ Spatio-TRAP |              |
|            | Unet           |                          |              |
|            | FCN8s          |                          |              |
|            | DeepLabV3Plus  | DeepLabV3+ w/ ResNet101  |              |
|            | AdaPKC         | To be merged             |              |
|            | TFoc           | To be merged             |              |
|            | KuRadNet       | Default baseline model.  | sv (RD only) |
|            | PKCIn4CWR      | PKCIn ReDA version       |              |
|            | PKCIn4CWR_Dns  | PKCIn Vanilla version    |              |
|            | TARSS4CWR_V1   | TARSS-Net w/ Depth-TRAP  |              |
|            | TARSS4CWR_V2   | TARSS-Net w/ Spatio-TRAP |              |
|            | Unet           |                          |              |
|            | FCN8s          |                          |              |
|            | DeepLabV3Plus  | DeepLabV3+ w/ ResNet101  |              |
|            | AdaPKC         | To be merged             |              |
|            | TFoc           | To be merged             |              |
|            |                |                          |              |
| KuRALS-PDR | BaseLine3D     | Baseline model w/ 3DConv | sv (RA/RD)   |
|            | TARSSNet_V1_RD | TARSS-Net w/ Depth-TRAP  | sv (RD only) |
|            | TARSSNet_V1_RA | TARSS-Net w/ Depth-TRAP  | sv (RA only) |
|            | TARSSNet_V2_RD | TARSS-Net w/ Spatio-TRAP | sv (RD only) |
|            | TARSSNet_V2_RA | TARSS-Net w/ Spatio-TRAP | sv (RA only) |
|            | PKCIn          | To be merged             |              |
|            | PKCOn          | To be merged             |              |
|            | AdaPKC         | To be merged             |              |
|            | TFoc           | To be merged             |              |

| DataSet                  | Model          | Description              | Task Type    |
| ------------------------ | -------------- | ------------------------ | ------------ |
| Dataet                   | BaseLine3D     | Baseline model w/ 3DConv | sv (RA/RD)   |
| CARRADA<br />CARRADA-RAC | TARSSNet_V1_RD | TARSS-Net w/ Depth-TRAP  | sv (RD only) |
|                          | TARSSNet_V1_RA | TARSS-Net w/ Depth-TRAP  | sv (RA only) |
|                          | TARSSNet_V2_RD | TARSS-Net w/ Spatio-TRAP | sv (RD only) |
|                          | TARSSNet_V2_RA | TARSS-Net w/ Spatio-TRAP | sv (RA only) |
|                          | PKCIn          | To be merged             |              |
|                          | PKCOn          | To be merged             |              |
|                          | AdaPKC         | To be merged             |              |
|                          | TFoc           | To be merged             |              |

## PeCoRSS Installation

- Please first clone or download the repo from [PeCoLab/PeCoRSS](https://github.com/zlw9161/PeCoLab/tree/main)
> *Please note that the PeCoLab is a integration of different perception libs, PeCoRSS is one of the libs for Radar Semantic Segmentation.*

### Installation Steps

1. Install basic PeCoRSS lib using pip:

```bash
$ cd PeCoLab/PeCoRSS
$ pip install -e .
```

2. Install the rest dependencies using pip or conda (taking ***pip*** as an example):

```bash
$ pip install scikit-image scikit-learn scipy # for metric stats or data process
$ pip install tensorboard # for visualization
```

## Running Tutorial for Single View RSS (svrss)

### Path Setup (Data & Logs)

1. Before setting, pls make sure your project home path 'SVRSS_HOME' with `PeCoRSS/svrss/utils/check_home.py`
2. Using ./utils/set_paths.py to set the data/log paths:
```bash
$ cd PeCoRSS/svrss/utils/
$ python set_paths.py --carrada /user_path/datasets/Carrada --cwr /user_path/datasets/KulRALS/KuCW --pdr /user_path/datasets/KulRALS/KuPD --logs /user_path/logs
```
- The paths will then be written in `svrss/config_files/config.ini`.

### Accelerating PeakConv (Optional)

>**Please Note**: This is optional for running this repo, however if you would like to use the faster version of PKC-based models, this step is required.

Installing the cuda implementation of PRF gathering;

```bash
$ cd PeCoLab/PeCoRSS
$ python setup.py develop --with-cuda
```

#### Accelerating Effects

| Model          | GPU Mem.  | GMac   | FPS   | Input Size     | #Params. |
| -------------- | --------- | ------ | ----- | -------------- | -------- |
| PKCIn4CWR      | 12007 MB  | 162.52 | 29.5  | 1,1,5,124,2048 | 1.21 M   |
| PKCIn4CWR(accelerated) | 5697 MB   | 162.52 | 63.7  | 1,1,5,124,2048 | 1.21 M   |
| PKCIn4PDR      | 3007 MB   | 32.77  | 137.5 | 1,1,5,64,800   | 1.21 M   |
| PKCIn4CWR((accelerated))| 1761 MB   | 32.77  | 396.6 | 1,1,5,64,800   | 1.21 M   |


### Training

Using training script `svrss/run.py` to train a model, this script will load the configuration info from a JSON config file, e.g., `config_files/tarssnet_v1.json`. The command line is as follows:

```bash
$ python run.py --cfg config_files/tarssnet_v1.json
```

>**Please Note**: The model and dataset relation is listed in **Model-Data Support Table**. 

### Testing

Actually, if you finished the training stage, the test results will also be calculated and saved in the log file. However, you can also just test the saved model using the script `svrss/infer.py`.

Similar with the training stage, using the script `infer.py` to evaluate the trained model as the following command line:

```bash
$ python infer.py --cfg /home/logs/carrada/tarssnet_v1/name_of_the_model/config.json
```

### Visulization w/ Tensorboard

When the model is trained and saved in you log path, e.g., `userpath/logs/datatype/exp_name/model.pt`.
You can use the tensorboard tool to check what happened during training and inferring:
```bash
$ tensorboard --logdir=userpath/logs/datatype/exp_name/ --port 8898 (or any other port)
```

## Acknowledgements

- Thank [CARRADA dataset](https://arxiv.org/abs/2005.01456) for providing the multi-view Radar dataset.
- Thank [MVRSS](https://arxiv.org/abs/2103.16214) for providing the basic model framework of multi-view RSS network. And we build our framework using the basic framework of `mvrss` lib.

## License

The PeCoRSS repo is released under the Apache 2.0 license.
