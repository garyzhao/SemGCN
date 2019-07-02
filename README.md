# Semantic Graph Convolutional Networks for 3D Human Pose Regression (CVPR 2019)

This repository holds the Pytorch implementation of [Semantic Graph Convolutional Networks for 3D Human Pose Regression](https://arxiv.org/abs/1904.03345) by Long Zhao, Xi Peng, Yu Tian, Mubbasir Kapadia and Dimitris N. Metaxas. If you find our code useful in your research, please consider citing:

```
@inproceedings{zhaoCVPR19semantic,
  author    = {Zhao, Long and Peng, Xi and Tian, Yu and Kapadia, Mubbasir and Metaxas, Dimitris N.},
  title     = {Semantic Graph Convolutional Networks for 3D Human Pose Regression},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3425--3435},
  year      = {2019}
}
```

## Introduction

We propose Semantic Graph Convolutional Networks (SemGCN), a novel graph convolutional network architecture that operates on regression tasks with graph-structured data. The code for training and evaluating our approach for 3D human pose estimation on the [Human3.6M Dataset](http://vision.imar.ro/human3.6m/) is provided in this repository.

In this repository, 3D human poses are predicted according to **Configuration #1** in [our paper](https://arxiv.org/pdf/1904.03345.pdf): we only leverage 2D joints of the human pose as inputs. We utilize the method described in Pavllo et al. [2] to normalize 2D and 3D poses in the dataset, which is different from the original implementation in our paper. To be specific, 2D poses are scaled according to the image resolution and normalized to [-1, 1]; 3D poses are aligned with respect to the root joint . Please refer to the corresponding part in Pavllo et al. [2] for more details. We predict 17 joints as Martinez et al. [1]. We also provide the results of Martinez et al. [1] in the same setting for comparison.

### Results on Human3.6M

Under Protocol 1 (mean per-joint position error) and Protocol 2 (mean per-joint position error after rigid alignment).

| Method | 2D Detections | # of Epochs | # of Parameters | MPJPE (P1) | P-MPJPE (P2) |
|:-------|:-------|:-------:|:-------:|:-------:|:-------:|
| Martinez et al. [1] | Ground truth | 200  | 4.29M | 44.17 mm | 34.35 mm |
| SemGCN | Ground truth | 50 | 0.27M | 41.16 mm | 31.57 mm |
| SemGCN (w/ Non-local) | Ground truth | 30 | 0.43M | **39.87 mm** | **30.16 mm** |

More results using different 2D detections will be added in the incoming updates.

### References

[1] Martinez et al. [A simple yet effective baseline for 3d human pose estimation](https://arxiv.org/pdf/1705.03098.pdf). ICCV 2017.

[2] Pavllo et al. [3D human pose estimation in video with temporal convolutions and semi-supervised training](https://arxiv.org/pdf/1811.11742.pdf). CVPR 2019.

## Quick start

This repository is build upon Python v2.7 and Pytorch v1.1.0 on Ubuntu 16.04. NVIDIA GPUs are needed to train and test. See [`requirements.txt`](requirements.txt) for other dependencies. We recommend installing Python v2.7 from [Anaconda](https://www.anaconda.com/), and installing Pytorch (>= 1.1.0) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version. Then you can install dependencies with the following commands.

```
git clone git@github.com:garyzhao/SemGCN.git
cd SemGCN
pip install -r requirements.txt
```

### Dataset setup
You can find the instructions for setting up the Human3.6M in [`data/README.md`](data/README.md). For this short guide, we focus on Human3.6M. The code for data preparation is borrowed from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).

### Evaluating our pretrained models
The pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1c7Iz6Tt7qbaw0c1snKgcGOD-JGSzuZ4X?usp=sharing). Put `checkpoint` in the project root directory.

To evaluate Martinez et al. [1], run:
```
python main_linear.py --evaluate checkpoint/pretrained/ckpt_linear.pth.tar
```

To evaluate SemGCN without non-local blocks, run:
```
python main_gcn.py --evaluate checkpoint/pretrained/ckpt_semgcn.pth.tar
```

To evaluate SemGCN with non-local blocks, run:
```
python main_gcn.py --non_local --evaluate checkpoint/pretrained/ckpt_semgcn_nonlocal.pth.tar
```

Note that the error is calculated in an **action-wise** manner.

### Training from scratch
If you want to reproduce the results of our pretrained models, run the following commands.

For Martinez et al. [1]:
```
python main_linear.py
```

For SemGCN without non-local blocks:
```
python main_gcn.py --epochs 50
```
By default the application runs in training mode. This will train a new model for 50 epochs without non-local blocks, using ground truth 2D detections. You may change the value of `num_layers` (4 by default) and `hid_dim` (128 by default) if you want to try different network settings. Please refer to [`main_gcn.py`](main_gcn.py) for more details.

For SemGCN with non-local blocks:
```
python main_gcn.py --non_local --epochs 30
```
This will train a new model with non-local blocks for 30 epochs, using ground truth 2D detections.

## Acknowledgement

Part of our code is borrowed from the following repositories.

- [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)
- [3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)

We thank to the authors for releasing their codes. Please also consider citing their works.
