# Dataset setup

## Human3.6M
We provide two ways to set up the Human3.6M dataset on our pipeline. You can either use the [dataset preprocessed by Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) (fastest way) or convert the original dataset from scratch. The two methods produce the same result. After this step, you should end up with two files in the `data` directory: `data_3d_h36m.npz` for the 3D poses, and `data_2d_h36m_gt.npz` for the ground-truth 2D poses.

### Setup from preprocessed dataset
Download the [h36m.zip archive](https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip) (source: [3D pose baseline repository](https://github.com/una-dinosauria/3d-pose-baseline)) to the `data` directory, and run the conversion script from the same directory. This step does not require any additional dependency.

```sh
cd data
wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip
python prepare_data_h36m.py --from-archive h36m.zip
cd ..
```

If the Dropbox link does not work, please download the dataset from [Google Drive](https://drive.google.com/drive/folders/1c7Iz6Tt7qbaw0c1snKgcGOD-JGSzuZ4X?usp=sharing).

### Setup from original source
Alternatively, you can download the dataset from the [Human3.6m website](http://vision.imar.ro/human3.6m/) and convert it from its original format. This is useful if the other link goes down, or if you want to be sure to use the original source. MATLAB is required for this step.

First, we need to convert the 3D poses from `.cdf` to `.mat`, so they can be loaded from Python scripts. To this end, we have provided the MATLAB script `convert_cdf_to_mat.m` in the `data` directory. Extract the archives named `Poses_D3_Positions_S*.tgz` (subjects 1, 5, 6, 7, 8, 9, 11) to a directory named `pose`, and set up your directory tree so that it looks like this:

```
/path/to/dataset/convert_cdf_to_mat.m
/path/to/dataset/pose/S1/MyPoseFeatures/D3_Positions/Directions 1.cdf
/path/to/dataset/pose/S1/MyPoseFeatures/D3_Positions/Directions.cdf
...
```
Then run `convert_cdf_to_mat.m` from MATLAB.

Finally, as before, run the Python conversion script specifying the dataset path:
```sh
cd data
python prepare_data_h36m.py --from-source /path/to/dataset/pose
cd ..
```

## 2D detections for Human3.6M
We provide support for the following 2D detections:

- `gt`: ground-truth 2D poses, extracted through the camera projection parameters.
- `sh_pt_mpii`: Stacked Hourglass detections, pretrained on MPII.
- `sh_ft_h36m`: Stacked Hourglass detections, fine-tuned on Human3.6M.

The 2D detection source is specified through the `--keypoints` parameter, which loads the file `data_2d_${DATASET}_${DETECTION}.npz` from the `data` directory, where `DATASET` is the dataset name (e.g., `h36m`) and `DETECTION` is the 2D detection source (e.g., `sh_pt_mpii`). Since all the files are encoded according to the same format, it is trivial to create a custom set of 2D detections.

Ground-truth poses (`gt`) have already been extracted by the previous step. The other detections must be downloaded manually (see instructions below). You only need to download the detections you want to use.

### Stacked Hourglass detections
These detections (both pretrained and fine-tuned) are provided by [Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) in their repository on 3D human pose estimation. The 2D poses produced by the pretrained model are in the same archive as the dataset ([h36m.zip](https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip)). The fine-tuned poses can be downloaded [here](https://drive.google.com/open?id=0BxWzojlLp259S2FuUXJ6aUNxZkE). Put the two archives in the `data` directory and run:

```sh
cd data
python prepare_data_2d_h36m_sh.py -pt h36m.zip
python prepare_data_2d_h36m_sh.py -ft stacked_hourglass_fined_tuned_240.tar.gz
cd ..
```
