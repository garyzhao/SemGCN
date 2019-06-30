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