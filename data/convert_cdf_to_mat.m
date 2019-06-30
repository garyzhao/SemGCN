% Extract "Poses_D3_Positions_S*.tgz" to the "pose" directory
% and run this script to convert all .cdf files to .mat

pose_directory = 'pose';
dirs = dir(strcat(pose_directory, '/*/MyPoseFeatures/D3_Positions/*.cdf'));

paths = {dirs.folder};
names = {dirs.name};

for i = 1:numel(names)
    data = cdfread(strcat(paths{i}, '/', names{i}));
    save(strcat(paths{i}, '/', names{i}, '.mat'), 'data');
end