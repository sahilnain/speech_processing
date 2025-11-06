% Add npy function to search path
addpath("D:\masters_course\Speech_processing\npy-matlab\npy-matlab\")
savepath

% Variables
featureFile = "FBank_train\train";

featureDir = dir(fullfile(featureFile,'**/*.*'));

sdir = [featureDir.isdir];
one = logical(ones(length(featureDir),1));
removeMask = (sdir' == one);
featureDir(removeMask) = [];

% Read the features
for fileIter = 1:length(featureDir)
    b = readNPY(append(featureDir(fileIter).folder,'/',featureDir(fileIter).name));
end