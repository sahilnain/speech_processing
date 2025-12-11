
%Test and train HMMs where each HMM represents a digit.
clear;
% Variables
featureFile_train = "FBank_train\train";
featureFile_test = "FBank_test\test";

featureDir_train = dir(fullfile(featureFile_train,'**/*.*'));
sdir = [featureDir_train.isdir];
one = logical(ones(length(featureDir_train),1));
removeMask = (sdir' == one);
featureDir_train(removeMask) = [];

featureDir_test = dir(fullfile(featureFile_test,'**/*.*'));
sdir = [featureDir_test.isdir];
one = logical(ones(length(featureDir_test),1));
removeMask = (sdir' == one);
featureDir_test(removeMask) = [];

%Create data objects containing all of the feature data
data_train = struct('name', {}, 'data', {});
for i = 1:length(featureDir_train)
    filename = featureDir_train(i).name;
    X = readNPY(append(featureDir_train(i).folder,'/',featureDir_train(i).name));

    data_train(i).name = filename;
    data_train(i).data    = X;

end
data_test = struct('name', {}, 'X', {});
for i = 1:length(featureDir_test)
    filename = featureDir_test(i).name;
    X = readNPY(append(featureDir_test(i).folder,'/',featureDir_test(i).name));

    data_test(i).name = filename;
    data_test(i).data   = X;

end





%Initialise the HMM's for the digits, We assume each digit is modelled by an HMM with the
%same amount of states
N           = 5; %Number of states
N_silence   = 3; %Number of states for silence
feature_dim = 80;

for digit = 1:9
    
    HMMs(digit) = create_HMM(N,feature_dim,int2str(digit));

end

HMMs(10) = create_HMM(N,feature_dim,'z');         %Zero
HMMs(11) = create_HMM(N,feature_dim,'o');         %Oh
HMMs(12) = create_HMM(N_silence,feature_dim,'s'); %Leading Silence
HMMs(13) = create_HMM(N_silence,feature_dim,'q'); %Trailing Silence
HMMs(14) = create_HMM(1,feature_dim,'r'); %(Optional, interdigit silence)


HMMs = improve__digit_initialisation(data_train,HMMs,feature_dim);

epochs = 1;

for z = 1:epochs
%Train the models
tic
HMMs = train_HMMs(data_train,HMMs,feature_dim);
toc
end

%Test the models and disp WER store the accuracy in a vec
gamma = -100;
WER = test_HMM(data_test,HMMs,gamma);



