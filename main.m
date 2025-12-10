
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



%Initialise the HMM's for the digits, We assume each digit is modelled by an HMM with the
%same amount of states
N           = 7; %Number of states
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


HMMs = improve__digit_initialisation(featureDir_train,HMMs,feature_dim);

epochs = 15;

for z = 1:epochs
%Train the models
HMMs = train_HMMs(featureDir_train,HMMs,feature_dim);
end

%Test the models and disp WER store the accuracy in a vec
gamma = -100;
WER = test_HMM(featureDir_test,HMMs,gamma);



