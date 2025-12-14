%This script contains the training and testing procedure of the
%phoneme-based approach
clear;
%Configure the data path
% Variables
featureFile_train = "FBank_train\train";
featureFile_test = "FBank_test\test";
N           = 3; %Number of states for phoneme 
N_silence   = 3; %Number of states for silence
feature_dim = 80;

Phonemes = {'Z' ,'IY','R','OW','W','AH','N','T',...
    'UW','TH','F','AO','AY','V','S','IH','K','EH',...
    'EY'};

digits = {'1','2','3','4','5','6','7','8','9','z','o','s','q','r'};

%Configurable params
oh_indep  = false;

%Get the directories to the training and testing data
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
    disp(i)

end

% uncomment when working with single delta of MFCC or just MFCC
% numCoeffs = 13;
% % set to 2*numCoeffs when working with delta for MFCC or
% % set to numCoeffs when working with just MFCC
% feature_dim = 26;
% 
% dataItems = length(data_train);
% for i = 1:dataItems
% %     % uncomment for working with MFCC features
% %     % Converting Fbank to MFCC
% %     mfcc = dct(data_train(i).data, [], 2);      % DCT along the feature dimension
% %     mfcc = mfcc(:, 1:numCoeffs);
% %     % Converting MFCC to single delta MFCC
% %     Xpad = [mfcc(1,:); mfcc; mfcc(end,:)];
% %     delta = (Xpad(3:end,:) - Xpad(1:end-2,:)) / 2;
% %     Xsd   = [mfcc, delta];       % [T x (2D)]
% %     data_train(i).data = Xsd;
% end
% 
% dataItems = length(data_test);
% for i = 1:dataItems
% %     % uncomment for working with MFCC features
% %     % Converting Fbank to MFCC
% %     mfcc = dct(data_test(i).data, [], 2);      % DCT along the feature dimension
% %     mfcc = mfcc(:, 1:numCoeffs);
% %     % Converting MFCC to single delta MFCC
% %     Xpad = [mfcc(1,:); mfcc; mfcc(end,:)];
% %     delta = (Xpad(3:end,:) - Xpad(1:end-2,:)) / 2;
% %     Xsd   = [mfcc, delta];       % [T x (2D)]
% %     data_test(i).data = Xsd;
% end

%Create the phoneme HMMs. If we want an independant model for oh, we create
%an extra HMM
for i =1:size(Phonemes,2)
    HMMs(i) = create_HMM(N,feature_dim,Phonemes{i});
end
HMMs(20) = create_HMM(N_silence,feature_dim,'s'); %Leading Silence
HMMs(21) = create_HMM(N_silence,feature_dim,'q'); %Trailing Silence
HMMs(22) = create_HMM(1,feature_dim,'r'); %Trailing Silence

if(oh_indep == true)
HMMs(23) = create_HMM(7,feature_dim,'o'); %Independant model for oh
digit2phon = {{'W','AH','N'},{'T','UW'},{'TH','R','IY'},...
    {'F','AO','R'},{'F','AY','V'},{'S','IH','K','S'},{'S','EH','V','AH','N'},...
    {'EY','T'},{'N','AY','N'},{'Z','IY','R','OW'},{'o'},{'s'},{'q'},{'r'}};

else
%Construct a mapping from phonemes to digits
digit2phon = {{'W','AH','N'},{'T','UW'},{'TH','R','IY'},...
    {'F','AO','R'},{'F','AY','V'},{'S','IH','K','S'},{'S','EH','V','AH','N'},...
    {'EY','T'},{'N','AY','N'},{'Z','IY','R','OW'},{'OW'},{'s'},{'q'},{'r'}};
end

%Create a lookup table which maps digits to a phoneme sequence
digit2phon_lookup = containers.Map(digits, digit2phon);

%Create a lookup table which maps phonemes to an HMM entry number
phon2HMM_lookup =   containers.Map({HMMs.tag}, num2cell(1:numel(HMMs)));

%Next, we will go over all  utterances of a single digit in order to
%improve the initialisation of these Phoneme HMM's

%HMMs = improve_phoneme_initialisation(data_train,HMMs,feature_dim,digit2phon_lookup,phon2HMM_lookup);

%Start training the model for a number of iterations
epochs = 10;
gamma = -75;
for z = 1:epochs
    %Train the models
    tic
    [HMMs, total_loglik] = train_phoneme_HMMs(data_train,HMMs,feature_dim,digit2phon_lookup,phon2HMM_lookup);
    toc
end

WER = test_phoneme_HMM(data_test,HMMs,digit2phon_lookup,phon2HMM_lookup,gamma);

% dataItems = length(data_train);
% for i = 1:dataItems
%     % Converting Fbank to MFCC
%     mfcc = dct(data_train(i).data, [], 2);      % DCT along the feature dimension
%     mfcc = mfcc(:, 1:numCoeffs);
%     % One-line central difference (equivalent to N=1), with edge replication
%     Xpad = [mfcc(1,:); mfcc; mfcc(end,:)];
%     delta = (Xpad(3:end,:) - Xpad(1:end-2,:)) / 2;
%     Xsd   = [mfcc, delta];       % [T x (2D)]
%     data_train(i).data = Xsd;
% end
% 
% dataItems = length(data_test);
% for i = 1:dataItems
%     % Converting Fbank to MFCC
%     mfcc = dct(data_test(i).data, [], 2);      % DCT along the feature dimension
%     mfcc = mfcc(:, 1:numCoeffs);
%      % One-line central difference (equivalent to N=1), with edge replication
%     Xpad = [mfcc(1,:); mfcc; mfcc(end,:)];
%     delta = (Xpad(3:end,:) - Xpad(1:end-2,:)) / 2;
%     Xsd   = [mfcc, delta];       % [T x (2D)]
%     data_test(i).data = Xsd;
% end













