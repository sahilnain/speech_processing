
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
N           = 7; %Number of states
N_silence   = 3; %Number of states for silence
feature_dim = 80; %Set to K if working with mystery feature

% % Perform PCA over the mystery feature, only use when working with the
% mystery feature
% X_all = [];
% dataItems = length(data_train);
% for u = 1:dataItems
%     X_all = [X_all; data_train(u).data];
% end
% mu_pca = mean(X_all, 1);           % [1 × 256]
% X_centered = X_all - mu_pca;
% [coeff, score, latent, ~, explained] = pca(X_centered);
%
% % select first few components
% K = 50;
% coeff_red = coeff(:,1:K);   % [256 × K]
% feature_dim = K; %Set to K if working with mystery feature
% dataItems = length(data_train);
% for i = 1:dataItems
%     X_u_pca = (data_train(i).data - mu_pca) * coeff_red;   % [T × K]
%     data_train(i).data = X_u_pca;
% end
% 
% dataItems = length(data_test);
% for i = 1:dataItems
%     X_u_pca = (data_test(i).data - mu_pca) * coeff_red;   % [T × K]
%     data_test(i).data = X_u_pca;
% end

for digit = 1:9
    HMMs(digit) = create_HMM(N,feature_dim,int2str(digit));
end

HMMs(10) = create_HMM(N,feature_dim,'z');         %Zero
HMMs(11) = create_HMM(N,feature_dim,'o');         %Oh
HMMs(12) = create_HMM(N_silence,feature_dim,'s'); %Leading Silence
HMMs(13) = create_HMM(N_silence,feature_dim,'q'); %Trailing Silence
HMMs(14) = create_HMM(1,feature_dim,'r'); %(Optional, interdigit silence)

HMMs = improve__digit_initialisation(data_train,HMMs,feature_dim);

epochs = 10;
%Test the models and disp WER store the accuracy in a vec
gamma = -100;
for z = 1:epochs
    %Train the models
    tic
    [HMMs, total_loglik] = train_HMMs(data_train,HMMs,feature_dim);
    toc
end

WER = test_HMM(data_test,HMMs,gamma);

% numCoeffs = 13; % only useful when working with MFCC feature
%
% % set to 2*K when using delta for mystery feature or
% % set to 2*numCoeffs when working with delta for MFCC or
% % set to numCoeffs when working with just MFCC
% feature_dim = 2*K; % set to 2*K when using delta for mystery feature or

% dataItems = length(data_train);
% for i = 1:dataItems
%     % uncomment for working with MFCC features
%     % % Converting Fbank to MFCC
%     % mfcc = dct(data_train(i).data, [], 2);      % DCT along the feature dimension
%     % mfcc = mfcc(:, 1:numCoeffs);
%     % % Converting MFCC to single delta MFCC
%     % Xpad = [mfcc(1,:); mfcc; mfcc(end,:)];
%     % delta = (Xpad(3:end,:) - Xpad(1:end-2,:)) / 2;
%     % Xsd   = [mfcc, delta];       % [T x (2D)]
%     % % set to Xsd when working with single delta for MFCC else
%     % % set to mfcc when working with MFCC features
%     % data_train(i).data = Xsd;
% 
%     % % uncomment for working with Mystery features
%     % % only delta of mystery feature present here, if working with mystery
%     % % feature alone then the code is at the top.
%     % X_u_pca = (data_train(i).data - mu_pca) * coeff_red;   % [T × K]
%     % Xpad = [X_u_pca(1,:); X_u_pca; X_u_pca(end,:)];
%     % delta = (Xpad(3:end,:) - Xpad(1:end-2,:)) / 2;
%     % Xsd   = [X_u_pca, delta];       % [T x (2D)]
%     % data_train(i).data = Xsd;
% end
% 
% dataItems = length(data_test);
% for i = 1:dataItems
%     % uncomment for working with MFCC features
%     % % Converting Fbank to MFCC
%     % mfcc = dct(data_test(i).data, [], 2);      % DCT along the feature dimension
%     % mfcc = mfcc(:, 1:numCoeffs);
%     % % Converting MFCC to single delta MFCC
%     % Xpad = [mfcc(1,:); mfcc; mfcc(end,:)];
%     % delta = (Xpad(3:end,:) - Xpad(1:end-2,:)) / 2;
%     % Xsd   = [mfcc, delta];       % [T x (2D)]
%     % % set to Xsd when working with single delta for MFCC else
%     % % set to mfcc when working with MFCC features
%     % data_test(i).data = Xsd;
% 
%     % % uncomment for working with Mystery features
%     % % only delta of mystery feature present here, if working with mystery
%     % % feature alone then the code is at the top.
%     % X_u_pca = (data_test(i).data - mu_pca) * coeff_red;   % [T × K]
%     % Xpad = [X_u_pca(1,:); X_u_pca; X_u_pca(end,:)];
%     % delta = (Xpad(3:end,:) - Xpad(1:end-2,:)) / 2;
%     % Xsd   = [X_u_pca, delta];       % [T x (2D)]
%     % data_test(i).data = Xsd;
% end
% 
% for digit = 1:9
% 
%     HMMs(digit) = create_HMM(N,feature_dim,int2str(digit));
% 
% end
% 
% HMMs(10) = create_HMM(N,feature_dim,'z');         %Zero
% HMMs(11) = create_HMM(N,feature_dim,'o');         %Oh
% HMMs(12) = create_HMM(N_silence,feature_dim,'s'); %Leading Silence
% HMMs(13) = create_HMM(N_silence,feature_dim,'q'); %Trailing Silence
% HMMs(14) = create_HMM(1,feature_dim,'r'); %(Optional, interdigit silence)
% 
% 
% HMMs = improve__digit_initialisation(data_train,HMMs,feature_dim);
% 
% epochs = 10;
% 
% %Test the models and disp WER store the accuracy in a vec
% gamma   = -100;
% 
% for z = 1:epochs
%     %Train the models
%     tic
%     [HMMs, total_loglik] = train_HMMs(data_train, HMMs, feature_dim);
%     toc
% end
% WER = test_HMM(data_test, HMMs, gamma);

% Code used for plotting
% hold on;
% plot(WER_nozero(:))
% plot(WER_1_nozero(:))
% title("WER with Delta feature(Mystery features)")
% legend("Mystrey features","Delta feature")
% xlabel("Training iterations")
% ylabel("WER(percentage)")
% hold off;
% 
% hold on;
% plot(log_lik(2:end,1))
% title("Data likelihood")
% % legend("FBank","Delta feature")
% xlabel("Iterations")
% ylabel("Log likelihood")
% hold off;

