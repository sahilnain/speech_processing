% Add npy function to search path
%addpath("D:\masters_course\Speech_processing\npy-matlab\npy-matlab\")
%savepath

% Variables
featureFile = "FBank_train\train";

featureDir = dir(fullfile(featureFile,'**/*.*'));

sdir = [featureDir.isdir];
one = logical(ones(length(featureDir),1));
removeMask = (sdir' == one);
featureDir(removeMask) = [];

%Initialise the HMM's, We assume each digit is modelled by an HMM with the
%same amount of states
N = 5; %Number of states
N_silence = 2; %Number of states for silence
feature_dim = 80;

for digit = 1:9
    
    HMMs(digit) = create_digit_HMM(N,feature_dim,int2str(digit));

end

HMMs(10) = create_digit_HMM(N,feature_dim,'z'); %Zero
HMMs(11) = create_digit_HMM(N,feature_dim,'o'); %Oh
HMMs(12) = create_digit_HMM(N_silence,feature_dim,'s'); %Leading Silence
HMMs(13) = create_digit_HMM(N_silence,feature_dim,'q'); %Trailing Silence
HMMs(14) = create_digit_HMM(2,feature_dim,'r'); %(Optional, interdigit silence)

%Create the lookup table
tags = {HMMs.tag};
indices = num2cell(1:numel(HMMs));  % Make cell array of indices
lookup = containers.Map(tags, indices);

jema = 1;
leadarray = [];
trailarray = [];

%Determine better initialisation estimates for the HMM's
for fileIter = 1:length(featureDir)
    %Loop over the filenames, find the ones where the utterance is a single
    %digit
    filename = featureDir(fileIter).name;
    b = readNPY(append(featureDir(fileIter).folder,'/',featureDir(fileIter).name));

    sequence = split(filename,{'a','b'});
    sequence = sequence{1};
    if(strlength(sequence) == 1)
        d = lookup(sequence);
        %Divide the data for this utterance into N segments
        %and assign these to the update of the states
        step = floor(size(b,1)/N);
        for i = 1:N
            data = b((i-1)*step +1:i*step,:);
            %Using this data, compute a 1x80 estimate for mu and sigma
            mu_init = mean(data, 1);        
            sigma_init = std(data, 0, 1);
            HMMs(d).update(i).nom_mu = HMMs(d).update(i).nom_mu + step.*mu_init;
            HMMs(d).update(i).denom = HMMs(d).update(i).denom + step;
            HMMs(d).update(i).nom_sigma = (step - 1)*sigma_init.^2 + ...
            step.*((HMMs(d).update(i).nom_mu./HMMs(d).update(i).denom) - mu_init).^2 + HMMs(d).update(i).nom_sigma;    
        end
    end
    
    %Determine better initialisation estimates for silence
    leadarray = [leadarray ; b(1:2,:)]; %Accumulate leading silence
    trailarray = [trailarray ; b(end-1:end,:)]; %Accumulate trailing silence

    disp(jema)
    jema = jema +1;
end

%After accumulating the estimates, apply these to the HMMs for
%initialisation
for k = 1:11 %1 to oh
    for l = 1:HMMs(k).numStates
        %Maybe check here if some states have collapsed


        HMMs(k).emission(l).mu = HMMs(k).update(l).nom_mu ./ HMMs(k).update(l).denom;
        
        HMMs(k).emission(l).sigma = sqrt(HMMs(k).update(l).nom_sigma ./ (HMMs(k).update(l).denom - 1));
        HMMs(k).update(l).nom_mu = zeros(1,feature_dim);
        HMMs(k).update(l).nom_sigma = zeros(1,feature_dim);
        HMMs(k).update(l).denom = 0;       
    end
end

mu_lead = mean(leadarray, 1);        
sigma_lead = std(leadarray, 0, 1);
for l = 1:HMMs(12).numStates
    %Maybe check here if some states have collapsed


    HMMs(12).emission(l).mu = mu_lead;
    HMMs(12).emission(l).sigma = sigma_lead;
    HMMs(12).update(l).nom_mu = zeros(1,feature_dim);
    HMMs(12).update(l).nom_sigma = zeros(1,feature_dim);
    HMMs(12).update(l).denom = 0;       
end

mu_trail = mean(trailarray, 1);        
sigma_trail = std(trailarray, 0, 1);
for l = 1:HMMs(13).numStates
    %Maybe check here if some states have collapsed
    HMMs(13).emission(l).mu = mu_trail;
    HMMs(13).emission(l).sigma = sigma_trail;
    HMMs(13).update(l).nom_mu = zeros(1,feature_dim);
    HMMs(13).update(l).nom_sigma = zeros(1,feature_dim);
    HMMs(13).update(l).denom = 0;       
end

for z = 1:5
% Read the features
for fileIter = 1:length(featureDir)
    filename = featureDir(fileIter).name;
    b = readNPY(append(featureDir(fileIter).folder,'/',featureDir(fileIter).name));
    
    %Construct a state sequence by concatenating the different HMM's to
    %each other
    sequence = split(filename,{'a','b'});
    sequence = sequence{1};
    
    Composite_HMM = create_composite_HMM(HMMs,sequence,lookup);
    disp(fileIter)
    %Generate the alpha's and beta's and gamma's
    [alpha_u, beta_u, gamma_u] = comp_forward_backward(Composite_HMM,b);

    %Generate an update for mu and sigma
    denom = sum(exp(gamma_u),1);
    mu = (b'*exp(gamma_u));

    

    
    %sigma
    for i = 1:size(Composite_HMM.stateMap,1)
        %Underlying HMM state
        digit = Composite_HMM.stateMap(i,2:3);

        diff = (b - Composite_HMM.emission(i).mu)';

        %We enforce the diagonal covariance here
        nom = (diff.^2) * exp(gamma_u(:,i));
        %Store the update for the covariance in the correct digt state
        HMMs(digit(1)).update(digit(2)).nom_sigma = HMMs(digit(1)).update(digit(2)).nom_sigma + nom';
        %Store the mu and the denom in the correct state
        HMMs(digit(1)).update(digit(2)).denom = HMMs(digit(1)).update(digit(2)).denom + denom(i);
        HMMs(digit(1)).update(digit(2)).nom_mu = HMMs(digit(1)).update(digit(2)).nom_mu + mu(:,i)';
    end

end

for k = 1:numel(HMMs)
    for l = 1:HMMs(k).numStates
        %Maybe check here if some states have collapsed


        HMMs(k).emission(l).mu = HMMs(k).update(l).nom_mu ./ HMMs(k).update(l).denom;
        HMMs(k).emission(l).sigma = HMMs(k).update(l).nom_sigma ./ HMMs(k).update(l).denom;
        HMMs(k).update(l).nom_mu = zeros(1,feature_dim);
        HMMs(k).update(l).nom_sigma = zeros(1,feature_dim);
        HMMs(k).update(l).denom = 0;

        %if(sum(HMMs(k).emission(l).mu) == 0)
         %   HMMs(k).emission(l).mu = HMMs(k).emission(1).mu;
         %   HMMs(k).emission(l).sigma = HMMs(k).emission(1).sigma;
       % end
        
    end
end

end
