function [HMMs] = improve__digit_initialisation(featureDir,HMMs,feature_dim)

%Create the lookup table
tags = {HMMs.tag};
indices = num2cell(1:numel(HMMs));  % Make cell array of indices
lookup = containers.Map(tags, indices);

%Arrays for modelling the Silence estimates
entries_per_file = 2;
leadarray  = zeros(entries_per_file*length(featureDir),feature_dim);
trailarray = zeros(entries_per_file*length(featureDir),feature_dim);

%Determine better initialisation estimates for the HMM's
for fileIter = 1:length(featureDir)
    disp(fileIter)
    %Loop over the filenames, find the ones where the utterance is a single
    %digit
    filename = featureDir(fileIter).name;
    b = featureDir(fileIter).data;

    sequence = split(filename,{'a','b'});
    sequence = sequence{1};
    if(strlength(sequence) == 1)
        d = lookup(sequence);
        %Divide the data for this utterance into N segments
        %and assign these to the update of the states
        N = HMMs(d).numStates;
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
    leadarray(2*(fileIter-1) + 1:2*fileIter,:)  = b(1:2,:); %Accumulate leading silence
    trailarray(2*(fileIter-1) + 1:2*fileIter,:) = b(end-1:end,:); %Accumulate trailing silence
end

%After accumulating the estimates, apply these to the HMMs for
%initialisation
for k = 1:11 %1 to oh
    for l = 1:HMMs(k).numStates
        
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
    HMMs(12).emission(l).mu = mu_lead;
    HMMs(12).emission(l).sigma = sigma_lead; 
end

mu_trail = mean(trailarray, 1);        
sigma_trail = std(trailarray, 0, 1);
for l = 1:HMMs(13).numStates
    HMMs(13).emission(l).mu = mu_trail;
    HMMs(13).emission(l).sigma = sigma_trail;     
end
%Set the mu and sigma for inter-symbol silence to the mu and sigma of the
%leading silence
for l = 1:HMMs(14).numStates
    HMMs(14).emission(l).mu = mu_trail;
    HMMs(14).emission(l).sigma = sigma_trail;
     
end





end