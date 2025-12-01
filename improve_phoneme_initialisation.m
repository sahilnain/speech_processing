function [HMMs] = improve_phoneme_initialisation(featureDir,HMMs,feature_dim,digit2phon_lookup,phon2HMM_lookup)

%Determine better initialisation estimates for the HMM's

%Arrays for modelling the Silence estimates
entries_per_file = 2;

leadarray  = zeros(entries_per_file*length(featureDir),feature_dim);
trailarray = zeros(entries_per_file*length(featureDir),feature_dim);



for fileIter = 1:length(featureDir)
    %Loop over the filenames, find the ones where the utterance is a single
    %digit
    disp(fileIter)

    filename = featureDir(fileIter).name;
    b = readNPY(append(featureDir(fileIter).folder,'/',featureDir(fileIter).name));

    sequence = split(filename,{'a','b'});
    sequence = sequence{1};
    if(strlength(sequence) == 1)
    %Convert this digit to a chain of phonemes
    phon_sequence = digit2phon_lookup(sequence);
    Composite = create_composite_HMM(HMMs,phon_sequence,phon2HMM_lookup,2);

    N = Composite.numStates;
    map = Composite.stateMap(:,2:3);
    step = floor(size(b,1)/N);
        for i = 1:N
            data = b((i-1)*step +1:i*step,:);
            %Using this data, compute a 1x80 estimate for mu and sigma
            mu_init = mean(data, 1);        
            sigma_init = std(data, 0, 1);
    
    
            HMMs(map(i,1)).update(map(i,2)).nom_mu = HMMs(map(i,1)).update(map(i,2)).nom_mu + step.*mu_init;
            HMMs(map(i,1)).update(map(i,2)).denom = HMMs(map(i,1)).update(map(i,2)).denom + step;
            HMMs(map(i,1)).update(map(i,2)).nom_sigma = (step - 1)*sigma_init.^2 + ...
            step.*((HMMs(map(i,1)).update(map(i,2)).nom_mu./HMMs(map(i,1)).update(map(i,2)).denom) - mu_init).^2 + HMMs(map(i,1)).update(map(i,2)).nom_sigma;    
        end
    end

    %Determine better initialisation estimates for silence
    leadarray(2*(fileIter-1) + 1:2*fileIter,:)  = b(1:2,:); %Accumulate leading silence
    trailarray(2*(fileIter-1) + 1:2*fileIter,:) = b(end-1:end,:); %Accumulate trailing silence
end

%After accumulating the estimates, apply these to the HMMs for
%initialisation
for k = 1:19   %'Z' to 'EY'
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
for l = 1:HMMs(20).numStates
    HMMs(20).emission(l).mu = mu_lead;
    HMMs(20).emission(l).sigma = sigma_lead;      
end

mu_trail = mean(trailarray, 1);        
sigma_trail = std(trailarray, 0, 1);
for l = 1:HMMs(21).numStates
    HMMs(21).emission(l).mu = mu_trail;
    HMMs(21).emission(l).sigma = sigma_trail;     
end
end






