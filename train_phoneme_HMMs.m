function [HMMs] = train_phoneme_HMMs(featureDir,HMMs,feature_dim,digit2phon_lookup,phon2HMM_lookup)


for fileIter = 1:length(featureDir)
    %Loop over the filenames, find the ones where the utterance is a single
    %digit
    disp(fileIter)

    filename = featureDir(fileIter).name;
    b = readNPY(append(featureDir(fileIter).folder,'/',featureDir(fileIter).name));

    sequence = split(filename,{'a','b'});
    sequence = sequence{1};
    %Convert this digit sequence into a phoneme large phoneme sequence
    phon_sequence = {};
    for v = sequence
        phon_sequence = [phon_sequence, digit2phon_lookup(v)];
    end
    %Create the composite model for this sequence of phonemes
    Composite_HMM = create_composite_HMM(HMMs,phon_sequence,phon2HMM_lookup,1);

    %Generate the gamma's
    gamma_u = comp_forward_backward(Composite_HMM,b);

    %Generate an update for mu and sigma
    denom = sum(exp(gamma_u),1);
    mu = (b'*exp(gamma_u));

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

%After accumulating estimates for all the HMMs, apply these estimates and
%reset the accumulators
for k = 1:numel(HMMs)
    for l = 1:HMMs(k).numStates
       
        HMMs(k).emission(l).mu = HMMs(k).update(l).nom_mu ./ HMMs(k).update(l).denom;
        HMMs(k).emission(l).sigma = HMMs(k).update(l).nom_sigma ./ HMMs(k).update(l).denom;
        HMMs(k).update(l).nom_mu = zeros(1,feature_dim);
        HMMs(k).update(l).nom_sigma = zeros(1,feature_dim);
        HMMs(k).update(l).denom = 0;

        
    end
end
end