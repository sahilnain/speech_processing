function [HMMs] = train_HMMs(featureDir,HMMs,feature_dim)

%Create the lookup table
tags = {HMMs.tag};
indices = num2cell(1:numel(HMMs));  % Make cell array of indices
lookup = containers.Map(tags, indices);

for fileIter = 1:length(featureDir)
    filename = featureDir(fileIter).name;
    b = featureDir(fileIter).data;
    disp(fileIter)
    %Construct a state sequence by concatenating the different HMM's to
    %each other
    sequence = split(filename,{'a','b'});
    sequence = sequence{1};
    
    Composite_HMM = create_composite_HMM(HMMs,sequence,lookup,0);
  
    %Generate the alpha's and beta's and gamma's
    gamma_u = comp_forward_backward(Composite_HMM,b);

    
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
       
        HMMs(k).emission(l).mu = HMMs(k).update(l).nom_mu ./ HMMs(k).update(l).denom;
        HMMs(k).emission(l).sigma = HMMs(k).update(l).nom_sigma ./ HMMs(k).update(l).denom;
        HMMs(k).update(l).nom_mu = zeros(1,feature_dim);
        HMMs(k).update(l).nom_sigma = zeros(1,feature_dim);
        HMMs(k).update(l).denom = 0;

        
    end
end
end