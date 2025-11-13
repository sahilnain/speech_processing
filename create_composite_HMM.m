function [CompositeHMM] = create_composite_HMM(HMMs,sequence,lookup)

sequence = join(['s' , sequence ,'q'],"");
totalStates = 0;
stateMap = [];
CompositeHMM.emission = [];


for v = sequence
    d = lookup(v); %Which digit HMM do we need for this char in the utterance
    ns = HMMs(d).numStates;

    
    CompositeHMM.emission = [ CompositeHMM.emission ,HMMs(d).emission];

    for s = 1:ns
        % Keep track of which global state belongs to which digit/state
        % rows: [globalIndex, digit, localState]
        stateMap = [stateMap; totalStates+s, d, s];
    end  


    totalStates = totalStates + ns;
end

%if we assume we know the transition probabilities a priori, we can easily
%construct the large A matrix
A_total = spdiags([0.9*ones(totalStates,1) 0.1*ones(totalStates,1)], [0 1], totalStates, totalStates);
A_total(totalStates,totalStates) = 1;
%Construct the large pi vector
pi_total = zeros(1, totalStates);
pi_total(1) = 1;


      




CompositeHMM.A  = A_total;
CompositeHMM.pi = pi_total;
CompositeHMM.stateMap = stateMap;


end