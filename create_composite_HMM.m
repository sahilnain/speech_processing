function [CompositeHMM] = create_composite_HMM(HMMs,sequence,lookup,mode)

%Mode 0: digit HMM
%Mode 1: Phoneme HMM with silence
%Mode 2: Phomeme HMM without silence
if mode == 0
    sequence = join(['s' , sequence ,'q'],"");
elseif mode == 1
    sequence = [{'s'}, sequence, {'q'}];
end

totalStates = 0;
stateMap = [];
CompositeHMM.emission = [];


for v = sequence
    if mode == 0
        d = lookup(v); %Which digit HMM do we need for this char in the utterance
    else
        d = lookup(v{1});
    end
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
CompositeHMM.numStates = totalStates;

end