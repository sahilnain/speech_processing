function [CompositeHMM] = create_composite_HMM(HMMs,sequence,lookup)

sequence = join(['s' , sequence ,'q'],"");
totalStates = 0;
for v = sequence
    totalStates = totalStates + HMMs(lookup(v)).numStates;
end

A_total = zeros(totalStates);
pi_total = zeros(1, totalStates);
% Keep track of which global state belongs to which digit/state
stateMap = []; % rows: [globalIndex, digit, localState]
CompositeHMM.emission = [];
offset = 0;
for idx = 1:length(sequence)
    d = lookup(sequence(idx));
    ns = HMMs(d).numStates;
    A =  HMMs(d).A;
    
    % Copy A block into total transition matrix
    A_total(offset+1:offset+ns, offset+1:offset+ns) = A;
    
    % Connect last state of this digit to first state of next digit
    if idx < length(sequence)
        A_total(offset+ns, offset+ns+1) = 1.0;
    end
    
    % Initial probability: first digit starts at its first state
    if idx == 1
        pi_total(offset+1) = 1.0;
    end
    
    % Record mapping
    for s = 1:ns
        stateMap = [stateMap; offset+s, d, s];
    end
    
    offset = offset + ns;


    CompositeHMM.emission = [ CompositeHMM.emission ,HMMs(d).emission];


end

CompositeHMM.A  = A_total;
CompositeHMM.pi = pi_total;
CompositeHMM.stateMap = stateMap;


end