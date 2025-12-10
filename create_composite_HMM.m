function [CompositeHMM] = create_composite_HMM(HMMs,sequence,lookup,mode)

%Mode 0: digit HMM
%Mode 1: Phoneme HMM with silence
%Mode 2: Phomeme HMM without silence
if mode == 0
    %Add the possibility we might encounter an intersymbol silence, account
    %for this in the transition matrix A
    sequence2 = [];
    for v = sequence(1:end-1)
        sequence2 = join([sequence2 ,v, 'r'],"");
    end
    sequence = join(['s',sequence2,sequence(end),'q'],"");


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

%construct the large A matrix: when we are in the first N-1 states of a
%digit HMM, we can only transition to the next state of that digit HMM. In
%the last state of a digit HMM we can transition either to inter-silence or
%to the next digit in our training sequence
if(mode ~=2)
A_total = eye(totalStates,totalStates)*0.9;
offset = 1;

d_silence_inter = size(HMMs,2);



numStates_inter_silence = HMMs(d_silence_inter).numStates;

for v = 1:length(sequence)-1
    if mode == 0
        d = lookup(sequence(v)); %Which digit HMM do we need for this char in the utterance
        d_next = lookup(sequence(v+1));
    else
        d = lookup(sequence{v});
        d_next = lookup(sequence{v+1});
    end
    ns = HMMs(d).numStates;

    %When we're not in the final state of one of the HMMs, we can only go
    %to the next state within that HMM or remain in the same state
    indices = sub2ind(size(A_total), offset:offset + ns -2, offset+1:offset + ns - 1);
    A_total(indices) = 0.1;

    offset = offset + ns ;
    %In the final state of the HMM, go to the next state if we are in
    %leading silence. Go to interdigit or the next digit if we are in a
    %digit which is not the last. Go to the next digit if we are in
    %interdigit silence. Go to trailing silence if we are in the last
    %digit.
    if(d_next == d_silence_inter)
        A_total(offset-1,offset) = 0.1/2;
        A_total(offset-1,offset + numStates_inter_silence)    = 0.1/2;
    else
        A_total(offset-1,offset) = 0.1;
    end  
end
ns = HMMs(d_next).numStates;
indices = sub2ind(size(A_total), offset:offset + ns -2, offset+1:offset + ns - 1);
A_total(indices) = 0.1;
A_total(end,end) = 1;


%Construct the pi vector, we know we will start in leading silence 
pi_total = zeros(1, totalStates);
pi_total(1) = 1;

CompositeHMM.A  = A_total;
CompositeHMM.pi = pi_total;
end

CompositeHMM.stateMap = stateMap;
CompositeHMM.numStates = totalStates;

end