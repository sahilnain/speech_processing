function [WER] = test_phoneme_HMM(featureDir,HMMs,digit2phon_lookup,phon2HMM_lookup,gamma)
%From the original mappings, reconstruct the digit HMMs using the phonemes

keysList = {'1','2','3','4','5','6','7','8','9','z','o','s','q','r'};
indices = num2cell(1:14);  % Make cell array of indices
lookup_inv = containers.Map(indices,keysList);

%Reconstruct each digit HMM using phonemes
for i = 1:numel(keysList)
    k = keysList{i};
    HMMs_digit(i) = create_composite_HMM(HMMs,digit2phon_lookup(k),phon2HMM_lookup,2);
end


%Combine all of the HMMs into a super HMM with a large intial state
%probabilty and large transition probability matrix
stateMap = [];
SuperHMM.emission = [];
init_pi = [];
totalStates = 0;
Statevec    = zeros(size(HMMs_digit,2) -1 ,1);

for d = 1:size(HMMs_digit,2)

    SuperHMM.emission = [ SuperHMM.emission ,HMMs_digit(d).emission];
    ns = HMMs_digit(d).numStates;
    Statevec(d) = ns;

    for s = 1:ns
        % Keep track of which global state belongs to which digit/state
        % rows: [globalIndex, digit, localState]
        stateMap = [stateMap; totalStates+s, d, s];
    end 
    
    if(d <= 9) %Digits 1-9
        init_pi = [init_pi , 1, zeros(1,ns-1)];
    elseif(d <= 11)      %Zeros
         init_pi = [init_pi , 0.5, zeros(1,ns-1)];  
    elseif(d == 12)      %Leading silence
         init_pi = [init_pi , 1, zeros(1,ns-1)]; 
         Leading_silence_index = totalStates +1;
    elseif(d == 13)                %Trailing silence
        init_pi = [init_pi , zeros(1,ns)];
        Trailing_silence_index = totalStates +1; 
    else                %Interdigit silence
        init_pi = [init_pi , zeros(1,ns)];
        Interdigit_silence_index = totalStates + 1;
    end
    totalStates = totalStates + ns;
end
pi_sum = sum(init_pi);
SuperHMM.pi = init_pi./pi_sum;

%Loop over all of the states of the HMMs again to construct the transition
%probability matrix A, digits can't connect to leading silence, trailing
%silence can't connect to digits. Interdigit silence can't connect to
%leading and trailing silence
trans_prob = SuperHMM.pi;
%From leading silence, we can go to all digits and to trailing silence
trans_prob(Trailing_silence_index) = trans_prob(Leading_silence_index);
trans_prob(Leading_silence_index) = 0;
%From digits, we can go to all digits, trailing silence and intersymbol
%silence
trans_prob_digit = init_pi;
trans_prob_digit(Interdigit_silence_index) = 1;
trans_prob_digit(Trailing_silence_index) = 1;
trans_prob_digit(Leading_silence_index) = 0;
trans_prob_digit = trans_prob_digit./sum(trans_prob_digit);
%From inter symbol silence, we can only go to digits
trans_prob_silence = init_pi;
trans_prob_silence(Interdigit_silence_index) = 0;
trans_prob_silence(Trailing_silence_index) = 0;
trans_prob_silence(Leading_silence_index) = 0;
trans_prob_silence(Leading_silence_index) = 0;
trans_prob_silence = trans_prob_silence./sum(trans_prob_silence);

A_total = eye(totalStates,totalStates)*0.9;
offset = 1;
for d = 1:size(HMMs_digit,2)
    ns = HMMs_digit(d).numStates;

    %When we're not in the final state of one of the HMMs, we can only go
    %to the next state within that HMM or remain in the same state
    indices = sub2ind(size(A_total), offset:offset + ns -2, offset+1:offset + ns - 1);
    A_total(indices) = 0.1;

    offset = offset + ns ;
    %In the final state of the HMM, we can go to all other HMMs with equal
    %probability.(exceptions for zero, oh, leading and trailing silence :()
    %or interdigit silence
    %We can just sum the final state row of A_total with the trans_prob
    %vector scaled, to ensure each row sums to one.
    if(d < 12)
        A_total(offset-1,:) = A_total(offset-1,:) + 0.1.*trans_prob_digit;
    elseif(d == 12)
        A_total(offset-1,:) = A_total(offset-1,:) + 0.1.*trans_prob;
    elseif(d == 14)
        %You can only go to digits now
        A_total(offset-1,:) = A_total(offset-1,:) + 0.1.*trans_prob_silence;
    else
        A_total(offset-1,offset-1) = 1;
    end
end
SuperHMM.A = A_total;
SuperHMM.stateMap = stateMap;

num_total = 0;
denom_total = 0;

for fileIter = 1:length(featureDir)

    filename = featureDir(fileIter).name;
    b = featureDir(fileIter).data;
    [state_sequence,digit_sequence] = Viterbi_recognition(SuperHMM,b,gamma);
    %Now we want to convert the digit sequence of length T to an utterance
    %where the relevant digits only occur once. When a state jumps by more
    %then one, we know we are measuring a new digit, be careful for
    %interdigit silence as this has only one digit
    jump_indices = [find(abs(diff(state_sequence)) > 1); length(digit_sequence)];
    digits =  arrayfun(@(key) lookup_inv(key), digit_sequence(jump_indices), 'UniformOutput', false);
    
    sequence = split(filename,{'a','b'});
    sequence = sequence{1};
    sequence_est = join(string(digits), '');
    sequence_est = erase(sequence_est, ["q","r","s"]);

    num_total = num_total + editDistance(sequence,sequence_est);
    denom_total = denom_total + size(sequence,2);

    %Convert to a metric to test accuracy
    disp(fileIter)
end

WER = num_total/denom_total*100;


end