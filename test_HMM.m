b = load('HMM_train.mat');
HMMs = b.HMMs;

%Combine all of the HMMs into a super HMM with a large intial state
%probabilty and large transition probability matrix
stateMap = [];
SuperHMM.emission = [];
SuperHMM.pi = [];
totalStates = 0;
Statevec    = zeros(size(HMMs,2) -1 ,1);

for d = 1:size(HMMs,2)-1

    SuperHMM.emission = [ SuperHMM.emission ,HMMs(d).emission];
    ns = HMMs(d).numStates;
    Statevec(d) = ns;

    for s = 1:ns
        % Keep track of which global state belongs to which digit/state
        % rows: [globalIndex, digit, localState]
        stateMap = [stateMap; totalStates+s, d, s];
    end 
    
    if(d <= 9) %Digits 1-9
        SuperHMM.pi = [SuperHMM.pi , 1, zeros(1,ns-1)];
    elseif(d <= 11)      %Zeros
         SuperHMM.pi = [SuperHMM.pi , 0.5, zeros(1,ns-1)];  
    elseif(d == 12)      %Leading silence
         SuperHMM.pi = [SuperHMM.pi , 1, zeros(1,ns-1)]; 
         Leading_silence_index = totalStates +1;
    else                %Trailing silence
        SuperHMM.pi = [SuperHMM.pi , zeros(1,ns)];
        Trailing_silence_index = totalStates +1;
    end

    totalStates = totalStates + ns;
end
pi_sum = sum(SuperHMM.pi);
SuperHMM.pi = SuperHMM.pi./pi_sum;

A_total = eye(totalStates,totalStates)*0.7;
%Loop over all of the states of the HMMs again to construct the transition
%probability matrix A, digits can't connect to leading silence, trailing
%silence can't connect to digits
offset = 1;
trans_prob = SuperHMM.pi;
%Change the leading silence and trailing silence probs
trans_prob(Trailing_silence_index) = trans_prob(Leading_silence_index);
trans_prob(Leading_silence_index) = 0;
for d = 1:size(HMMs,2)-1
    ns = HMMs(d).numStates;

    %When we're not in the final state of one of the HMMs, we can only go
    %to the next state within that HMM or remain in the same state
    indices = sub2ind(size(A_total), offset:offset + ns -2, offset+1:offset + ns - 1);
    A_total(indices) = 0.3;

    offset = offset + ns ;
    %In the final state of the HMM, we can go to all other HMMs with equal
    %probability.(exceptions for zero, oh, leading and trailing silence :()
    %We can just sum the final state row of A_total with the trans_prob
    %vector scaled, to ensure each row sums to one.
    if(d ~= 13)
        A_total(offset-1,:) = A_total(offset-1,:) + 0.3.*trans_prob;
    else
        A_total(offset-1,offset-1) = 1;
    end
end
SuperHMM.A = A_total;
SuperHMM.stateMap = stateMap;

%Now that we have our SUPER HMM, we can start the Viterbi algorithm
%(HOW MUCH DO WE NEED TO HAVE LESS STATES THAN OBSERVATIONS HERE?)
featureFile = "FBank_test\test";
featureDir = dir(fullfile(featureFile,'**/*.*'));

sdir = [featureDir.isdir];
one = logical(ones(length(featureDir),1));
removeMask = (sdir' == one);
featureDir(removeMask) = [];

for fileIter = 1:length(featureDir)

    filename = featureDir(fileIter).name;
    b = readNPY(append(featureDir(fileIter).folder,'/',featureDir(fileIter).name));
    [state_sequence,digit_sequence] = Viterbi_recognition(SuperHMM,b);



    disp(fileIter)







end


