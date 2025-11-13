function [HMM_digit] = create_digit_HMM(numStates,outputDim,tag)
%Initialise a HMM for an entire digit
HMM_digit.tag = tag;
HMM_digit.numStates = numStates;

A = zeros(numStates);
for i = 1:numStates
    if i < numStates
        A(i,i) = 0.9;  % self-loop
        A(i,i+1) = 0.1; % next state
    else
        A(i,i) = 1.0; % final state self-loop
    end
end
    
HMM_digit.A = A;

HMM_digit.pi = [1 0 0];
    
    % --- Gaussian emission parameters (mean & covariance) ---
for s = 1:numStates
    HMM_digit.emission(s).mu  = ones(1, outputDim);       % random init
    HMM_digit.emission(s).Sigma = ones(1,outputDim);          % identity cov
    HMM_digit.emission(s).mixWeight = 1.0;               % one mixture
end

end