function [alpha] = comp_forward(HMM,data)

%Compute the alpha's using the forward procedure
%These represent the probability of observations up until time t, given we
%are at a state i at that time index
%The output will be an TxN matrix (time x total states)

T = size(data,1); %Number of time frames
D = size(data,2); %Dimension of one frame
N = size(HMM.stateMap,1); %Number of states

log_alpha = zeros(T,N);
log_b = zeros(T*D,N); %We will calculate all b's for all time frames while iterating over the states
                     %The j'th column corresponds to the b's of the j'th
                     %state

%Initialise the first row of alpha
%Fill in o1 in all of the b's

%Do it per state because i'm lazy
for i = 1:N
model_em = HMM.emission(1);

mu = HMM.emission(i).mu;           
sigma = HMM.emission(i).Sigma; 
% regularize tiny variances
sigma = sigma + 1e-6;

diff = data - mu; 
result = 0.5*log(2*pi*sigma) - (diff).^2 ./(2*sigma.^2);
log_b(:,i) = reshape(result', [], 1);
end

%alpha(1,:) = HMM.pi



end