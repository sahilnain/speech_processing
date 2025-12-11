function [gamma_log] = comp_forward_backward(HMM,data)

%Compute the alpha's using the forward procedure
%These represent the probability of observations up until time t, given we
%are at a state i at that time index

%Compute the beta's using the backward procedure
%These represent...


%The output will be an TxN matrix (time x total states)
T = size(data,1); %Number of time frames
D = size(data,2); %Dimension of one frame
N = size(HMM.stateMap,1); %Number of states

%Initialise the matrices with the probabilities, we will work with log probabilities 
alpha_log = zeros(T,N);
beta_log  = zeros(T,N);
gamma_log = zeros(T,N);
b_log = zeros(T,N);      %We will calculate all b's for all time frames while iterating over the states
                     %The j'th column corresponds to the b's of the j'th
                     %state

%Determine the b probabilities
% Vectorised computation of b_log for Gaussian emissions with independent dimensions

% Number of states
N = numel(HMM.emission);

% Stack emission parameters into matrices: N × D
mu    = vertcat(HMM.emission.mu);      % N × D
sigma = vertcat(HMM.emission.sigma);   % N × D

% Dimensions
[M, D] = size(data);

% Expand data (M × 1 × D), mu and sigma (1 × N × D)
data_expanded  = reshape(data,  [M 1 D]);
mu_expanded    = reshape(mu,    [1 N D]);
sigma_expanded = reshape(sigma, [1 N D]);

% Compute log Gaussian contributions for all states & all data
result = -0.5 * log(2*pi*sigma_expanded) ...
         - ( (data_expanded - mu_expanded).^2 ./ (2*sigma_expanded) );

% Sum over dimensions to get final b_log (M × N)
b_log = sum(result, 3);

%generate alpha for the first time index
alpha_log(1,:) = log(HMM.pi) +b_log(1,:);
%generate alpha for the last time index
beta_log(T,:) = 0;

for t = T-1:-1:1
    % sum_term = log(A) + b_log(t+1,:) + beta_log(t+1,:);
    sum_term = log(HMM.A) + (b_log(t+1,:) + beta_log(t+1,:));  

    max_val = max(sum_term,[],2);
    beta_log(t,:) = max_val + log(sum(exp(sum_term - max_val),2));
end



%Generate the alpha's for the other t's using recursion

for t = 2:T
%The second term there is a log of a large sum, this is numerically unstable
%We thus rewrite the expression in order to improve stability

%The sum is: log(sum(exp(alpha_log's)*a_ij's)) = log(sum(exp(alpha_log's +
%log(a_ij's)))
sum_term_alpha = alpha_log(t-1,:) + log(HMM.A)';
max_values_alpha = max(sum_term_alpha,[],2);
second_term_alpha = max_values_alpha + log(sum(exp(sum_term_alpha - max_values_alpha),2));

%We get some Nan values here when the max is -inf, fix this by putting it
%back to -Inf
second_term_alpha(isnan(second_term_alpha)) = -Inf;
alpha_log(t,:) = b_log(t,:) + second_term_alpha';

end

%Because we know our emission model only consists of one component, (so
%no mixing), the gamma(i,j) = gamma(i)
%The gamma's represent the a posteriori probability or the probability of
%being in a state i at time t, given the observations and the model
%estimate
num_gamma = alpha_log + beta_log;
max_values_gamma = max(num_gamma,[],2);
denom_gamma = max_values_gamma + log(sum(exp(num_gamma - max_values_gamma),2));

gamma_log = num_gamma - denom_gamma; 
%Looking at the data, you wouldn't think they sum to one per t
%but apparently the values are so small that they actually do sum to one
end