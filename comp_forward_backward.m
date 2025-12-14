function [gamma_log, loglik] = comp_forward_backward(HMM,data)

%The output will be an TxN matrix (time x total states)
T = size(data,1); %Number of time frames
D = size(data,2); %Dimension of one frame
N = size(HMM.stateMap,1); %Number of states

%Initialise the matrices with the probabilities, we will work with log probabilities 
A_log = log(HMM.A);
A_log_T = A_log';
alpha_log = zeros(T,N);
beta_log  = zeros(T,N);
gamma_log = zeros(T,N);
b_log = zeros(T,N);      %We will calculate all b's for all time frames while iterating over the states
                     %The j'th column corresponds to the b's of the j'th
                     %state

% Extract all mu and sigma into matrices
% mu: [1 x D], so mu_all: [N x D]
mu_all = vertcat(HMM.emission.mu);  % [N x D]
sigma_all = vertcat(HMM.emission.sigma);  % [N x D]

% Reshape for broadcasting
% data: [T x D] -> [T x 1 x D]
data_reshaped = reshape(data, [size(data, 1), 1, size(data, 2)]);  % [T x 1 x D]

% mu_all: [N x D] -> [1 x N x D]
mu_reshaped = reshape(mu_all, [1, size(mu_all, 1), size(mu_all, 2)]);  % [1 x N x D]

% sigma_all: [N x D] -> [1 x N x D]
sigma_reshaped = reshape(sigma_all, [1, size(sigma_all, 1), size(sigma_all, 2)]);  % [1 x N x D]

% Vectorized computation using broadcasting
diff = data_reshaped - mu_reshaped;  % [T x N x D]
diff_squared = diff.^2;

log_term = -0.5 * log(2 * pi * sigma_reshaped);
ratio_term = -diff_squared ./ (2 * sigma_reshaped);

% Sum across D dimension (dimension 3)
b_log = sum(log_term + ratio_term, 3);  % [T x N]


%Generate the alpha's for the other t's using recursion

%generate alpha for the first time index
alpha_log(1,:) = log(HMM.pi) + b_log(1,:);

% Generate the alpha's for the other t's using recursion
for t = 2:T  
    sum_term_alpha = alpha_log(t-1,:) + A_log_T;
    
    % Find maximum along rows
    max_values_alpha = max(sum_term_alpha, [], 2);
    
    % Compute second term with log-sum-exp trick
    % Subtract max for numerical stability and avoid underflow
    exp_term = exp(sum_term_alpha - max_values_alpha);
    
    % Sum and take log, add back the max
    log_sum_exp = max_values_alpha + log(sum(exp_term, 2));
    
    % Handle -Inf cases (log(0))
    log_sum_exp(~isfinite(max_values_alpha)) = -Inf;
    
    % Update alpha_log
    alpha_log(t,:) = b_log(t,:) + log_sum_exp';
end

beta_log(T,:) = 0;

for t = T-1:-1:1
    sum_term = A_log + (b_log(t+1,:) + beta_log(t+1,:));
    % Find max for each row
    max_val = max(sum_term, [], 2);
    % Subtract max for numerical stability
    centered = sum_term - max_val;
    exp_term = exp(centered);
    % Sum across columns
    sum_exp = sum(exp_term, 2);
    % Compute log-sum-exp
    beta_log(t,:) = (max_val + log(sum_exp))';
end

%Because we know our emission model only consists of one component, (so
%no mixing), the gamma(i,j) = gamma(i)
%The gamma's represent the a posteriori probability or the probability of
%being in a state i at time t, given the observations and the model
%estimate
num_gamma = alpha_log + beta_log;
max_values_gamma = max(num_gamma,[],2);
exp_gamma = exp(num_gamma - max_values_gamma);
denom_gamma = max_values_gamma + log(sum(exp_gamma,2));

gamma_log = num_gamma - denom_gamma; 
%Looking at the data, you wouldn't think they sum to one per t
%but apparently the values are so small that they actually do sum to one

% Utterance log-likelihood from forward messages:
loglik = logsumexp(alpha_log(end,:));     % == log p(X | HMM)
end

function s = logsumexp(x)
    m = max(x);
    s = m + log(sum(exp(x - m)));
end