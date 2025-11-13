function [alpha_log,beta_log] = comp_forward_backward(HMM,data)

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
b_log = zeros(T,N);      %We will calculate all b's for all time frames while iterating over the states
                     %The j'th column corresponds to the b's of the j'th
                     %state

%Determine the b probabilities
for i = 1:N

mu = HMM.emission(i).mu;           
sigma = HMM.emission(i).Sigma; 

diff = data - mu; 
result = 0.5*log(2*pi*sigma) - (diff).^2 ./(2*sigma.^2);

%If we assume the coëfficients D are independant, we take the product of all D
% components of b. This is becomes a sum in the log domain. So sum a row
% together

b_log(:,i) = sum(result,2);
end


%generate alpha for the first time index
alpha_log(1,:) = log(HMM.pi) +b_log(1,:);
%generate alpha for the last time index
beta_log(T,:) = 0;

%Generate the alpha's for the other t's using recursion

for t = 2:T
%The second term here is a log of a large sum, this is numerically unstable
%We thus rewrite the expression in order to improve stability

%The sum is: log(sum(exp(alpha_log's)*a_ij's)) = log(sum(exp(alpha_log's +
%log(a_ij's)))
sum_term_alpha = alpha_log(t-1,:) + log(HMM.A)'; %I think there should be a transpose here, maybe ask the prof
max_values_alpha = max(sum_term_alpha,[],2);
second_term_alpha = max_values_alpha + log(sum(exp(sum_term_alpha - max_values_alpha),2));

%Now we calculate the beta's
%For this we now have a log of a sum of an exp of a sum of three terms
sum_term_beta = b_log(end-t+2,:) + beta_log(end-t+2,:) + log(HMM.A);
max_values_beta = max(sum_term_beta,[],2);
second_term_beta = max_values_beta + log(sum(exp(sum_term_beta - max_values_beta),2));

%We get some Nan values here when the max is -inf, fix this by putting it
%back to -Inf
second_term_alpha(isnan(second_term_alpha)) = -Inf;
second_term_beta(isnan(second_term_beta)) = -Inf;

alpha_log(t,:) = b_log(t,:) + second_term_alpha';
beta_log(end-t+1,:) =  second_term_beta';



end




end