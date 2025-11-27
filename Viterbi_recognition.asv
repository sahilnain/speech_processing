function [best_path,digit_path] = Viterbi_recognition(HMM,data)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%The output will be an TxN matrix (time x total states)
T = size(data,1); %Number of time frames
D = size(data,2); %Dimension of one frame
N = size(HMM.stateMap,1); %Number of states

%Initialise the matrices with the probabilities, we will work with log probabilities 

b_log = zeros(T,N);  %We will calculate all b's for all time frames while iterating over the states
                     %The j'th column corresponds to the b's of the j'th
                     %state

%Determine the b probabilities
for i = 1:N

mu = HMM.emission(i).mu;           
sigma = HMM.emission(i).sigma; 

diff = data - mu; 
result = -0.5*log(2*pi*sigma) - (diff).^2 ./ (2*sigma);%0.5*log(2*pi*sigma) - (diff).^2 ./(2*sigma.^2);

%If we assume the coëfficients D are independant, we take the product of all D
% components of b. This is becomes a sum in the log domain. So sum a row
% together
b_log(:,i) = sum(result,2);
end

%We will now compute the max probability for each of the states at time t,
%along with the mapping of from which state each maximum came from. This
%will allow us to extract the optimal state sequence at t = T
V_log = zeros(N,T);
local_map =  zeros(N,T);
global_map = cell(N,T);

%Initialise by calculating the log_prob of V_1(j)
V_log(:,1) = (log(HMM.pi) + b_log(1,:))';

for i = 1:N
    global_map{i,1} = i;  % Each path starts with its state
end

%AI Generated:
% Recursion (t = 2 to T)
for t = 2:T
    
        %Select the column of V at time t-1 and sum with all of the columns
        %of log(A). For one column, add the log_probs of bj(ot) to all rows
        probs =(V_log(:,t-1) + log(HMM.A))+ b_log(t,:);
        
        % Find the best previous state
        [max_prob, best_prev] = max(probs);
        V_log(:, t) = max_prob;
        local_map(:, t) = best_prev;
    for j = 1:N
        % Update path history: append current state to best previous path
        global_map{j, t} = [global_map{best_prev(j), t-1}, j];
    end
    
    
end

    % Termination
    [max_prob, best_final_state] = max(V_log(:, T));
    
    % Backtrack to find best path
    best_path = global_map{best_final_state, T};
    %Translate to the digit HMMs using the state map
    digit_path = HMM.stateMap(best_path,2);

end