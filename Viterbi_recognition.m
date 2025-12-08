function [best_path, digit_path] = Viterbi_recognition(HMM, data,gamma)
% Vectorised Viterbi with word-insertion penalty gamma
%
% HMM.gamma = word insertion penalty
%   gamma > 0 → fewer insertions (more deletions)
%   gamma < 0 → fewer deletions (more insertions)

T = size(data,1);      % time frames
N = size(HMM.stateMap,1);
%gamma = -2;     % new insertion/deletion control

%% -------------------------------------------------------------
%  EMISSION LOG-LIKELIHOODS  b_log(t,j)
%% -------------------------------------------------------------

b_log = zeros(T, N);

for j = 1:N
    mu    = HMM.emission(j).mu;
    sigma = HMM.emission(j).sigma;

    diff   = data - mu; 
    result = -0.5*log(2*pi*sigma) - (diff).^2 ./ (2*sigma);
    b_log(:,j) = sum(result,2);
end


%% -------------------------------------------------------------
%  PRECOMPUTE STATIC MATRICES FOR VECTORISATION
%% -------------------------------------------------------------

logA = log(HMM.A);        % N×N
V_log = zeros(N,T);
local_map = zeros(N,T);
global_map = cell(N,T);

% Precompute a mask indicating word-boundary transitions (i→j)
digit_id = HMM.stateMap(:,2);     % size N×1
boundary_mask = digit_id ~= digit_id.';   % N×N logical
boundary_cost = gamma * boundary_mask;    % add this to logA


%% -------------------------------------------------------------
%  INITIALISATION
%% -------------------------------------------------------------

V_log(:,1) = (log(HMM.pi) + b_log(1,:))';

for j = 1:N
    global_map{j,1} = j;
end


%% -------------------------------------------------------------
%  RECURSION (fully vectorised)
%% -------------------------------------------------------------

for t = 2:T

    % expand previous V_log into N×N (column j receives all prev states)
    prevScores = V_log(:,t-1);

    % combine scores:
    % each column j: prevScores(i) + logA(i,j) + boundaryCost(i,j)
    probs = prevScores + logA + boundary_cost;

    % add emission for time t (broadcast to each column)
    probs = probs + b_log(t,:);   % adds row vector to each column

    % choose best predecessor for each state j
    [V_log(:,t), local_map(:,t)] = max(probs, [], 1);

    % update paths
    for j = 1:N
        prev = local_map(j,t);
        global_map{j,t} = [global_map{prev,t-1}, j];
    end
end


%% -------------------------------------------------------------
%  TERMINATION
%% -------------------------------------------------------------

[~, best_final_state] = max(V_log(:,T));
best_path = global_map{best_final_state, T}';

digit_path = HMM.stateMap(best_path,2);

end
