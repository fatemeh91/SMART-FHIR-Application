clear all
fold_indx=1;
Exp.class=2; Exp.p=5; 
%%
addpath(genpath('/Users/Shamim/Documents/MATLAB/Optimization/minFunc/'))
addpath ('/Users/Shamim/Documents/MATLAB/Datasets/utility/','/Users/Shamim/Documents/MATLAB/LogisticRegression/')
addpath('/Users/Shamim/Documents/MATLAB/RL/SKF_VNetwork/','/Users/Shamim/Documents/MATLAB/RL/V_learning/')
addpath(genpath('/Users/Shamim/Documents/MATLAB/bnt/DTBox-0.5'),'/Users/Shamim/Documents/MATLAB/bnt/subOptSLDS')
%%
% A "mode" is define as a set of AR coefficients and the corresponding
% noise covariance matrix {A_i, Q_i}, for i=1..J (here J=4)
%
% Generate Bivariate Time-series from three categories (Z1, Z2, Z3), with different
% proportions of time spent within each of four dynamical modes below:

Z1 = [0.97 0.01 0.01 0.01; 0.01 0.97 0.01 0.01; 0.1 0.1 0.7 0.1; 0.1 0.1 0.1 0.7];
Z2 = [0.7 0.1 0.1 0.1; 0.1 0.7 0.1 0.1; 0.01 0.01 0.97 0.01; 0.01 0.01 0.01 0.97];

% multiple output example:
A(:,:,1) = [[[-0.5 -0.1;0.4 -0.4], [0.3 -0.3;0.4 -0.3]]; [eye(2,2) zeros(2,2)]];
A(:,:,2) = [[[-0.3 -0.7;0.5 -0.3], [0.6 -0.6;0.2 -0.6]]; [eye(2,2) zeros(2,2)]];
A(:,:,3) = [[[0.5 -0.2;0.2 -0.5], [0.6 -0.4;0.4 -0.5]]; [eye(2,2) zeros(2,2)]];
A(:,:,4) = [[[-0.5 0.2;0.2 -0.6], [0.6 -0.3;0.4 -0.2]]; [eye(2,2) zeros(2,2)]];

C(:,:,1) = [1 0 0 0;0 1 0 0]; C(:,:,2) = [1 0 0 0;0 1 0 0];  C(:,:,3) = [1 0 0 0;0 1 0 0]; C(:,:,4) = [1 0 0 0;0 1 0 0];
Q(:,:,1) = [sqrt(1) 0 0 0;0,sqrt(4) 0 0 ; 0 0 1e-3 0; 0 0 0 1e-3]; Q(:,:,2) = [sqrt(2) 0 0 0;0,sqrt(3) 0 0 ; 0 0 1e-3 0; 0 0 0 1e-3];
Q(:,:,3) = [sqrt(3) 0 0 0;0,sqrt(2) 0 0 ; 0 0 1e-3 0; 0 0 0 1e-3]; Q(:,:,4) = [sqrt(4) 0 0 0;0,sqrt(1) 0 0 ; 0 0 1e-3 0; 0 0 0 1e-3];
R(:,:,1) = [sqrt(0.001) 0;0 sqrt(0.001)]; R(:,:,2) = [sqrt(0.01) 0;0 sqrt(0.01)]; R(:,:,3) = [sqrt(0.1) 0;0 sqrt(0.1)]; R(:,:,4) = [sqrt(1) 0;0 sqrt(1)];R = zeros(size(R));
init_state = zeros(4,4);
% 21 simulated subjects from the three categories Z1, Z2, Z3
NUM_Subjects = 200;
OUTCOME = -1*ones(NUM_Subjects,1); OUTCOME(NUM_Subjects/2+1:end)=1;
T = 300;
for s_ind = 1:NUM_Subjects
    if     OUTCOME(s_ind)==-1, Z = Z1;
    elseif OUTCOME(s_ind)==1, Z = Z2;
    else warning('Unknown Category')
    end
    models=zeros(1,T);models(1) = 1;
    for t = 2:T
        models(t) = find(rand>[0 cumsum(Z(models(t-1),:))],1,'last');
    end
    [x,y] = sample_lds(A, C, Q, R, init_state, T, models);
    DATA{s_ind} = y(:,101:T);
    True_seg{s_ind}=models(101:T);
end
%%
subplot(2,1,1),plot(DATA{end}')
subplot(2,1,2),plot(True_seg{end},'ro')
%%
kk=0;r_t_all=[];tmp=[];
for k=1:numel(DATA)
    kk=kk+1;
    DATA_RL{k}.y = convert_to_lagged_form(DATA{k}, Exp.p);  T=size(DATA_RL{k}.y,2);
    DATA_RL{k}.covar = ones(size(DATA_RL{k}.y,2),0); % RL may need previous action
    DATA_RL{k}.r_t = zeros(T-1,1); DATA_RL{k}.r_t(end)=0.1*OUTCOME(k);
end
%% ----------- Setup -----------
clc, fold_index=1; Exp.QNet_MaxIter=100; Exp.maxStochIter=5;  Exp.Momentum_factor=1e-1;
Exp.num_action = 1;   Exp.num_layers=3; % 1, 2, 3, 4
Exp.beta_fac=[1 1 1]; Exp.L1=10; Exp.L2=5; Exp.L3=1; 
Exp.covar_dim=size(DATA_RL{1}.covar,2); Exp.obs_dim=size(DATA_RL{1}.y,1); 
Exp.NN_input = Exp.obs_dim  + Exp.covar_dim ;  Exp.num_outputs = 1; 
WLinit = initializeNNWeights(Exp); 

% training and testing data
rng(1000*fold_index,'twister'); Nall=numel(DATA_RL);indices = randperm(Nall);
index_train = indices(1:round(0.70*Nall)); index_test = indices(round(0.70*Nall)+1:Nall);
data.data_train=DATA_RL(index_train);  data.data_test=DATA_RL(index_test);
outcome.outcome_train = OUTCOME(index_train); outcome.outcome_test = OUTCOME(index_test);
%% mexify
 mexifyVNetwork(WLinit, Exp)

%% initialize Q-Network parameters (supervised training of the Q-Network)
RL_data.data_train=data.data_train;
for k=1:numel(data.data_train)
    T = size(data.data_train{k}.y,2);
    RL_data.data_train{k}.state_t = [data.data_train{k}.y(:,1:end-1)'/100  data.data_train{k}.covar(1:end-1,:)];
    RL_data.data_train{k}.state_tp1 = [data.data_train{k}.y(:,2:end)'/100  data.data_train{k}.covar(1:end-1,:)];
end
[WL, fval] = trainVNetwork(RL_data, WLinit, 'L1L2', 1e-4, Exp.maxStochIter, Exp.QNet_MaxIter, Exp.Momentum_factor);
hold on, plot(fval.fval), plot(fval.fval_validation,'r')


%% training performance
outcome_hat=[];
for subj_indx=1:numel(data.data_train)    
    values = evalVNetwork(RL_data.data_train{subj_indx}.state_t, WL);
    outcome_hat(subj_indx)=quantile(values,0.9)-quantile(values,0.1);
end
%
subplot(2,1,1),plot(values),
stats = Performance_Stats(outcome.outcome_train(:),outcome_hat(:),1);
