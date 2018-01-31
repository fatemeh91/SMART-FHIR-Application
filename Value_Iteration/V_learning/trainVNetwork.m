function [WL, fval] = trainVNetwork(DATA, WL, Reg, lambda, maxStochIter, MaxEpoch, Momentum_factor)

if nargin<3, Reg='L2'; end
if nargin<4, lambda=1e-3; end
if nargin<5, maxStochIter=3; end
if nargin<6, MaxEpoch=50; end
if nargin<7, Momentum_factor=0.1; end

options=[]; options.numDiff=0; options.GradObj='on'; options.MaxIter=MaxEpoch; options.maxStochIter= maxStochIter;
options.progTol = 1e-4;   options.MaxEpoch= MaxEpoch; options.Momentum_factor=Momentum_factor;  EXTRA.options=options;
EXTRA.Regulizer = Reg; EXTRA.lambda = lambda;
par_vec0=[];
EXTRA.dim_info.num_layers=numel(WL);
for l=1:EXTRA.dim_info.num_layers, par_vec0 = [par_vec0;WL{l}(:)];    EXTRA.dim_info.sizeWL{l}=size(WL{l}); end
% *** Call the optimizer
tic,[par_vec,fval] = runminFunc_batching(par_vec0, DATA, @RL_CostFunc, EXTRA); toc %
% *** Convert supervised learning par_vec to model parmeters
st=1;
for l=1:EXTRA.dim_info.num_layers
    WL{l} = reshape(par_vec(st:st-1+prod(EXTRA.dim_info.sizeWL{l})), EXTRA.dim_info.sizeWL{l}(1), EXTRA.dim_info.sizeWL{l}(2));
    st=st+prod(EXTRA.dim_info.sizeWL{l});
end
end

function [par_vec,fval] = runminFunc_batching(x0, DATA, func, EXTRA)
options = EXTRA.options; dims=EXTRA.dim_info; Reg = EXTRA.Regulizer;
lambda=[];
for l=1:dims.num_layers
    lambda_WL = ones(dims.sizeWL{l}); lambda_WL(1,:)=0; lambda =  [lambda ; lambda_WL(:)];
end
lambda = EXTRA.lambda * lambda;
if strcmpi(Reg,'L2'),       cost = @(w,RL,VStp1)penalizedL2(w, @(w)func(w, RL, VStp1, dims), lambda);
elseif strcmpi(Reg,'L1L2') % elastic net
    costL2 = @(w,RL,VStp1)penalizedL2(w, @(w)func(w, RL, VStp1, dims), lambda);
    cost = @(w,RL,VStp1)pseudoGradL1(@(w)costL2(w,RL,VStp1),w,lambda);
else   cost = @(w,RL,VStp1)func(w, RL, VStp1, dims);
end
funObj_validate = @(w,batch_indx,VStp1)getValPerf(w, DATA.data_train(batch_indx), VStp1, dims);
[par_vec, fval] = stochgradComplex_RL(cost, x0, options, DATA.data_train, funObj_validate, dims);
end

function VP = getValPerf(par_vec, RL_data, VStp1, dims)
st=1;  num_layers=dims.num_layers; WL=cell(num_layers,1);
for l=1:num_layers
    WL{l} = reshape(par_vec(st:st-1+prod(dims.sizeWL{l})), dims.sizeWL{l}(1), dims.sizeWL{l}(2));
    st=st+prod(dims.sizeWL{l});
end
N=numel(RL_data); L=zeros(1,N);
parfor sindex=1:N
    L(sindex) =  BackpropVNetwork(RL_data{sindex}.state_t, RL_data{sindex}.r_t, VStp1{sindex}, WL); %-loss
end
VP = -nanmean(L);
end

% addpath(genpath('/Users/Shamim/Documents/MATLAB/bnt/automatic_differentiation_codes/adimat'))
% gg = admDiffFD(@(par_vec0)objFun(par_vec0, bdata(2), VStp1(2)) , 1 , par_vec)';
% [e,g]=objFun(par_vec, bdata(2), VStp1(2));
% plot(g,'.'),hold on,plot(gg,'r-o')
