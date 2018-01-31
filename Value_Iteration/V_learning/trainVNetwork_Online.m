function [WL, fval] = trainQNetwork_Online(DATA, WL, Reg, lambda, MaxEpoch)

if nargin<3, Reg='L2'; end
if nargin<4, lambda=1e-3; end
if nargin<5, MaxEpoch=3; end

minFuncArgs.derivativeCheck = 'off'; minFuncArgs.display = 'off';
minFuncArgs.MaxIter = MaxEpoch;
minFuncArgs.TolFun = 1e-4; minFuncArgs.TolX = 1e-4;
minFuncArgs.GradObj = 'on'; minFuncArgs.Method = 'lbfgs';

par_vec0=[];
dims.num_layers=numel(WL); 
for l=1:dims.num_layers, par_vec0 = [par_vec0;WL{l}(:)];    dims.sizeWL{l}=size(WL{l}); end
% *** Call the optimizer

% lam=[];
% for l=1:dims.num_layers
%     lambda_WL = ones(dims.sizeWL{l}); lambda_WL(1,:)=0; lam =  [lam ; lambda_WL(:)];
% end
% lambda = lambda * lam;
% if strcmpi(Reg,'L2'),       cost = @(w,RL,QStp1)penalizedL2(w, @(w)RL_CostFunc(w, RL, QStp1, dims), lambda);
% elseif strcmpi(Reg,'L1L2') % elastic net
%     costL2 = @(w,RL,QStp1)penalizedL2(w, @(w)RL_CostFunc(w, RL, QStp1, dims), lambda);
%     cost = @(w,RL,QStp1)pseudoGradL1(@(w)costL2(w,RL,QStp1),w,lambda);
% else   cost = @(w,RL,QStp1)RL_CostFunc(w, RL, QStp1, dims);
% end

VStp1=cell(numel(DATA),1);
parfor iindx=1:numel(DATA),    VStp1{iindx} = evalVNetwork(DATA{iindx}.state_tp1, WL); end
[par_vec, fval] = minFunc(@RL_CostFunc, par_vec0, minFuncArgs, DATA, VStp1, dims);

% *** Convert supervised learning par_vec to model parmeters
st=1;
for l=1:dims.num_layers
    WL{l} = reshape(par_vec(st:st-1+prod(dims.sizeWL{l})), dims.sizeWL{l}(1), dims.sizeWL{l}(2));
    st=st+prod(dims.sizeWL{l});
end

% addpath(genpath('/Users/Shamim/Documents/MATLAB/bnt/automatic_differentiation_codes/adimat'))
% gg = admDiffFD(@(par_vec0)cost(par_vec0, bdata(2), QStp1(2)) , 1 , par_vec)';
% [e,g]=cost(par_vec, bdata(2), QStp1(2));
% plot(g,'.'),hold on,plot(gg,'r-o')
