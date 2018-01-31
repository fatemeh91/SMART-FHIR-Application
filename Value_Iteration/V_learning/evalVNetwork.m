function values = evalVNetwork(state_t, WL, mex_flg)
% INPUT:
%   state_t 
%   WL
%   mex_flg
% OUTPUT:
%   values

if nargin<3, mex_flg=0; end
if mex_flg,    values = evalVNetwork_mex(state_t, WL);
else           values = evalVNetwork_not_mex(state_t, WL);
end
end


function values = evalVNetwork_not_mex(state_t, WL)

num_layers = numel(WL); values=[];
if num_layers==1, WL1=WL{1};
    values = evalVNetwork1(state_t, WL1);
elseif num_layers==2, WL1=WL{1}; WL2=WL{2};
    values = evalVNetwork2(state_t, WL1, WL2);
elseif num_layers==3, WL1=WL{1}; WL2=WL{2}; WL3=WL{3};
    values = evalVNetwork3(state_t, WL1, WL2, WL3);
elseif num_layers==4, WL1=WL{1}; WL2=WL{2}; WL3=WL{3};  WL4=WL{4};
    values = evalVNetwork4(state_t, WL1, WL2, WL3, WL4);
else
    disp('Warning: evalQNetwork encountered unsupported NN structure ...!')
end
end

function values = evalVNetwork_mex(state_t, WL)

num_layers = numel(WL); values=[];
if num_layers==1, WL1=WL{1};
    values = evalVNetwork1_mex(state_t, WL1);
elseif num_layers==2, WL1=WL{1}; WL2=WL{2};
    values = evalVNetwork2_mex(state_t, WL1, WL2);
elseif num_layers==3, WL1=WL{1}; WL2=WL{2}; WL3=WL{3};
    values = evalVNetwork3_mex(state_t, WL1, WL2, WL3);
elseif num_layers==4, WL1=WL{1}; WL2=WL{2}; WL3=WL{3};  WL4=WL{4};
    values = evalVNetwork4_mex(state_t, WL1, WL2, WL3, WL4);
else
    disp('Warning: evalQNetwork encountered unsupported NN structure ...!')
end
end