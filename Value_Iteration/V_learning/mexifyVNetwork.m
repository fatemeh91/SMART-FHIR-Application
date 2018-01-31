function mexifyVNetwork(WL,Exp)
if Exp.num_layers==1
    WL1=WL{1};
    codegen   -config:mex -args {zeros(1,Exp.NN_input),WL1} g_evalVNetwork1.m
    codegen   -config:mex -args {zeros(1,Exp.NN_input),WL1} evalVNetwork1.m
elseif Exp.num_layers==2
    WL1=WL{1}; WL2=WL{2};
    codegen  -config:mex -args {zeros(1,Exp.NN_input),WL1,WL2} g_evalVNetwork2.m
    codegen  -config:mex -args {zeros(1,Exp.NN_input),WL1,WL2} evalVNetwork2.m
elseif Exp.num_layers==3
    WL1=WL{1}; WL2=WL{2}; WL3=WL{3};
    codegen  -config:mex -args {zeros(1,Exp.NN_input),WL1,WL2,WL3} g_evalVNetwork3.m
    codegen  -config:mex -args {zeros(1,Exp.NN_input),WL1,WL2,WL3} evalVNetwork3.m
elseif Exp.num_layers==4
    WL1=WL{1}; WL2=WL{2}; WL3=WL{3}; WL4=WL{4};
    codegen  -config:mex -args {zeros(1,Exp.NN_input),WL1,WL2,WL3,WL4} g_evalVNetwork4.m
    codegen  -config:mex -args {zeros(1,Exp.NN_input),WL1,WL2,WL3,WL4} evalVNetwork4.m
end

system('mv *_mex.mex*  codegen');
system('rm -rf codegen/mex');
addpath('codegen');