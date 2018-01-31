function WLinit = initializeNNWeights(Exp)
rng(1e5)
if Exp.num_layers==1
    WLinit{1} = Exp.beta_fac(1)*sqrt(6/(Exp.num_outputs+Exp.NN_input+1))*randn(Exp.num_outputs,Exp.NN_input+1);
elseif Exp.num_layers==2
    WLinit{1} = Exp.beta_fac(1)*sqrt(6/(Exp.L1+Exp.NN_input+1))*randn(Exp.L1,Exp.NN_input+1);
    WLinit{2} = Exp.beta_fac(2)*sqrt(6/(Exp.num_outputs+Exp.L1+1))*randn(Exp.num_outputs,Exp.L1+1);
elseif Exp.num_layers==3
    WLinit{1} = Exp.beta_fac(1)*sqrt(6/(Exp.L1+Exp.NN_input+1))*randn(Exp.L1,Exp.NN_input+1);
    WLinit{2} = Exp.beta_fac(2)*sqrt(6/(Exp.L2+Exp.L1+1))*randn(Exp.L2,Exp.L1+1);
    WLinit{3} = Exp.beta_fac(3)*sqrt(6/(Exp.num_outputs+Exp.L2+1))*randn(Exp.num_outputs,Exp.L2+1);
elseif Exp.num_layers==4
    WLinit{1} = Exp.beta_fac(1)*sqrt(6/(Exp.L1+Exp.NN_input+1))*randn(Exp.L1,Exp.NN_input+1);
    WLinit{2} = Exp.beta_fac(2)*sqrt(6/(Exp.L2+Exp.L1+1))*randn(Exp.L2,Exp.L1+1);
    WLinit{3} = Exp.beta_fac(3)*sqrt(6/(Exp.L3+Exp.L2+1))*randn(Exp.L3,Exp.L2+1);
    WLinit{4} = Exp.beta_fac(4)*sqrt(6/(Exp.num_outputs+Exp.L3+1))*randn(Exp.num_outputs,Exp.L3+1);
end
