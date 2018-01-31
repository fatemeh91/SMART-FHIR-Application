function action_values = evalQNetwork3(S_t, WL1, WL2, WL3)%#codgen
% INPUT:
%   S_t             input feature vector      (1 x p)
%   WL1             weights for input layer   (num_nodes_L1, p+1 )
%   WL2             weights for input layer   (num_nodes_L2, num_nodes_L1+1)
%   WL3             weights for output layer  (num_actions, num_nodes_L2+1)
% OUTPUT:
%   action_values   value of each of the actions within state S_t: Q(S_t,a_t)
%   d_WL1, d_WL2, d_WL2    gradients of the network outputs with respect to the network weights

%  Copyright 2014-2015   Shamim Nemati < shamim at seas.harvard.edu >
%  $Revision: 1.1 $  $Date: 02/01/2015  $

T = size(S_t,1);
%% input layer
inputL1 = [ones(T,1) S_t];
etaL1 = inputL1*WL1';
sigmaL1 = actG(etaL1);
%% middle layer
inputL2 = [ones(T,1) sigmaL1];
etaL2 = inputL2*WL2';
sigmaL2 = actG(etaL2);
%% output layer
inputL3 = [ones(T,1) sigmaL2];
action_values = inputL3*WL3';
end
