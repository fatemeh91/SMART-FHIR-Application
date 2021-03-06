function action_values = evalQNetwork1(S_t, WL1)%#codgen
% INPUT:
%   S_t             input feature vector      (1 x p)
%   WL1             weights for input layer   (num_nodes_L1, p+1 )
%   WL2             weights for output layer  (num_actions, num_nodes_L1+1)
% OUTPUT:
%   action_values   value of each of the actions within state S_t: Q(S_t,a_t)
%   d_WL1, d_WL2    gradients of the network outputs with respect to the network weights

%  Copyright 2014-2015   Shamim Nemati < shamim at seas.harvard.edu >
%  $Revision: 1.1 $  $Date: 02/01/2015  $

T = size(S_t,1);
%% input layer
inputL1 = [ones(T,1) S_t];
action_values = inputL1*WL1';
end