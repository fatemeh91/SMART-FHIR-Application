function [value, d_WL1, d_WL2, d_WL3, d_WL4, d_St] = g_evalVNetwork4(S_t, WL1, WL2, WL3, WL4)%#codgen
% INPUT:
%   S_t             input feature vector      (1 x p)
%   WL1             weights for input layer   (num_nodes_L1, p+1 )
%   WL2             weights for input layer   (num_nodes_L2, num_nodes_L1+1)
%   WL3             weights for output layer  (num_nodes_L3, num_nodes_L2+1)
%   WL4             weights for output layer  (num_actions, num_nodes_L3+1)
% OUTPUT:
%   value   value of each of the actions within state S_t: Q(S_t,a_t)
%   d_WL1, d_WL2, d_WL3, d_WL4    gradients of the network outputs with respect to the network weights

%  Copyright 2014-2015   Shamim Nemati < shamim at seas.harvard.edu >
%  $Revision: 1.1 $  $Date: 02/01/2015  $

T = size(S_t,1);  num_features_p1=size(WL1,2);
%% input layer
inputL1 = [ones(T,1) S_t];
etaL1 = inputL1*WL1';
sigmaL1 = actG(etaL1);
%% middle layer 1
inputL2 = [ones(T,1) sigmaL1];
etaL2 = inputL2*WL2';
sigmaL2 = actG(etaL2);
%% middle layer 2
inputL3 = [ones(T,1) sigmaL2];
etaL3 = inputL3*WL3';
sigmaL3 = actG(etaL3);
%% output layer
inputL4 = [ones(T,1) sigmaL3];
value = inputL4*WL4';

if nargout>1
    d_WL4 = zeros(size(WL4,1),size(WL4,2),1);
    d_WL3 = zeros(size(WL3,1),size(WL3,2),1);
    d_WL2 = zeros(size(WL2,1),size(WL2,2),1);
    d_WL1 = zeros(size(WL1,1),size(WL1,2),1);  d_St=zeros(num_features_p1-1,1,T);
    
    for t=1:T
        %% backpropagate through the output layer
        d_WL4 =  d_WL4+  inputL4(t,:);
        d_inputL4= WL4' ;
        %% backpropagate through the middle layer 2
        d_etaL3 = d_inputL4(2:end)' .*   g_actG(etaL3(t,:));
        d_WL3 =  d_WL3 + d_etaL3' * inputL3(t,:);
        d_inputL3 = WL3' * d_etaL3';
        %% backpropagate through the middle layer 1
        d_etaL2 = d_inputL3(2:end)' .*   g_actG(etaL2(t,:));
        d_WL2 =  d_WL2 + d_etaL2' * inputL2(t,:);
        d_inputL2= WL2' * d_etaL2';
        %% backpropagate through the input layer
        d_etaL1 =       d_inputL2(2:end)' .* g_actG(etaL1(t,:));
        d_WL1 =  d_WL1 + d_etaL1' * inputL1(t,:);
        d_St(:,t)= d_etaL1 * WL1(:,2:end);
    end
end
end
