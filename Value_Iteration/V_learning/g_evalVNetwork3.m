function [value, d_WL1, d_WL2, d_WL3, d_St] = g_evalVNetwork3(S_t, WL1, WL2, WL3)%#codgen
% INPUT:
%   S_t             input feature vector      (1 x p)
%   WL1             weights for input layer   (num_nodes_L1, p+1 )
%   WL2             weights for input layer   (num_nodes_L2, num_nodes_L1+1)
%   WL3             weights for output layer  (num_actions, num_nodes_L2+1)
% OUTPUT:
%   value   value of each of the actions within state S_t: V(S_t)
%   d_WL1, d_WL2, d_WL3    gradients of the network outputs with respect to the network weights

%  Copyright 2014-2015   Shamim Nemati < shamim at seas.harvard.edu >
%  $Revision: 1.1 $  $Date: 02/01/2015  $

T = size(S_t,1);  num_features_p1=size(WL1,2);
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
etaL3 = WL3*inputL3';
value = etaL3';

if nargout>1
    d_WL3 = zeros(size(WL3,1),size(WL3,2),1);
    d_WL2 = zeros(size(WL2,1),size(WL2,2),1); 
    d_WL1 = zeros(size(WL1,1),size(WL1,2),1);  d_St=zeros(num_features_p1-1,1,T);
    
    for t=1:T
        %% backpropagate through the output layer
        d_WL3 =  d_WL3 +  inputL3(t,:);
        d_inputL3= WL3' ;
        %% backpropagate through the middle layer
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
