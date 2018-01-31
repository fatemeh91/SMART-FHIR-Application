function [value, d_WL1, d_St] = g_evalVNetwork1(S_t, WL1)%#codgen
% INPUT:
%   S_t             input feature vector      (1 x p)
%   WL1             weights for input layer   (num_actions, p+1 )
% OUTPUT:
%   value   value of each of the actions within state S_t: V(S_t)
%   d_WL1           gradients of the network outputs with respect to the network weights

%  Copyright 2014-2015   Shamim Nemati < shamim at seas.harvard.edu >
%  $Revision: 1.1 $  $Date: 02/01/2015  $

T = size(S_t,1); [~,num_features_p1]=size(WL1);
%% input layer
inputL1 = [ones(T,1) S_t];
value = inputL1*WL1';

%% backprop
if nargout>1
    d_WL1 = zeros(size(WL1,1),size(WL1,2));
    d_St = zeros(1,num_features_p1-1,T);
    %% backpropagate through the output layer
    for t=1:T
        %% backpropagate through the input layer
        d_WL1 = d_WL1 + inputL1(t,:);
        d_St(:,:,t) = WL1(:,2:end);
    end
end
end
