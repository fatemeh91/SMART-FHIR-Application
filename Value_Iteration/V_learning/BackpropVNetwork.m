function [Loss, dL_WL, dL_St] = BackpropVNetwork(S_t, r_t, VStp1, WL)
% INPUT:
% S_t               current state  (1 x p)
% a_t               current action (1 x num_actions)
% r_t               reward associated with the current (state,action) pair
% VStp1             value of each of the actions within state S_t+1: Q(S_t+1,a_t)
% WL(:,:,la)        weights for layer la
%                   la=1 : (p+1 x num_nodes_L1), la=last : (num_nodes_L1+1 x num_actions)
% OUTPUT:
% Loss              Q-learning loss value
% dL_WL(:,:,la)     gradients of the loss with respect to the network weights at lth layer

%  Copyright 2014-2015   Shamim Nemati < shamim at seas.harvard.edu >
%  $Revision: 1.1 $  $Date: 02/01/2015  $

num_layers=numel(WL);
gamma=0.99;
T = size(S_t,1);  Ttot=T; T=1;
Loss=0;
if nargout >1
    dL_WL=cell(num_layers,1);  dL_St = zeros(size(S_t));
    for la=1:num_layers, dL_WL{la}=zeros(size(WL{la})); end
    for t=1:Ttot
        % Immediate reward + discounted long-term reward of choosing the best action in state S_t+1
        y = r_t(t) + gamma * VStp1(t,:)*double((t~=Ttot)); % this is a scaler
        % The following returns the value of all actions
        % V value of the (S_t,a_t) pair with the new network weights
        [V, g_WL, g_St]= g_evalVNetwork(S_t(t,:), WL);
        E = (y - V); % error
        Loss = Loss + 0.5/T * E.^2;  % lost
        % dE_dW = dE_dV x dV_dW
        for la=1:num_layers
            dL_WL{la} = dL_WL{la} - E * g_WL{la} / T;
        end
        % dE_dSt = dE_dV x dV_dSt
        dL_St(t,:) = - E * g_St';
    end
else
    for t=1:Ttot
        % Immediate reward + discounted long-term reward of choosing the best action in state S_t+1
        y = r_t(t) + gamma * VStp1(t,:)*double((t~=Ttot)); % this is a scaler
        % V value of the (S_t,a_t) pair with the new network weights
        V = evalVNetwork(S_t(t,:), WL, 1);
        E = (y - V); % error
        Loss = Loss + 0.5/T * E.^2;  % lost
    end
end
