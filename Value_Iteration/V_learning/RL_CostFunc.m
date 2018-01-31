function [E, g_W] = RL_CostFunc(par_vec, RL, VStp1 , dims)

INPUT_size = numel(RL);  num_layers=dims.num_layers; WL=cell(num_layers,1);
st=1;
for l=1:num_layers
    WL{l} = reshape(par_vec(st:st-1+prod(dims.sizeWL{l})), dims.sizeWL{l}(1), dims.sizeWL{l}(2));
    st=st+prod(dims.sizeWL{l});
end

if nargout >1
    E =zeros(1,INPUT_size) ; g_W = zeros(length(par_vec),INPUT_size);
    parfor n=1:INPUT_size
        % evaluate gradient of the Q-network with respect to the RL loss function
        [E(n), dL_WL] = BackpropVNetwork(RL{n}.state_t, RL{n}.r_t, VStp1{n}, WL);  % temporal difference error and gradient of error with respect to network parameters
        tmp=[]; for l=1:num_layers, tmp=[tmp;dL_WL{l}(:)]; end
        g_W(:,n) =  tmp;
    end
    g_W = nanmean(g_W,2);
else
    E =zeros(1,INPUT_size);
    parfor n=1:INPUT_size
        % evaluate gradient of the Q-network with respect to the RL loss function
        E(n) = BackpropVNetwork(RL{n}.state_t, RL{n}.r_t, VStp1{n}, WL);
    end
end
E = nanmean(E);