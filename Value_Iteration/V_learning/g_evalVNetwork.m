function [V, g_WL, g_St] = g_evalVNetwork(S_t, WL)

num_layers=numel(WL); V=[];g_WL=cell(num_layers,1);
if num_layers==1,     [V, g_WL{1}, g_St]= g_evalVNetwork1_mex(S_t, WL{1});
elseif num_layers==2,     [V, g_WL{1}, g_WL{2}, g_St]= g_evalVNetwork2_mex(S_t, WL{1}, WL{2});
elseif num_layers==3, [V, g_WL{1}, g_WL{2}, g_WL{3}, g_St]= g_evalVNetwork3_mex(S_t, WL{1}, WL{2}, WL{3});
elseif num_layers==4, [V, g_WL{1}, g_WL{2}, g_WL{3}, g_WL{4}, g_St]= g_evalVNetwork4_mex(S_t, WL{1}, WL{2}, WL{3}, WL{4});
else disp('Warning: BackpropVNetwork encountered unsupported NN structure ...!')
end