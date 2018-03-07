function [f,pGrad,H] = pseudoGradL1(funObj,w,lambda)
%% Psuedo-gradient calculation

%[f,g,H] = funObj(w);
[f,g] = funObj(w); H=[];
f = f + sum(lambda.*abs(w));

pGrad = zeros(size(g));
pGrad(g < -lambda) = g(g < -lambda) + lambda(g < -lambda);
pGrad(g > lambda) = g(g > lambda) - lambda(g > lambda);
nonZero = w~=0 | lambda==0;
pGrad(nonZero) = g(nonZero) + lambda(nonZero).*sign(w(nonZero));

end