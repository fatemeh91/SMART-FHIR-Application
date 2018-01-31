function y = g_actF(X)%#codgen
y = actF(X) .* (1-actF(X)); 
end
