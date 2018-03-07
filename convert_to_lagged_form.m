function yy = convert_to_lagged_form(y, k)
% Create an observation vector yy(:,t) containing the last k values of y, newest first
% e.g., k=2, y = (a1 a2 a3 a4)     yy  = a2 a3 a4
%                (b1 b2 b3 b4)           b2 b3 b4
%                                        a1 a2 a3
%                                        b1 b2 b3

[s T] = size(y);
bs = s*ones(1,k);
yy = zeros(k*s, T-k+1);
for i=1:k
  yy(block(i,bs), :) = y(:, k-i+1:end-i+1);
end

%yy=[zeros(k*s,k-1) yy];





function sub = block(blocks, block_sizes)

% BLOCK Return a vector of subscripts corresponding to the specified blocks.

% sub = block(blocks, block_sizes)

%

% e.g., block([2 5], [2 1 2 1 2]) = [3 7 8].



blocks = blocks(:)';

block_sizes = block_sizes(:)';

skip = [0 cumsum(block_sizes)];

start = skip(blocks)+1;

fin = start + block_sizes(blocks) - 1;

sub = [];

for j=1:length(blocks)

sub = [sub start(j):fin(j)];

end

