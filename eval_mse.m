function [MSE, MSESEQ] = eval_mse(OS, TS)
%
%

if nargin < 2,
   error('Function requires 2 input arguments');
end;

if size(OS) ~= size(TS),
   error('Dimensions of outpur and target sequences do not match');
end;  
MSESEQ = 0.5 .* (OS - TS) .^ 2;
MSE = sqrt( sum(sum(MSESEQ)) ./ prod(size(OS)));
%MSE =sum(sum(MSESEQ)) ./ prod(size(OS));
