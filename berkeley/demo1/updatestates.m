function x = updatestates(xu,q);
% PURPOSE : Performs the resampling step of the SIR algorithm
%           in order(numSamples) steps.
% INPUTS  : - xu = Predicted state samples.
%           - q = Normalised importance ratios.
% OUTPUTS : - x = Resampled state samples.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 08-09-98

if nargin < 2, error('Not enough input arguments.'); end

[N,timeStep]=size(xu);  % N = number of samples;
u = rand(N+1,1);
t = -log(u);
x = 10.*ones(size(xu));
T = cumsum(t);
Q = cumsum(q);
i = 1;
j = 1;
while j <= N,
  if (Q(j,1)*T(N,1)) > T(i,1)
    x(i,1) = xu(j,1); 
    i = i+1;
  else
    j = j+1;
  end;
end;

