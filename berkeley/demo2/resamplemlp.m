function [x,P] = resamplemlp(xu,q,P);
% PURPOSE : Performs the resampling stage of the hybrid SIR
%           in order(number of samples) steps.
% INPUTS  : - xu = The networks weights samples.
%           - q = Normalised importance ratios.
%           - P = The weights covariance for each trajectory.
% OUTPUTS : - x = Resampled networks weights samples.
%           - P = Resampledweights covariance for each trajectory.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 08-09-98

if nargin < 2, error('Not enough input arguments.'); end
[N,time,numWeights] = size(xu);

u = rand(N+1,1);
t = -log(u);
x = 10.*ones(size(xu));
T = cumsum(t);
Q = cumsum(q);

% RESAMPLING:
% ==========
i = 1;
j = 1;
Ptmp=P;
while j <= N,
  if (Q(j)*T(N)) > T(i)
    x(i,:,:) = xu(j,:,:);
    P(i,:,:) = Ptmp(j,:,:); 
    i = i+1;
  else
    j = j+1;
  end;
end;
x(N,:,:)=x(N-1,:,:); % Bug: See new version with residual sampling.

 



