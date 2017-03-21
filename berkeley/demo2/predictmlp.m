function xu = predictmlp(x,Q);
% PURPOSE : Performs the prediction step of the hybrid SIR
%           algorithm.
% INPUTS  : - x = The current network weights samples.
%           - Q = Process noise variance parameter.
% OUTPUTS : - xu = Predicted network weights samples.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 08-09-98

if nargin < 2, error('Not enough input arguments.'); end
xu = x + sqrt(Q)*randn(size(x));

