function q = importanceweights(xu,y,R);
% PURPOSE : Computes the normalised importance ratios for the 
%           model described in the file sirdemo1.m.
% INPUTS  : - xu = The predicted state samples.
%           - y = The output measurements.
%           - R = The measurement noise covariance.
% OUTPUTS : - q = The normalised importance ratios.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 08-09-98


if nargin < 3, error('Not enough input arguments.'); end

[rows,cols] = size(xu);
q = zeros(size(xu));
m = (xu.^(2))./20;
for s=1:rows,
  q(s,1) = exp(-.5*R^(-1)*(y- m(s,1))^(2))./sum(exp(-.5*R^(-1)*(y.*ones(size(xu))-m).^(2)));
end;

subplot(224); 
plot(xu,q,'+')      
ylabel('Likelihood function','fontsize',15);
xlabel('Hidden state support','fontsize',15)
axis([-30 30 0 0.03]);

