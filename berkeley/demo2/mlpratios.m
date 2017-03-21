function q = mlpratios(xu,input,y,s1,s2,R);
% PURPOSE : To evaluate the normalised importance ratios. 
% INPUTS  : - xu = The predicted network weights samples.
%           - input = The input observations.
%           - y = The output observations.
%           - s1 = Number of neurons in the hidden layer.
%           - s2 = Number of neurons in the output layer (=1).
%           - R = Measurement noise variance parameter.
% OUTPUTS : - q = The normalised importance ratios.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 08-09-98

if nargin < 6, error('Not enough input arguments.'); end

[numsamples,time,numweights] = size(xu);
q = zeros(numsamples,1);
m = zeros(numsamples,1); 
for s=1:numsamples,
  m(s,1) = mlp(input,xu(s,1,:),s1,s2);
  q(s,1) = exp(-.5*inv(R)*(y- m(s,1))^(2));
end;
q = q./sum(q(:,1));

figure(1);
subplot(223)       
hist(q,[0:.002:.09]);
ylabel('Histogram','fontsize',15);
xlabel('Importance ratios','fontsize',15);
subplot(224)
plot(q,'m')
axis([0 numsamples 0 .5]);
ylabel('Importance ratios','fontsize',15);
xlabel('Sample space','fontsize',15);
