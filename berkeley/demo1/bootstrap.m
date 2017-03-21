function [x,q] = bootstrap(actualx,y,R,Q,initVar,numSamples);
% PURPOSE : This m file performs the bootstrap algorithm (a.k.a. SIR,
%           particle filter, etc.) for the model specified in the
%           file sirdemo1.m. 
% INPUTS  : - actualx = The true hidden state. 
%           - y = The observation.
%           - R = The measurement noise variance parameter.
%           - Q = The process noise variance parameter.
%           - initVar = The initial variance of the state estimate.
%           - numSamples = The number of samples.
% OUTPUTS : - x = The estimated state samples.
%           - q = The normalised importance ratios.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 08-09-98

if nargin < 6, error('Not enough input arguments.'); end

[rows,cols] = size(y);      % rows = Max number of time steps.
S = numSamples;             % Number of samples;
Nstd = 2;                   % No of standard deviations for error bars;
x=zeros(S,rows);
xu=zeros(S,rows);
q=zeros(S,rows);

% SAMPLE FROM THE PRIOR:
% =====================
x(:,1) = sqrt(initVar)*randn(S,1);
mean(x(:,1));
cov(x(:,1));

figure(1)
clf;
subplot(221)
plot(actualx)
ylabel('State x','fontsize',15);
xlabel('Time','fontsize',15);

% UPDATE AND PREDICTION STAGES:
% ============================
for t=1:rows-1,
   
  figure(1)  
  subplot(222)
  plot(y)
  ylabel('Output y','fontsize',15);
  xlabel('Time','fontsize',15);
  hold on
  plot(t*ones(1,49),[-19:1:29],'r');
  hold off
  subplot(223)
  hold on
  plot(t,mean(x(1:S,t,1)),'ro',t,actualx(t,1),'go');
  hold on
  errorbar(t,mean(x(1:S,t,1)),Nstd*std(x(1:S,t,1)),Nstd*std(x(1:S,t,1)),'k')
  legend('Posterior mean estimate','True value');
  ylabel('Sequential state estimate','fontsize',15)
  xlabel('Time','fontsize',15)

  xu(:,t) = predictstates(x(:,t),t,Q);
  q(:,t+1) = importanceweights(xu(:,t),y(t+1,1),R);
  x(:,t+1) = updatestates(xu(:,t),q(:,t+1));
end;










