function [x, q, m] = hybridsir(input,y,s1,s2,S,Q,initVar1,initVar2,R,KalmanR,KalmanQ,KalmanP,tsteps);
% PURPOSE : To train an MLP with the hybrid SIR algorithm. 
% INPUTS  : - input = The input observations.
%           - y = The output observation.
%           - s1 = Number of neurons in the hidden layer.
%           - s2 = Number of neurons in the output layer (1).
%           - S = Number of samples describing the network weights.
%           - Q = Process noise variance parameter.
%           - initVar1 = Initial variance of the hidden layer weights.
%           - initVar2 = Initial variance of the output layer weights.
%           - R = Measurement noise variance parameter.
%           - KalmanR = EKF measurement noise hyperparameter.
%           - KalmanQ = EKF process noise hyperparameter.
%           - KalmanP = initial EKF covariances for each trajectory.            
%           - tsteps = Number of time steps (input error checking).
% OUTPUTS : - x = Samples describing the network weights.
%           - q = Normalised importance ratios.
%           - m = Samples describing the network one-step-ahead prediction.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 08-09-98

if nargin < 13, error('Not enough input arguments.'); end
[rows,cols] = size(y);  % rows = Max number of time steps.
                        % S = Number of samples;
if (rows ~= tsteps), error('y must be of size Nx1'), end

[r,inputdim] = size(input);
T = s2*(s1+1) + s1*(inputdim+1);  % Number of states (MLP weights). 
Nstd = 3;                   % No of standard deviations for error bars;
x=zeros(S,rows,T);
xu=zeros(S,rows,T);
q=zeros(S,rows);
m=zeros(S,rows);
H=zeros(T,S);
P=zeros(S,T,T);
for s=1:S,
  P(s,:,:)= sqrt(KalmanP)*eye(T,T);
end;
figure(1)
clf;

% SAMPLE FROM THE PRIOR:
% =====================
x(:,1,:) = sqrt(initVar2)*randn(S,1,T); % Prior for output layer.
x(:,1,s1+2:T) = sqrt(initVar1)*randn(S,1,T-(s1+1)); % Prior for hidden layer.

% UPDATE AND PREDICTION STAGES:
% ============================
for t=1:rows-1,
  
  for s=1:S,
    m(s,t+1) = mlp(input(t+1,:),x(s,t,:),s1,s2);
  end;

  figure(1)  
  subplot(221)
  plot(y)
  ylabel('Output','fontsize',15);
  xlabel('Time','fontsize',15);
  hold on
  plot(t*ones(1,49),[-19:1:29],'r');
  hold off
  figure(1)
  subplot(222)
  hold on
  plot(t,mean(m(1:S,t+1)),'ro',t,y(t+1,1),'go');
  hold on
  errorbar(t,mean(m(1:S,t+1)),Nstd*std(m(1:S,t+1)),Nstd*std(m(1:S,t+1)),'k')
  legend('Posterior mean estimate','True value');
  ylabel('One-step-ahead prediction','fontsize',15)
  xlabel('Time','fontsize',15)

  xu(:,t,:) = predictmlp(x(:,t,:),Q);
  [xu(:,t,:),P] = gradupdate(xu(:,t,:),input(t+1,:),y(t+1,1),s1,s2,P,KalmanR,KalmanQ);  
  q(:,t) = mlpratios(xu(:,t,:),input(t+1,:),y(t+1,1),s1,s2,R);
  [x(:,t+1,:),P] = resamplemlp(xu(:,t,:),q(:,t),P);

end;











