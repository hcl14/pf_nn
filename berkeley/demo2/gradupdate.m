function [x,P] = gradupdate(xu,input,y,s1,s2,P,KalmanR,KalmanQ);
% PURPOSE : Updates each of the sample trajectories using and extended
%           Kalman filter. 
% INPUTS  : - xu = The networks weights samples.
%           - input = The network input.
%           - y = The network output.
%           - P = The weights covariance for each trajectory.
%           - s1 = Number of neurons in the hidden layer.
%           - s2 = Number of neurons in the output layer (1).
%           - KalmanR = EKF measurement noise hyperparameter.
%           - KalmanQ = EKF process noise hyperparameter. 
% OUTPUTS : - x = The updated weights samples.
%           - P = The updated weights covariance for each trajectory.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 08-09-98

if nargin < 8
    error('Not enough input arguments.'); 
end

[N,time,numWeights] = size(xu);
x = 10.*ones(size(xu));

% GRADIENT PROPAGATION
% ====================
increment=zeros(1,1,numWeights);
m = zeros(N,1); 
H = zeros(numWeights,N);
Qekf= KalmanQ*eye(numWeights,numWeights);
Rekf= KalmanR;
Pekf=eye(numWeights,numWeights);
xekf=zeros(1,numWeights);
for s=1:N,
  [m(s,1) H(:,s)] = mlph(input,xu(s,1,:),s1,s2);
  Pekf(:,:) = P(s,:,:);
  K = (Pekf+Qekf) * H(:,s) * ((Rekf + H(:,s)'*(Pekf+Qekf)*H(:,s))^(-1));
  error1 = y-m(s,1); 
  xekf(1,:)=xu(s,1,:); 
  x(s,1,:) = xekf' + K * error1;
  P(s,:,:) = Pekf -  K*H(:,s)'*(Pekf+Qekf) + Qekf;
end;



 





