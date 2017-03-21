% PURPOSE :  Sequential Tracking of Options Prices with an MLP model.
%       The model is estimated both with a plain extended Kalman filter 
%       (EKF) and an EKF where the process noise covariance is estimated 
%       adaptively.
             
% AUTHORS : Nando de Freitas and Mahesan Niranjan - Thanks for the acknowledgement :-)
% DATE    : 08-09-98

% LOAD THE DATA:
% =============

fprintf('\n')
fprintf('Loading the data')
fprintf('\n')
clear;
load c2925.prn;         load p2925.prn;
load c3025.prn;         load p3025.prn;
load c3125.prn;         load p3125.prn;
load c3225.prn;         load p3225.prn;
load c3325.prn;         load p3325.prn;
X=[2925; 3025; 3125; 3225; 3325];
[d1,i1]=sort(c2925(:,1));  Y1=c2925(i1,:);      Z1=p2925(i1,:);
[d2,i2]=sort(c3025(:,1));  Y2=c3025(i2,:);      Z2=p3025(i2,:);
[d3,i3]=sort(c3125(:,1));  Y3=c3125(i3,:);      Z3=p3125(i3,:);
[d4,i4]=sort(c3225(:,1));  Y4=c3225(i4,:);      Z4=p3225(i4,:);
[d5,i5]=sort(c3325(:,1));  Y5=c3325(i5,:);      Z5=p3325(i5,:);
d=Y1(:,1);   N=length(d);
% d - date to maturity.
S(1,:) = Y1(:,3)';   C(1,:) = Y1(:,2)';  P(1,:) = Z1(:,2)';
S(2,:) = Y2(:,3)';   C(2,:) = Y2(:,2)';  P(2,:) = Z2(:,2)';
S(3,:) = Y3(:,3)';   C(3,:) = Y3(:,2)';  P(3,:) = Z3(:,2)';
S(4,:) = Y4(:,3)';   C(4,:) = Y4(:,2)';  P(4,:) = Z4(:,2)';
S(5,:) = Y5(:,3)';   C(5,:) = Y5(:,2)';  P(5,:) = Z5(:,2)';
% S - Stock price.
% C - Call option price.
% P - Put Option price.
% X - Strike price.
% Normalise with respect to the strike price:
for i=1:5
   Cox(i,:) = C(i,:) / X(i);
   Sox(i,:) = S(i,:) / X(i);
   Pox(i,:) = P(i,:) / X(i);
end
N = 204;
Cpred=zeros(N,5);
Ppred=zeros(N,5);

% PLOT THE LOADED DATA:
% ====================

figure(1)
clf;
plot(Cox');
ylabel('Call option prices','fontsize',15);
xlabel('Time to maturity','fontsize',15);
fprintf('\n')
fprintf('Press a key to continue')  
pause;
fprintf('\n')
fprintf('\n')
fprintf('Training the MLP with EKF')
fprintf('\n')

% SIMULATION:
% ==========

for ii=1:1   % Only one call price. Change 1 to 3, etc. for other prices.
  X = X(ii,1);
  S = Sox(ii,1:N);
  C = Cox(ii,1:N);
  P = Pox(ii,1:N);
  counter=1:1:N;
  tm = (224*ones(size(counter))-counter)/260;
  x = [S' tm']';

  % SIMULATION PARAMETERS:
  % =====================
  s1=6;             % Neurons in the hidden layer.
  s2=1;             % Neurons in the output layer.
  Q= 0.00001;       % Process noise covariance hyperparameter.
  R= 0.00001;       % Measurement noise covariance hyperparameter.
  initP = 10;       % Initial EKF covariance parameter.
  initVar = 1;      % Initial weights variance.
  window=10;        % Size of moving window to estimate Q.

  % PERFORM EKF ESTIMATION:
  % ======================
  tInit=clock;
  [p,theta,thetaR,PR,Innovations] = mlpekf(x,C,s1,s2,R,Q,initP,initVar,N);
  durationekf = etime(clock,tInit);

  % PERFORM EKF WITH EVIDENCE MAXIMISATION:
  % ======================================
  fprintf('\n')
  fprintf('Training the MLP with EKF/adaptive Q')
  fprintf('\n')
  tInit=clock;
  [p2,theta2,thetaR2,PRecord2,Innovations2,qplot] = mlpekfQ(x,C,s1,s2,R,Q,initP,initVar,window,N);
  durationekf2=etime(clock,tInit);

  errorT = norm(C(104:204)-p(104:204));
  errorT2 = norm(C(104:204)-p2(104:204));
  fprintf('\n')

  fprintf('Strike price = %d',X(ii))
  fprintf('\n')
  fprintf('EKF error    = %d',errorT)
  fprintf('\n')
  fprintf('EKF-Q erro r = %d',errorT2)
  fprintf('\n')
  fprintf('EKF duration   = %d seconds.\n',durationekf)
  fprintf('EKF-Q duration = %d seconds.\n',durationekf2)
  fprintf('\n')

  % PLOT FITTING:
  % ============

  figure(1)
  clf;
  subplot(221)
  plot(1:length(p),C,'g',1:length(p),p2,'b',1:length(p),p,'r')  
  legend('True value','EKF-Q estimate','EKF estimate');
  ylabel('One-step-ahead prediction','fontsize',15)
  xlabel('Time to maturity','fontsize',15)
  axis([0 204 0 .15]);
  subplot(222)
  plot(p,C,'r+',p2,C,'b+')
  ylabel('True value','fontsize',15)
  xlabel('Prediction','fontsize',15)
  legend('EKF estimate','EKF-Q estimate');
  hold on
  c=0.02:.01:.18;
  plot(c,c,'g');
  axis([0.02 .18 0.02 .18]);
  hold off
  subplot(223)
  plot(1:length(p),Innovations2)
  ylabel('Innovations variance for EKF-Q','fontsize',15)
  xlabel('Time','fontsize',15)
  subplot(224);
  plot(1:length(x),Innovations)
  ylabel('Innovations variance for EKF','fontsize',15)
  xlabel('Time','fontsize',15)

end %ii




 









