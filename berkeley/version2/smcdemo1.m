% PURPOSE : To approximate a noisy nolinear function with RBFs, where the number
%           of parameters and parameter values are estimated via sequential reversible jump
%           Markov Chain Monte Carlo (MCMC) simulation.
             
% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 21-01-99

clear;
echo off;

% INITIALISATION AND PARAMETERS:
% =============================
N = 500                      % Number of time steps.
t = 1:1:N;                   % Time.
bFunction = 'rjGaussian'     % Type of basis function.
par.doPlot = 0;              % 1 plot. 0 don't plot.
par.dalpha = 1e-3;           % Hyperparameter for alpha. 
par.dmu = 1e-3;              % Hyperparameter for mu.
par.dsigma = 1e-4;           % Hyperparameter for sigma.    
par.kMax = 10;               % Maximum number of basis.
par.P0 = 1;                  % KF covariance diagonal entries for alpha.
par.P02 = .1;

par.probBirth = .025;         % Model transition probabilities.
par.probUpdate = .95;
par.probDeath = .025;

par.alpha0 = 1;              % Variance for the KF mean of alpha.
par.mu0 = 1e-1;              % Variance of mu's Gaussian distribution.
par.mu02 = 1e-3;             % Variance of mu's Gaussian distribution.
par.k0 = 5;                  % Initial variance of k.
par.sigma0 = 1e-2;           % LOG variance of sigma's initial uniform distribution.
par.muMean = 0.5;            % Mean of mu proposal.
par.alphaMean = 3            % Mean of alpha proposal.
S= 500                       % Number of particles.

% GENERATE THE DATA:
% =================
noiseVar = 0.01;
x = 4*rand(N,1)-2;                    % Input data - uniform in [-2,2].
u = randn(N,1);
noise = sqrt(noiseVar)*u;             % Measurement noise
y = zeros(N,1);                       % Output data.
for t=1:N/2,
  y(t)=x(t)+(2+t/150)*exp(-15*((x(t)+.8).^(2)))+2*exp(-15*((x(t)-.8).^(2)))+noise(t);
end;
for t=N/2+1:N,
  y(t) = x(t) + 2*exp(-15*((x(t)).^(2))) + noise(t); 
end;
x=(x+2)/4;                            % Rescaling to [0,1].

figure(1)
clf;
subplot(211)
plot(x(1:N/2),y(1:N/2),'b+');
ylabel('y_{1:500}','fontsize',15);
xlabel('x_{1:500}','fontsize',15);
subplot(212)
plot(x(N/2+1:N),y(N/2+1:N),'b+');
ylabel('y_{501:1000}','fontsize',15);
xlabel('x_{501:1000}','fontsize',15);
fprintf('\n')
fprintf('Press a key to continue')
fprintf('\n')
pause;


% PERFORM SEQUENTIAL REVERSE JUMP MCMC WITH RADIAL BASIS:
% ======================================================
[rec,mov,q,ypred] = smcrbf(x,y,S,N,bFunction,par);

% COMPUTE CENTROID, MAP AND VARIANCE ESTIMATES:
% ============================================

[S,N]=size(ypred);
yp=mean(ypred);

% PLOTS:
% =====

figure(4)
clf
subplot(211)
plot(x,y,'b+',x,yp,'ro');
ylabel('Output data','fontsize',15);
xlabel('Input data','fontsize',15);
subplot(212)
bi=50;
plot(x,y,'b+',x(bi:N,1),yp(1,bi:N)','ro');
ylabel('Output data','fontsize',15);
xlabel('Input data','fontsize',15);
mu1=-90*ones(N/2,S);
mu2=-90*ones(N/2,S);
mu3=-90*ones(N/2,S);
alpha1=-90*ones(N,S);
alpha2=-90*ones(N,S);
alpha3=-90*ones(N/2,S);
alpha4=-90*ones(N/2,S);
alpha5=-90*ones(N/2,S);
for t=1:N/2,
  for s=1:S,
    if rec.k(s,t) >= 2
      mu1(t,s)=rec.mu{t}{s}(1,1);
      mu2(t,s)=rec.mu{t}{s}(2,1);
      alpha1(t,s)=rec.alpha{t}{s}(1,1);
      alpha2(t,s)=rec.alpha{t}{s}(2,1); 
      alpha3(t,s)=rec.alpha{t}{s}(3,1);
      alpha4(t,s)=rec.alpha{t}{s}(4,1); 
    end;
  end;
end;
for t=N/2+1:N,
  for s=1:S,
    if rec.k(s,t) >= 1
      mu3(t-N/2,s)=rec.mu{t}{s}(1,1);
      alpha1(t,s)=rec.alpha{t}{s}(1,1);
      alpha2(t,s)=rec.alpha{t}{s}(2,1); 
      alpha5(t-N/2,s)=rec.alpha{t}{s}(3,1);
    end;
  end;
end;
mu1mean=zeros(N/2,1);
mu2mean=zeros(N/2,1);
mu3mean=zeros(N/2,1);
alpha1mean=zeros(N,1);
alpha2mean=zeros(N,1);
alpha3mean=zeros(N/2,1);
alpha4mean=zeros(N/2,1);
alpha5mean=zeros(N/2,1);
for t=1:N/2,
  i=find(mu1(t,:)>-10); 
  mu1mean(t) = mean(mu1(t,i));
  i=find(mu2(t,:)>0); 
  mu2mean(t) = mean(mu2(t,i));
  i=find(alpha1(t,:)>-10); 
  alpha1mean(t) = mean(alpha1(t,i));
  i=find(alpha2(t,:)>0); 
  alpha2mean(t) = mean(alpha2(t,i));
  i=find(alpha3(t,:)>0); 
  alpha3mean(t) = mean(alpha3(t,i));
  i=find(alpha4(t,:)>0); 
  alpha4mean(t) = mean(alpha4(t,i));
end;

for t=N/2+1:N,
  i=find(mu3(t-N/2,:)>-10); 
  mu3mean(t) = mean(mu3(t-N/2,i));
  i=find(alpha1(t,:)>-10); 
  alpha1mean(t) = mean(alpha1(t,i));
  i=find(alpha2(t,:)>0); 
  alpha2mean(t) = mean(alpha2(t,i));
  i=find(alpha5(t-N/2,:)>0); 
  alpha5mean(t) = mean(alpha5(t-N/2,i));
end;

kmean=mean(rec.k);
sigmean=mean(rec.sigma);

figure(1)
clf
subplot(311)
plot(1:N,y,'b:',1:N,yp,'r','linewidth',2);
ylabel('Prediction','fontsize',15);
subplot(312)
plot(kmean,'linewidth',2)
hold on;
k1=2*ones(N/2,1);
k2=1*ones(N/2,1);
thek=[k1;k2];
plot(1:N,thek,'r','linewidth',1);
ylabel('k','fontsize',15);
subplot(313)
plot(exp(sigmean),'linewidth',2)
hold on
plot(1:N,var(noise)*ones(N,1),'r','linewidth',1);
ylabel('\sigma^{2}','fontsize',15);
axis([0 N-1 -.1 .5]);
xlabel('Time','fontsize',15);
zoom on

figure(2)
clf
subplot(411)
plot([mu1mean mu2mean],'linewidth',2)
hold on
plot(1:N/2,0.3*ones(N/2,1),'r','linewidth',1);
plot(1:N/2,0.7*ones(N/2,1),'r','linewidth',1);
ylabel('\mu_1 and \mu_2','fontsize',15);
hold on
plot(N/2+1:N,mu3mean(N/2+1:N),'linewidth',2)
hold on
plot(N/2+1:N,0.5*ones(N/2,1),'r','linewidth',1);
subplot(412)
plot(alpha1mean,'linewidth',2)
hold on
plot(1:N,-2*ones(N,1),'r','linewidth',1);
ylabel('b','fontsize',15);
subplot(413)
plot(alpha2mean,'linewidth',2)
ylabel('\beta','fontsize',15);
hold on
plot(1:N,4*ones(N,1),'r','linewidth',1);
zoom on
subplot(414)
plot([alpha3mean alpha4mean],'linewidth',2)
hold on
plot(1:N/2,2*ones(N/2,1),'r','linewidth',1);
t=1:1:N/2;
t=t';
hold on
plot(t,2*ones(N/2,1)+t/150,'r','linewidth',1);
ylabel('\alpha_1 and \alpha_2','fontsize',15);
plot(N/2+1:N,alpha5mean(N/2+1:N),'linewidth',2)
hold on;
plot(N/2+1:N,2*ones(N/2,1),'r','linewidth',1);
xlabel('Time','fontsize',15);


figure(3)
clf;
domain = zeros(S,1);
range = zeros(S,1);
support=[0:.005:.2];
%v=[0 1];
%caxis(v);
for t=10:50:N,
  [range,domain]=hist(exp(rec.sigma(:,t)),support);
  waterfall((domain),t,range/sum(range))
  hold on
end;
rotate3d on;
ylabel('t','fontsize',15)
xlabel('\sigma^{2}_t','fontsize',15)
zlabel('P(\sigma^{2}_t|y_{1:t})','fontsize',15)
view(-30,75);
rotate3d on;
a=get(gca);
set(gca,'ygrid','off');
set(gca,'linewidth',1);


figure(5)
clf;
domain = zeros(S,1);
range = zeros(S,1);
support=[0:.05:6];
%v=[0 1];
%caxis(v);
for t=1:50:N,
  [range,domain]=hist(rec.k(:,t),support);
  waterfall((domain),t,range/sum(range))
  hold on
end;
rotate3d on;
ylabel('t','fontsize',15)
xlabel('k_t','fontsize',15)
zlabel('P(k_t|y_{1:t})','fontsize',15)
view(-30,70);
rotate3d on;
a=get(gca);
set(gca,'ygrid','off');
set(gca,'linewidth',1);
set(gca,'gridlinestyle',':');







