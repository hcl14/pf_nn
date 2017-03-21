% PURPOSE : To approximate a noisy nolinear function with RBFs, where the number
%           of parameters and parameter values are estimated via reversible jump
%           Markov Chain Monte Carlo (MCMC) simulated annealing.
             
% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 21-05-99

clear;
echo off;

% INITIALISATION AND PARAMETERS:
% =============================
N = 100                       % Number of time steps.
t = 1:1:N;                    % Time.
chainLength = 500;            % Length of the Markov chain simulation.
burnIn = 100;                 % Burn In period. To allow for convergence.
bFunction = 'rjGaussian'      % Type of basis function.
par.doPlot = 0;               % 1 plot. 0 don't plot.
par.a = 2;                    % Hyperparameter for delta.  
par.b = 10;                   % Hyperparameter for delta.
par.e1 = 0.0001;              % Hyperparameter for nabla.    
par.e2 = 0.0001;              % Hyperparameter for nabla.   
par.v = 0;                    % Hyperparameter for sigma    
par.gamma = 0;                % Hyperparameter for sigma. 
par.kMax = 50;                % Maximum number of basis.
par.arbC = 0.2;               % Constant to choose moves.
par.merge = .1;               % Split-Merge parameter.
par.Lambda = .5;              % Hybrid Metropolis decision parameter.
par.sRW = .001;               % Variance of noise in the random walk.
par.Ta = 1;                   % Cooling parameter (~Initial temperature).
par.Tf = 1e-1;                % Cooling parameter (Final temperature). 
par.criterion = 2;            % 1-AIC  2-MDL
par.walkPer = 0.1             % Percentange of random walk interval. 

% GENERATE THE DATA:
% =================
noiseVar = 0.1;                       % Noise variance.
x = 4*rand(N,1)-2;                    % Input train data - uniform in [-2,2].
u = randn(N,1);
noise = sqrt(noiseVar)*u;             % Measurement noise
varianceN=var(noise);
y = x + 2*exp(-16*(x.^(2))) + 2*exp(-16*((x-.7).^(2)))   + noise;  % Output data.
x=(x+2)/4;                            % Rescaling to [0,1].
ynn = y-noise;
xv = 4*rand(N,1)-2;                    % Input test data - uniform in [-2,2].
uv = randn(N,1);
noisev = sqrt(noiseVar)*uv;   
yv = xv + 2*exp(-16*(xv.^(2))) + 2*exp(-16*((xv-.7).^(2)))   + noisev;  % Output test data.
xv=(xv+2)/4;               
yvnn = yv-noisev;
thevar=var(noise)
thevarv=var(noisev)

figure(1)
subplot(211)
plot(x,y,'b+');
ylabel('Output data','fontsize',15);
xlabel('Input data','fontsize',15);
%axis([0 1 -3 3]);
subplot(212)
plot(noise)
ylabel('Measurement noise','fontsize',15);
xlabel('Time','fontsize',15);

fprintf('\n')
fprintf('Press a key to continue')
fprintf('\n')
pause;

% PERFORM REVERSE JUMP MCMC WITH RADIAL BASIS:
% ===========================================
[k,mu,alpha,yp,ypv,post] = rjsann(x,y,chainLength,N,bFunction,par,xv,yv);

% COMPUTE CENTROID, MAP AND VARIANCE ESTIMATES:
% ============================================
[l,m]=size(mu{1});
[Nv,d]=size(xv);
l=chainLength-burnIn;

% FIND POSTERIOR MAXIMUM:
% ======================
[maxPos,j]=find(post(2:chainLength)==max(post(2:chainLength)))

% CONVERT YP AND YPV TO 2D VECTORS:
% ================================
ypred = zeros(N,l+1);
ypredv = zeros(Nv,l+1);
for i=1:N;
  ypred(i,:) = yp(i,1,burnIn:chainLength);
end;
for i=1:Nv;
  ypredv(i,:) = ypv(i,1,burnIn:chainLength);
end;

% COMPUTE THE TRAIN AND TEST SET ERRORS: (Percentage of unexplained variance)
% =====================================
Error=zeros(chainLength,1);
for i=1:chainLength,
%  Error(i) = inv(N) * trace((y-yp(:,:,i))'*(y-yp(:,:,i)));
  Error(i) = (y-yp(:,:,i))'*(y-yp(:,:,i))*inv((y-mean(y)*ones(size(y)))'*(y-mean(y)*ones(size(y))));
end;
Errorv=zeros(chainLength,1);
for i=1:chainLength,
%  Errorv(i) = inv(N) * trace((yv-ypv(:,:,i))'*(yv-ypv(:,:,i)));
  Errorv(i) = (yv-ypv(:,:,i))'*(yv-ypv(:,:,i))*inv((yv-mean(yv)*ones(size(yv)))'*(yv-mean(yv)*ones(size(yv))));
end;

% COMPUTE ERROR AT THE MAX OF THE POSTERIOR:
% =========================================
errorMin=Error(maxPos)
errorMinv=Errorv(maxPos)
mostProbableModel=k(maxPos)

% PLOT RESULTS:
% ============
figure(1)
clf;
[xv,i]=sort(xv);
yvnn=yvnn(i);
ypredv=ypredv(i);
yv=yv(i);
[x,i]=sort(x);
ynn=ynn(i);
ypred=ypred(i);
y=y(i);
clf;
subplot(211)
plot(x,ynn,'m--',x,y,'b+',x,ypred,'r')
ylabel('Train output','fontsize',15)
xlabel('Train input','fontsize',15)
legend('True function','Test data','Test data prediction');
subplot(212)
plot(xv,yvnn,'m--',xv,yv,'b+',xv,ypredv,'r')
ylabel('Test output','fontsize',15)
xlabel('Test input','fontsize',15)

figure(2)
clf
N=chainLength;
subplot(411)
plot(Error(1:N))
ylabel('Train error','fontsize',15)
%axis([0 chainLength 0 0.01]);
subplot(412)
plot(Errorv(1:N))
ylabel('Test error','fontsize',15)
%axis([0 chainLength 0 0.01]);
subplot(413)
plot(k(1:N))
ylabel('k','fontsize',15)
subplot(414)
plot(post(2:N))
ylabel('log(p(k,\mu_{1:k}|y))','fontsize',15)
xlabel('Iterations','fontsize',15)
zoom;











