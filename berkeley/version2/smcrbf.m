function [rec,mov,q,ypred] = smcrbf(x,y,S,Ndata,bFunction,par);
% PURPOSE : Computes the parameters and number of parameters of a radial basis function (RBF)
%           network using the sequential Monte Carlo and reversible jump MCMC algorithms. 
%           Please have a look at the paper first.
% INPUTS  : - x : Input data.
%           - y : Target data.
%           - Ndata: Number of time steps in the training data set.
%           - bFunction: Type of basis function.
%           - par: Record of simulation parameters (see defaults).
% OUTPUTS : - rec : Record containing the model parameters for all time steps (only for plotting).
%           - mov : Record containing the model parameters for one time step (efficient).
%           - q : importance ratios.
%           - ypred: One step ahead predictions.

% AUTHORS : Nando de Freitas and Arnaud Doucet - Thanks for the acknowledgement :-)
% DATE    : 21-01-99

% PERFORM INPUT CHECKING AND SET DEFAULT PARAMETERS:
% =================================================
if nargin < 5, error('Not enough input arguments.'); end
if (nargin==5)
  par.doPlot = 1;          % 1 plot. 0 don't plot.
  par.dalpha = 0.1;        % Hyperparameter for alpha. 
  par.dk = 3;              % Hyperparameter for k.
  par.dmu = 0.01;          % Hyperparameter for mu.
  par.dsigma = 0.001;      % Hyperparameter for sigma.    
  par.kMax = 10;           % Maximum number of basis.
  par.merge = .05;         % Split-Merge parameter. 
  par.P0 = 1;              % KF covariance diagonal entries for alpha.
  par.alpha0 = 1;          % Variance for the KF mean of alpha.
  par.mu0 = 1;             % Variance of mu's uniform distribution.
  par.k0 = 10;             % Initial variance of k.
  par.sigma0 = 0.1;          % Variance of sigma's initial uniform distribution.
  par.muMean = 0.5;        % Mean of mu proposal.
  par.alphaMean = 2;       % Mean of alpha proposal.
end;
[N,d] = size(x);           % N = number of train data, d = dimension of x.
[N,c] = size(y);           % c = dimension of y, i.e. number of outputs.
if Ndata ~= N, error('input must me N by d and output N by c.'); end;

% INITIALISATION:
% ==============
ypred = zeros(S,N);        % Output one-step-ahead prediction (1D case).
mov.k = ones(S,1);         % Model order - number of basis.
res.pos = ones(S,1);         % Index to track deleted basis function.
movO.k = ones(S,1);        % k at previous time step.
mov.sigma = ones(S,1);     % Output noise variance. (LOG)
mov.mu = cell(S,1);        % Radial basis centres.
mov.alpha = cell(S,1);     % KF radial basis net coefficients.
mov.P = cell(S,1);         % Kalman filter covariance of alpha.
movO.mu = cell(S,1);       % Past Mu.  
q = zeros(S,N);            % Importance weights.

% RECORD KEEPING FOR PLOTS:
% ========================
rec.k = zeros(S,N);
rec.sigma = zeros(S,N);
rec.mu = cell(N,1);
rec.alpha = cell(N,1);

% SAMPLE INITIAL CONDITIONS FROM THEIR PRIORS:
% ===========================================
for s=1:S,
  mov.k(s) = unidrnd(par.k0,1,1);
  mov.k(s) = 5; % Temporary for debugging.
  mov.k(s) = max(mov.k(s),1);
  mov.k(s) = min(mov.k(s),par.kMax);
  mov.alpha{s} = (mvnrnd(par.alphaMean*ones(1,mov.k(s)+d+1),sqrt(par.alpha0)*eye(mov.k(s)+d+1),1)');
  mov.P{s} = par.P0*eye(mov.k(s)+d+1);
  mov.mu{s}= par.muMean*ones(mov.k(s),d)+sqrt(par.mu0)*randn(mov.k(s),d);
  mov.sigma(s)= log(par.sigma0)*rand(1,1);
end;

% Initialise other records:
movO.mu = mov.mu;
res.k = mov.k;
resO.k = movO.k;
res.alpha = mov.alpha;
res.P = mov.P;
res.mu = mov.mu;
res.sigma = mov.sigma;
resO.mu = mov.mu;
pred.k = mov.k;
predO.k = movO.k;
pred.alpha = mov.alpha;
pred.P = mov.P;
pred.mu = mov.mu;
pred.sigma = mov.sigma;
pred.pos = res.pos;

% FILL THE REGRESSION MATRIX:
% ==========================
for s=1:S,
  M=zeros(1,mov.k(s)+d+1);
  M(:,1) = ones(1,1);
  M(:,2:d+1) = x(1,:);
  for j=d+2:mov.k(s)+d+1,
    M(:,j) = feval(bFunction,mov.mu{s}(j-d-1,:),x(1,:));
  end;
  ypred(s,1) = M*mov.alpha{s};
end;

% INITIALISE COUNTERS:
% ===================
aUpdate=0; rUpdate=0; aBirth=0; rBirth=0;
aDeath=0; rDeath=0; aMerge=0; rMerge=0;
birth=par.probBirth; death=par.probDeath;
if par.doPlot
  figure(1)
  clf;
  figure(2)
  clf;
  mu1=zeros(S,1);
  mu2=zeros(S,1);
  alpha1=zeros(S,1);
  alpha2=zeros(S,1);
end;

% UPDATE AND PREDICTION STAGES:
% ============================
for t=1:N-1,

  fprintf('Step : t = %i / %i  \r',t,N);
  fprintf('\n')
   
  % ONE-STEP-AHEAD OUTPUT PREDICTIONS:
  % =================================
  for s=1:S,
    M=zeros(1,mov.k(s)+d+1);
    M(:,1) = 1;
    M(:,2:d+1) = x(t+1,:);
    if mov.k(s)>0
      for j=d+2:mov.k(s)+d+1,
        M(:,j) = feval(bFunction,mov.mu{s}(j-d-1,:),x(t+1,:));
      end;
    end;
    ypred(s,t+1) = M*mov.alpha{s};
  end;

  % RECORD KEEPING FOR PLOTS:
  % ========================
  rec.k(:,t) = mov.k;
  rec.sigma(:,t) = mov.sigma;
  rec.mu{t} = mov.mu;
  rec.alpha{t} = mov.alpha;

  % PLOT SIMULATION RESULTS AT TIME t:
  % =================================
  if par.doPlot
    figure(1)
    subplot(511)
    hold on;
    plot(t,mean(ypred(:,t+1)),'r+',t,y(t+1,1),'bo')
    ylabel('O-S-A','fontsize',15);   
    subplot(512);
    hist(mov.k,0:1:8)
    ylabel('p(k)','fontsize',15);
    subplot(513);
    hold on;
    plot(t,mov.k,'*');
    ylabel('k','fontsize',15);
    subplot(514);
    hold on;
    bar([1 2 3 4 5 6],[aUpdate rUpdate aBirth rBirth aDeath rDeath]);
    ylabel('A-R','fontsize',15);
    xlabel('aU-1 rU-2 aB-3 rB-4 aD-5 rD-6','fontsize',15)
    subplot(515)
    hist(exp(mov.sigma))
    ylabel('p(\sigma^{2})','fontsize',15);
    figure(2)
    for s=1:S,
       if mov.k(s)>0 
         mu1(s) = mov.mu{s}(1,1);
         alpha1(s) = mov.alpha{s}(1,1);
         alpha2(s) = mov.alpha{s}(2,1);
         if mov.k(s) >1
           mu2(s) = mov.mu{s}(2,1);
         end;
       else
         mu1(s) = 0;
         mu2(s) = 0;
         alpha1(s) = 0;
         alpha2(s) = 0;
       end;
    end;
    subplot(411)
    hist(mu1)
    ylabel('p(\mu_1)','fontsize',15);   
    subplot(412)
    hist(mu2)
    ylabel('p(\mu_2)','fontsize',15);   
    subplot(413)
    hist(alpha1)
    ylabel('p(b)','fontsize',15);   
    subplot(414)
    hist(alpha2)
    ylabel('p(\beta)','fontsize',15);   
  end;

  % PREDICTION STEP:
  % =============== 
%sigma = exp(mov.sigma(s))
%mu = mov.mu{s}
%k = mov.k(s)
%alpha = mov.alpha{s}

  for s=1:S,
    u = rand(1,1);
    if u < par.probBirth;
      modelProb = 1;
    elseif u< par.probBirth+par.probUpdate;
      modelProb = 0;
    else
      modelProb = -1;
    end;    
    pred.k(s) = mov.k(s) + modelProb; 
    pred.k(s) = max(pred.k(s),0);
    pred.k(s) = min(pred.k(s),par.kMax);
    pred.mu{s} = mov.mu{s} + sqrt(par.dmu)*randn(size(mov.mu{s}));
    pred.sigma(s) = mov.sigma(s) + sqrt(par.dsigma)*randn(1,1);
    if ((pred.k(s) > mov.k(s)) & (pred.k(s) <= par.kMax))
      % Add a new basis function:
      % ========================
      pred.mu{s} = [pred.mu{s}; x(t+1,:) + sqrt(par.mu02)*randn(1,d)];
      tmpDiag= par.P02*eye(mov.k(s)+d+1+1); 
      tmpDiag(1:mov.k(s)+d+1,1:mov.k(s)+d+1) = mov.P{s};
      pred.P{s} = tmpDiag;       
      pred.alpha{s} = [mov.alpha{s}; par.alphaMean + sqrt(par.alpha0)*randn(1,1)];
    elseif ((pred.k(s) < mov.k(s)) & (pred.k(s) >= 0)) 
      % Delete one basis function at random:
      % ===================================
      pred.pos(s)= unidrnd(mov.k(s));
      if pred.pos(s)==1
        pred.mu{s} = pred.mu{s}(2:mov.k(s),:);
        alpha1 = mov.alpha{s}(1:pred.pos(s)+d+1-1,1);
        alpha2 = mov.alpha{s}(pred.pos(s)+d+2:mov.k(s)+d+1,1);
        pred.alpha{s} = [alpha1; alpha2];
        P1 = mov.P{s}(1:pred.pos(s)+d+1-1,1:pred.pos(s)+d+1-1);
        P2 = mov.P{s}(1:pred.pos(s)+d+1-1,pred.pos(s)+d+1+1:mov.k(s)+d+1);
        P3 = mov.P{s}(pred.pos(s)+d+1+1:mov.k(s)+d+1,1:pred.pos(s)+d+1-1);
        P4 = mov.P{s}(pred.pos(s)+d+1+1:mov.k(s)+d+1,pred.pos(s)+d+1+1:mov.k(s)+d+1);
        pred.P{s} = [P1 P2; P3 P4];
      elseif pred.pos(s)==mov.k(s)
        pred.mu{s} = pred.mu{s}(1:mov.k(s)-1,:);
        pred.alpha{s} = mov.alpha{s}(1:mov.k(s)+d,1);
        pred.P{s} = mov.P{s}(1:mov.k(s)+d,1:mov.k(s)+d);  
      else
        pred.mu{s} = [pred.mu{s}(1:pred.pos(s)-1,:); pred.mu{s}(pred.pos(s)+1:mov.k(s),:)];
        alpha1 = mov.alpha{s}(1:pred.pos(s)+d+1-1,1);
        alpha2 = mov.alpha{s}(pred.pos(s)+d+2:mov.k(s)+d+1,1);
        pred.alpha{s} = [alpha1; alpha2];
        P1 = mov.P{s}(1:pred.pos(s)+d+1-1,1:pred.pos(s)+d+1-1);
        P2 = mov.P{s}(1:pred.pos(s)+d+1-1,pred.pos(s)+d+1+1:mov.k(s)+d+1);
        P3 = mov.P{s}(pred.pos(s)+d+1+1:mov.k(s)+d+1,1:pred.pos(s)+d+1-1);
        P4 = mov.P{s}(pred.pos(s)+d+1+1:mov.k(s)+d+1,pred.pos(s)+d+1+1:mov.k(s)+d+1);
        pred.P{s} = [P1 P2; P3 P4];
      end;
    else
      pred.alpha{s} = mov.alpha{s};
      pred.P{s} = mov.P{s};
    end; % of if.
  end; % of s for loop.

%sigma2 = exp(pred.sigma(s))
%mu2 = pred.mu{s}
%k2 = pred.k(s)
%alpha2 = pred.alpha{s}
    

  % EVALUATE IMPORTANCE WEIGHTS:
  % ===========================
  m = zeros(S,1);
  for s=1:S,
    M=zeros(1,pred.k(s)+d+1);
    M(:,1) = 1;
    M(:,2:d+1) = x(t+1,:);
    if pred.k(s)>0    
      for j=d+2:pred.k(s)+d+1,
        M(:,j) = feval(bFunction,pred.mu{s}(j-d-1,:),x(t+1,:));
      end;
    end;
    m(s,1) = M*pred.alpha{s};
    Q = par.dalpha.*eye(size(pred.P{s}));
    evCov = exp(pred.sigma(s)) + M*(pred.P{s}+Q)*M';
    q(s,t+1) = exp(-0.5*(y(t+1,1)-m(s,1))*inv(evCov)*(y(t+1,1)- ...
						      m(s,1))').*inv(sqrt(evCov));
  end;  
  q(:,t+1) = q(:,t+1)./sum(q(:,t+1));
  

  % RESIDUAL RESAMPLING:
  % ====================
  res.k = pred.k;
  res.alpha = pred.alpha;
  res.P = pred.P;
  res.mu = pred.mu;
  res.sigma = pred.sigma;
  resO.mu = mov.mu;
  resO.k = mov.k;
  
  N_babies= zeros(1,S);
  % first integer part
  q_res = S.*q(:,t+1)';
  N_babies = fix(q_res);
  % residual number of particles to sample
  N_res=S-sum(N_babies);
  if (N_res~=0)
    q_res=(q_res-N_babies)/N_res;
    cumDist= cumsum(q_res);   
    % generate N_res ordered random variables uniformly distributed in [0,1]
    u = fliplr(cumprod(rand(1,N_res).^(1./(N_res:-1:1))));
    j=1;
    for i=1:N_res
      while (u(1,i)>cumDist(1,j))
        j=j+1;
      end
      N_babies(1,j)=N_babies(1,j)+1;
    end;
  end;

  % COPY RESAMPLED TRAJECTORIES:  
  % ============================
  index=1;
  for i=1:S
    if (N_babies(1,i)>0)
      for j=index:index+N_babies(1,i)-1
        res.pos(j) = pred.pos(i);
        res.mu{j} = pred.mu{i};
        res.alpha{j} = pred.alpha{i};
        res.k(j) = pred.k(i);
	resO.k(j) = mov.k(i);
        res.sigma(j) = pred.sigma(i);
        res.P{j} = pred.P{i};
        resO.mu{j} = mov.mu{i}; % Resample previous Mu.
      end;
    end;   
    index= index+N_babies(1,i);   
  end;


  % MOVE STEP:
  % =========
  % Move trajectories:
  mov.k = res.k;
  mov.alpha = res.alpha;  
  mov.P = res.P;
  mov.mu = res.mu;
  mov.sigma = res.sigma;
  movO.mu = resO.mu;
  movO.k = resO.k;

  for s=1:S,
    % COMPUTE THE CENTRES AND DIMENSION WITH REVERSIBLE JUMP MCMC MOVES:
    % =================================================================
    decision=rand(1);
    if ((decision <= birth) & (mov.k(s)<par.kMax) & (mov.k(s) == movO.k(s))),
      [mov,aBirth,rBirth] = smcrBirth1(aBirth,rBirth,mov,movO,x,y,par,bFunction,s,t,birth,death);
    elseif ((decision <= birth+death) & (mov.k(s)<par.kMax) & (mov.k(s) == movO.k(s)-1)),
      [mov,aBirth,rBirth] = smcrBirth2(aBirth,rBirth,res,mov,movO,x,y,par,bFunction,s,t,birth,death);       
    elseif ((decision <= 2*birth+death) & (mov.k(s)>0) & (mov.k(s) == movO.k(s)) ),
      [mov,aDeath,rDeath] = smcrDeath2(aDeath,rDeath,mov,movO,x,y,par,bFunction,s,t,birth,death);
    elseif ((decision <= 2*birth+2*death) & (mov.k(s)>0) & (mov.k(s) == movO.k(s)+1) ),
      [mov,aDeath,rDeath] = smcrDeath1(aDeath,rDeath,mov,movO,x,y,par,bFunction,s,t,birth,death);     
    elseif (mov.k(s)>0),
      [mov,aUpdate,rUpdate] = smcrUpdate(aUpdate,rUpdate,res,mov,movO,x,y,par,bFunction,s,t);
    end;

    % UPDATE BANK OF KALMAN FILTERS (RAO BLACKWELL STEP):
    % ==================================================
    M=zeros(1,mov.k(s)+d+1);
    M(:,1) = 1;
    M(:,2:d+1) = x(t+1,:);
    if mov.k(s)>0
      for j=d+2:mov.k(s)+d+1,
        M(:,j) = feval(bFunction,mov.mu{s}(j-d-1,:),x(t+1,:));
      end;
    end;
    error = y(t+1,1) - M*mov.alpha{s};
    Q = par.dalpha.*eye(size(mov.P{s}));
    evCov = exp(mov.sigma(s)) + M*(mov.P{s}+Q)*M';
    K = (mov.P{s} + Q) * M'*inv(evCov);           % Kalman filter gain.
    mov.alpha{s} = mov.alpha{s} + K*error;
    mov.P{s} = mov.P{s} - K*M*(mov.P{s}+Q) +Q;
  end; % End of s loop.
end;   % End of t loop.





























