function [k,mu,alpha,ypred,ypredv,post] = rjsann(x,y,chainLength,Ndata,bFunction,par,xv,yv);
% PURPOSE : Computes the parameters and number of parameters of a radial basis function (RBF)
%           network using the reversible jump MCMC simulated annealing algorithm. Please have a 
%           look at the paper first.
% INPUTS  : - x : Input data.
%           - y : Target data.
%           - chainLength: Number of iterations of the Markov chain.
%           - Ndata: Number of time steps in the training data set.
%           - bFunction: Type of basis function.
%           - par: Record of simulation parameters (see defaults).
%           - {xv,yv}: Validation data (optional).

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 21-01-99

% CHECK INPUTS AND SET DEFAULTS:
% =============================
if nargin < 5, error('Not enough input arguments.'); end
if ((nargin==5) | (nargin==7)),
  if nargin == 5,
    Validation = 0;
  else
    Validation = 1;
  end;
  hyper.a = 2;                      % Hyperparameter for delta.  
  hyper.b = 10;                     % Hyperparameter for delta.
  hyper.e1 = 0.0001;                % Hyperparameter for nabla.    
  hyper.e2 = 0.0001;                % Hyperparameter for nabla.   
  hyper.v = 0;                      % Hyperparameter for sigma    
  hyper.gamma = 0;                  % Hyperparameter for sigma. 
  kMax = 50;                        % Maximum number of basis.
  arbC = 0.25;                       % Constant for birth and death moves.
  doPlot = 1;                       % To plot or not to plot? Thats ...
  sigStar = .1;                     % Merge-split parameter.
  sWalk = .01;
  Lambda = .5;
  walkPer = 0.1;
  Ta = 1;                           % Cooling parameter (~Initial temperature).
  Tf = 0.1;                         % Cooling parameter (Final temperature). 
elseif ((nargin==6) | (nargin==8)),
  if nargin == 6,
    Validation = 0;
  else
    Validation = 1;
  end;
  Validation == 0
  hyper.a = par.a;                 
  hyper.b = par.b;                
  hyper.e1 = par.e1;           
  hyper.e2 = par.e2;           
  hyper.v = par.v;                 
  hyper.gamma = par.gamma;             
  kMax = par.kMax;                   
  arbC = par.arbC;
  doPlot = par.doPlot;    
  sigStar = par.merge;
  sWalk = par.sRW;
  Lambda = par.Lambda;
  walkPer = par.walkPer;
  Ta = par.Ta;
  Tf = par.Tf;
else
 error('Wrong Number of input arguments.');
end;
if Validation,
  [Nv,dv] = size(xv);   % Nv = number of test data, dv = dimension of xv.
end;
[N,d] = size(x)      % N = number of data, d = dimension of x.
[N,c] = size(y)      % c = dimension of y, i.e. number of outputs.
if Ndata ~= N, error('input must me N by d and output N by c.'); end

% HYPER-PARAMETERS AND INITIALISATION:
% ===================================
Tb = (Tf-Ta)/chainLength;         % Cooling parameter (Slope).
Temp =  ones(chainLength,1);      % SA temperature.
post = ones(chainLength,1);       % p(centres,k|y).
if Validation,
  ypredv = zeros(Nv,c,chainLength);  % Output fit (test set).
end;
ypred = zeros(N,c,chainLength);   % Output fit (train set).
k = ones(chainLength,1);          % Model order - number of basis.
mu = cell(chainLength,1);         % Radial basis centres.
alpha = cell(chainLength,c);      % Radial basis coefficients.
if par.criterion == 1,             % Model selection criterion:
  criterion= c+1;                 % AIC = c+1
elseif par.criterion == 2,
  criterion= ((c+1)/2) * log(N);  % MDL = ((c+1)/2) * log(N)
end;                            
                                
% DEFINE WALK INTERVAL FOR MU:
% ===========================
walk = walkPer*(max(x)-min(x));
walkInt=zeros(d,1);
for i=1:d,
  walkInt(i,1) = (max(x(:,i))-min(x(:,i))) + 2*walk(i);
end;

% INITIAL CONDITION: (for comparison purposes in exp 1. Delete it otherwise.)
% =================
k(1) = 20;

% DRAW THE INITIAL RADIAL CENTRES WITHOUT REPLACEMENT:
% ===================================================
mu{1}=zeros(k(1),d);
for i=1:d,
  mu{1}(:,i)= (min(x(:,i))-walk(i))*ones(k(1),1) + ((max(x(:,i))+walk(i))-(min(x(:,i))-walk(i)))*rand(k(1),1);
end

% FILL THE REGRESSION MATRIX:
% ==========================
M=zeros(N,k(1)+d+1);
M(:,1) = ones(N,1);
M(:,2:d+1) = x;
for j=d+2:k(1)+d+1,
  M(:,j) = feval(bFunction,mu{1}(j-d-1,:),x);
end;

% COMPUTE THE PREDICTION AT t=0:
% =============================
H=zeros(k(1)+1+d,k(1)+1+d,c);
F=zeros(k(1)+1+d,c);
P=zeros(N,N,c);
for i=1:c,
  H(:,:,i) = inv(M'*M);
  F(:,i) = H(:,:,i)*M'*y(:,i);
  P(:,:,i) = eye(N) - M*H(:,:,i)*M';
  alpha{1,i} = F(:,i); 
end;
for i=1:c,
  ypred(:,i,1) = M*alpha{1,i};
end;
if Validation,
  Mv=zeros(Nv,k(1)+d+1);
  Mv(:,1) = ones(Nv,1);
  Mv(:,2:d+1) = xv;
  for j=d+2:k(1)+d+1,
    Mv(:,j) = feval(bFunction,mu{1}(j-d-1,:),xv);
  end;
  for i=1:c,
    ypredv(:,i,1) = Mv*alpha{1,i};
  end;
end;

% INITIALISE COUNTERS:
% ===================
Nbirth=0;
Ndeath=0;
Nupdate=0;
aUpdate=0;
rUpdate=0;
aBirth=0;
rBirth=0;
aDeath=0;
rDeath=0;
aMerge=0;
rMerge=0;
aSplit=0;
rSplit=0;
aRW=0;
rRW=0;
match=0;
if doPlot,
  figure(3)
  clf;
end;

% ITERATE THE MARKOV CHAIN:
% ========================
for t=1:chainLength-1,
  t=t
%  thek=k(t)
%  thepost=post(t)

  % UPDATE DENOMINATOR FOR NON-HOMOGENEOUS STEP:
  % ============================================
  invH=zeros(k(t)+1+d,k(t)+1+d,c);
  P=zeros(N,N,c);
  for i=1:c,
    invH(:,:,i) = M'*M;
    P(:,:,i) = eye(N) - M*inv(invH(:,:,i))*M';
  end;
  oldM=M;
  small = 0;
  DenRatio =  exp(-criterion*k(t)) * (y(:,1)'*P(:,:,1)*y(:,1)+small)^(N/2);      
  for i=2:c,
    DenRatio = DenRatio * (y(:,i)'*P(:,:,i)*y(:,i)+small)^(N/2); 
  end;

  % COMPUTE THE CENTRES AND DIMENSION WITH METROPOLIS, BIRTH AND DEATH MOVES:
  % ========================================================================
  decision=rand(1);
  birth=arbC;
  death=arbC;
  if ((decision <= birth) & (k(t)<kMax)),
    [k,mu,M,match,aBirth,rBirth] = sarBirth(match,aBirth,rBirth,k,mu,M,x,y,t,criterion,bFunction,walkInt,walk);
  elseif ((decision <= birth+death) & (k(t)>0)),
    [k,mu,M,aDeath,rDeath] = sarDeath(aDeath,rDeath,k,mu,M,x,y,t,criterion,walkInt);
  elseif ((decision <= 2*birth+death) & (k(t)<kMax) & (k(t)>1)),
    [k,mu,M,aSplit,rSplit] = sarSplit(aSplit,rSplit,k,mu,M,x,y,t,bFunction,criterion,sigStar,walk);
  elseif ((decision <= 2*birth+2*death) & (k(t)>1)),
    [k,mu,M,aMerge,rMerge] = sarMerge(aMerge,rMerge,k,mu,M,x,y,t,bFunction,criterion,sigStar);
  else
    uLambda = rand(1);
    if ((uLambda>Lambda) & (k(t)>0))
       [k,mu,M,match,aRW,rRW] = sarRW(match,aRW,rRW,k,mu,M,x,y,t,criterion,bFunction,sWalk,walk);
    else
       [k,mu,M,match,aUpdate,rUpdate] = sarUpdate(match,aUpdate,rUpdate,k,mu,M,x,y,t,criterion,bFunction,walkInt,walk);
    end;
  end;

  % UPDATE NUMERATOR FOR NON-HOMOGENEOUS STEP:
  % =========================================
  invH=zeros(k(t+1)+1+d,k(t+1)+1+d,c);
  P=zeros(N,N,c);
  for i=1:c,
    invH(:,:,i) = M'*M;
    P(:,:,i) = eye(N) - M*inv(invH(:,:,i))*M';
  end;
  small = 0;
  NumRatio =  exp(-criterion*k(t+1)) * (y(:,1)'*P(:,:,1)*y(:,1)+small)^(N/2);      
  for i=2:c,
    NumRatio = NumRatio * (y(:,i)'*P(:,:,i)*y(:,i)+small)^(N/2); 
  end;

  % ANNEALING BIT:
  % =============
  Temp(t) = Ta+Tb*t;          % Cooling schedule. 
  U=rand(1);
  ratio = NumRatio/DenRatio;
  ratio = ratio^(inv(Temp(t))-1);  % Anneal ratio.
  acceptance = min(1,ratio);
  if (U<acceptance),
    mu{t+1} = mu{t+1};
    k(t+1) = k(t+1);
    M=M;
  else
    mu{t+1} = mu{t};
    k(t+1) = k(t);
    M=oldM;
  end;
 
  % UPDATE OTHER PARAMETERS WITH GIBBS:
  % ==================================
  H=zeros(k(t+1)+1+d,k(t+1)+1+d,c);
  F=zeros(k(t+1)+1+d,c);
  P=zeros(N,N,c);
  for i=1:c,
    H(:,:,i) = inv(M'*M);
    F(:,i) = H(:,:,i)*M'*y(:,i);
    P(:,:,i) = eye(N) - M*H(:,:,i)*M';
    alpha{t+1,i} = F(:,i);
  end;

  % COMPUTE THE POSTERIOR FOR MONITORING:
  % ==================================== 
  small = 0; % To avoid numerical problems.
  posterior  =exp(-criterion*k(t+1)) * (y(:,1)'*P(:,:,1)*y(:,1)+small)^(-N/2);
  for i=2:c,
    newpost =  (y(:,i)'*P(:,:,i)*y(:,i)+small)^(-N/2);
    posterior  = posterior * newpost;
  end;
  post(t+1) = log(posterior);

  % PLOT FOR FUN AND MONITORING:
  % ============================ 
  for i=1:c,
    ypred(:,i,t+1) = M*alpha{t+1,i};
  end;
  msError = inv(N) * trace((y-ypred(:,:,t+1))'*(y-ypred(:,:,t+1)));
%  arv = (y-ypred(:,:,t+1))'*(y-ypred(:,:,t+1))*inv((y-mean(y)*ones(size(y)))'*(y-mean(y)*ones(size(y))));
  if Validation,
    % FILL THE REGRESSION MATRIX:
    % ==========================
    Mv=zeros(Nv,k(t+1)+d+1);
    Mv(:,1) = ones(Nv,1);
    Mv(:,2:d+1) = xv;
    for j=d+2:k(t+1)+d+1,
      Mv(:,j) = feval(bFunction,mu{t+1}(j-d-1,:),xv);
    end;
    for i=1:c,
      ypredv(:,i,t+1) = Mv*alpha{t+1,i};
    end;
    msErrorv = inv(Nv) * trace((yv-ypredv(:,:,t+1))'*(yv-ypredv(:,:,t+1)));
  end;

  if doPlot,  
  figure(1)  
  clf
   if (c==2),
      plot(x(:,1),y(:,1),'b+',x(:,2),y(:,2),'r+',x(:,1),ypred(:,1,t+1),'bo',x(:,2),ypred(:,2,t+1),'ro');
   elseif c==1,
    plot(x,y,'b+',x,ypred(:,:,t+1),'ro');
  end;
  ylabel('Output','fontsize',15)
  xlabel('Input','fontsize',15)
  figure(3)
  subplot(611);
  hold on;
  plot(t,k(t),'*');
  ylabel('k','fontsize',15);
  subplot(612);
  hold on;
  plot(t,post(t+1),'*');
  ylabel('^p(k,mu|y)','fontsize',15);
  subplot(613);
  hold on;
  plot(t,msError,'r*');
  ylabel('Train error','fontsize',15);
  subplot(614);
  hold on;
  plot(t,msErrorv,'r*');
  ylabel('Test error','fontsize',15);
  subplot(615);
  hold on
  plot(t,Temp(t),'m*');
  ylabel('Temp','fontsize',15);
  subplot(616);
  hold on;
  bar([1 2 3 4 5 6 7 8 9 10 11 12 13],[match aUpdate rUpdate aBirth rBirth aDeath rDeath aMerge rMerge aSplit rSplit aRW rRW]);
  ylabel('Acceptance','fontsize',15);
  xlabel('match aU rU aB rB aD rD aM rM aS rS aRW rRW','fontsize',15)
  end;
end;











