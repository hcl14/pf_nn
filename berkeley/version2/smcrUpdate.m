function [mov,aUpdate,rUpdate] = smcrUpdate(aUpdate,rUpdate,res,mov,movO,x,y,par,bFunction,s,t);
% PURPOSE : Performs the update move of the reversible jump MCMC algorithm.
% INPUTS  : - aUpdate: Number of times the update move has been accepted.
%           - rUpdate: Number of times the update move has been rejected.
%           - mov : Current model parameters.
%           - movO : Old model parameters.
%           - x : Input data.
%           - y : Target data.
%           - par: Simulation parameters.
%           - bFunction: Type of basis function.
%           - t : Current time step.
%           - s : Current sample.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 21-01-99

if nargin < 10, error('Not enough input arguments.'); end
[arbN,d] = size(x);   % N = number of data, d = dimension of x.
[N,c] = size(y);      % c = dimension of y, i.e. number of outputs.
insideUpdate=1;
uU=rand(1);

M=zeros(1,mov.k(s)+d+1);
M(:,1) = 1;
M(:,2:d+1) = x(t+1,:);
for j=d+2:mov.k(s)+d+1,
  M(:,j) = feval(bFunction,mov.mu{s}(j-d-1,:),x(t+1,:));
end;
yprR = M*mov.alpha{s};

% PROPOSAL:
% ========
[oldk,arb] = size(movO.mu{s});
MuProp=zeros(mov.k(s),d);
if oldk==mov.k(s)
  for i=1:d,
    MuProp(:,i) = movO.mu{s}(:,i) + sqrt(par.dmu)*randn(mov.k(s),1);
  end;  
elseif oldk<mov.k(s)
% Add basis function at the end.
  for i=1:d,
    MuProp(1:oldk,i) = movO.mu{s}(:,i) + sqrt(par.dmu)*randn(oldk,1);
  end; 
  MuProp(oldk+1,:) = [x(t+1,:)+sqrt(par.mu02)*randn(1,d)];
elseif oldk>mov.k(s)      
% Delete basis function at the same position as in the prediction step.
  for i=1:d,
    if res.pos(s)==1
      MuProp(:,i) = movO.mu{s}(2:oldk,i) + sqrt(par.dmu)*randn(mov.k(s),1);
    elseif res.pos(s)==oldk
      MuProp(:,i) = movO.mu{s}(1:oldk-1,i)+ sqrt(par.dmu)*randn(mov.k(s),1);
    else
      MuProp(:,i) = [movO.mu{s}(1:res.pos(s)-1,i); movO.mu{s}(res.pos(s)+1:oldk,i)]+ sqrt(par.dmu)*randn(mov.k(s),1);
    end; 
  end; 
end;
Mprop=zeros(1,mov.k(s)+d+1);
Mprop(:,1) = 1;
Mprop(:,2:d+1) = x(t+1,:);
for j=d+2:mov.k(s)+d+1,
  Mprop(:,j) = feval(bFunction,MuProp(j-d-1,:),x(t+1,:));
end;
yprP = Mprop*mov.alpha{s};

% EVALUATE ACCEPTANCE:
% ===================
Q = par.dalpha.*eye(size(mov.P{s}));
covP = exp(mov.sigma(s)) + Mprop*(mov.P{s}+Q)*Mprop';
covR = exp(mov.sigma(s)) + M*(mov.P{s}+Q)*M';
likP = exp(-0.5*(y(t+1,1)-yprP)*inv(covP)*(y(t+1,1)-yprP)');
likR = exp(-0.5*(y(t+1,1)-yprR)*inv(covR)*(y(t+1,1)-yprR)');
ratio = (likP)/(likR);
acceptance = min(1,ratio);

% METROPOLIS STEP:
% ===============
if (uU<acceptance),
  mov.mu{s} = MuProp;
  aUpdate=aUpdate+1;
else
  mov.mu{s} = mov.mu{s};
  rUpdate=rUpdate+1;
end;












