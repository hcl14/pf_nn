function [mov,aBirth,rBirth] = smcrBirth2(aBirth,rBirth,res,mov,movO,x,y,par,bFunction,s,t,birth,death);
% PURPOSE : Performs the birth move of the reversible jump MCMC algorithm.
% INPUTS  : - aBirth: Number of times the birth move has been accepted.
%           - rBirth: Number of times the birth move has been rejected.
%           - mov : Current model parameters.
%           - movO : Old model parameters.
%           - x : Input data.
%           - y : Target data.
%           - par: Simulation parameters.
%           - bFunction: Type of basis function.
%           - t : Current time step.
%           - s : Current sample.
%           - birth: Probability of birth move.
%           - death: Probability of death move

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 21-01-99

if nargin < 12, error('Not enough input arguments.'); end
[arbN,d] = size(x);   % N = number of data, d = dimension of x.
[N,c] = size(y);      % c = dimension of y, i.e. number of outputs.
insideBirth=1;
uB=rand(1);
M=zeros(1,mov.k(s)+d+1);
M(:,1) = 1;
M(:,2:d+1) = x(t+1,:);
for j=d+2:mov.k(s)+d+1,
  M(:,j) = feval(bFunction,mov.mu{s}(j-d-1,:),x(t+1,:));
end;
yprR = M*mov.alpha{s};


% PROPOSAL: ADD A NEW BASIS FUNCTION:
% ==================================
MuProp = [mov.mu{s};  movO.mu{s}(res.pos(s),:) + sqrt(par.dmu)*randn(1,d)];
tmpDiag= par.P02*eye(mov.k(s)+d+1+1); 
tmpDiag(1:mov.k(s)+d+1,1:mov.k(s)+d+1) = mov.P{s};
PProp = tmpDiag;       
alphaProp = [mov.alpha{s}; par.alphaMean + sqrt(par.alpha0)*randn(1,1)];
Mprop=zeros(1,mov.k(s)+d+1+1);
Mprop(:,1) = 1;
Mprop(:,2:d+1) = x(t+1,:);
for j=d+2:mov.k(s)+d+1 + 1,
  Mprop(:,j) = feval(bFunction,MuProp(j-d-1,:),x(t+1,:));
end;
yprP = Mprop*alphaProp;

% EVALUATE ACCEPTANCE:
% ===================
QR = par.dalpha.*eye(size(mov.P{s}));
QP = par.dalpha.*eye(size(PProp)); % in this case QR=QP
covP = exp(mov.sigma(s)) + Mprop*(PProp+QP)*Mprop';
covR = exp(mov.sigma(s)) + M*(mov.P{s}+QR)*M';
likP = exp(-0.5*(y(t+1,1)-yprP)*inv(covP)*(y(t+1,1)-yprP)');
likR = exp(-0.5*(y(t+1,1)-yprR)*inv(covR)*(y(t+1,1)-yprR)');
ratio = (likP*par.probUpdate)/(likR*par.probBirth);
acceptance = min(1,ratio);

% METROPOLIS STEP:
% ===============
if (uB<acceptance),
  mov.mu{s} = MuProp;
  mov.alpha{s} = alphaProp;
  mov.P{s} = PProp;
  mov.k(s) = mov.k(s) + 1;
  aBirth=aBirth+1;
else
  mov.mu{s} = mov.mu{s};
  mov.alpha{s} = mov.alpha{s};
  mov.P{s} = mov.P{s};
  mov.k(s) = mov.k(s);
  rBirth=rBirth+1;
end;










