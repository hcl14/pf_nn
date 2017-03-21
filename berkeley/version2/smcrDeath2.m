function  [mov,aDeath,rDeath] = smcrDeath2(aDeath,rDeath,mov,movO,x,y,par,bFunction,s,t,birth,death);
% PURPOSE : Performs the death move of the reversible jump MCMC algorithm.
% INPUTS  : - aDeath: Number of times the death move has been accepted.
%           - rDeath: Number of times the death move has been rejected.
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
insideDeath=1;
uD=rand(1);

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

% DELETE A BASIS FUNCTION AT RANDOM:
% =================================
Pos= unidrnd(mov.k(s));
if Pos==1
  MuProp = mov.mu{s}(2:mov.k(s),:);
  alpha1 = mov.alpha{s}(1:Pos+d+1-1,1);
  alpha2 = mov.alpha{s}(Pos+d+2:mov.k(s)+d+1,1);
  alphaProp = [alpha1; alpha2];
  P1 = mov.P{s}(1:Pos+d+1-1,1:Pos+d+1-1);
  P2 = mov.P{s}(1:Pos+d+1-1,Pos+d+1+1:mov.k(s)+d+1);
  P3 = mov.P{s}(Pos+d+1+1:mov.k(s)+d+1,1:Pos+d+1-1);
  P4 = mov.P{s}(Pos+d+1+1:mov.k(s)+d+1,Pos+d+1+1:mov.k(s)+d+1);
  PProp = [P1 P2; P3 P4];
elseif Pos==mov.k(s)
  MuProp = mov.mu{s}(1:mov.k(s)-1,:);
  alphaProp = mov.alpha{s}(1:mov.k(s)+d,1);
  PProp = mov.P{s}(1:mov.k(s)+d,1:mov.k(s)+d);  
else
  MuProp = [mov.mu{s}(1:Pos-1,:); mov.mu{s}(Pos+1:mov.k(s),:)];
  alpha1 = mov.alpha{s}(1:Pos+d+1-1,1);
  alpha2 = mov.alpha{s}(Pos+d+2:mov.k(s)+d+1,1);
  alphaProp = [alpha1; alpha2];
  P1 = mov.P{s}(1:Pos+d+1-1,1:Pos+d+1-1);
  P2 = mov.P{s}(1:Pos+d+1-1,Pos+d+1+1:mov.k(s)+d+1);
  P3 = mov.P{s}(Pos+d+1+1:mov.k(s)+d+1,1:Pos+d+1-1);
  P4 = mov.P{s}(Pos+d+1+1:mov.k(s)+d+1,Pos+d+1+1:mov.k(s)+d+1);
  PProp = [P1 P2; P3 P4];
end;
Mprop=zeros(1,mov.k(s)+d+1-1);
Mprop(:,1) = 1;
Mprop(:,2:d+1) = x(t+1,:);
for j=d+2:mov.k(s)+d+1 - 1,
  Mprop(:,j) = feval(bFunction,MuProp(j-d-1,:),x(t+1,:));
end;
yprP = Mprop*alphaProp;

% EVALUATE ACCEPTANCE:
% ===================
QR = par.dalpha.*eye(size(mov.P{s}));
QP = par.dalpha.*eye(size(PProp));
covP = exp(mov.sigma(s)) + Mprop*(PProp+QP)*Mprop';
covR = exp(mov.sigma(s)) + M*(mov.P{s}+QR)*M';
likP = exp(-0.5*(y(t+1,1)-yprP)*inv(covP)*(y(t+1,1)-yprP)');
likR = exp(-0.5*(y(t+1,1)-yprR)*inv(covR)*(y(t+1,1)-yprR)');
ratio = (likP*par.probBirth)/(likR*par.probUpdate);
acceptance = min(1,ratio);


% METROPOLIS STEP:
% ===============
if (uD<acceptance),
  mov.mu{s} = MuProp;
  mov.alpha{s} = alphaProp;
  mov.P{s} = PProp;
  mov.k(s) = mov.k(s) - 1;
  aDeath=aDeath+1;
else
  mov.mu{s} = mov.mu{s};
  mov.alpha{s} = mov.alpha{s};
  mov.P{s} = mov.P{s};
  mov.k(s) = mov.k(s);
  rDeath=rDeath+1;
end;





















