function [k,mu,M,match,aBirth,rBirth] = sarBirth(match,aBirth,rBirth,k,mu,M,x,y,t,criterion,bFunction,walkInt,walk);
% PURPOSE : Performs the birth move of the reversible jump MCMC simulated annealing.
% INPUTS  : - match: Number of times a basis function already exists (probability zero in theory).
%             For completeness sake, I don't allow for duplicate basis functions here.
%           - aBirth: Number of times the birth move has been accepted.
%           - rBirth: Number of times the birth move has been rejected.
%           - k : Number of basis functions.
%           - mu : Basis functions centres.
%           - M : Regressors matrix.
%           - x : Input data.
%           - y : Target data.
%           - t : Current time step.
%           - bFunction: Type of basis function.
%           - criterion: Model selection criterion (MDL or AIC).
%           - walk, walkInt: Parameters defining the compact set from which mu is sampled.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 21-05-99

if nargin < 12, error('Not enough input arguments.'); end
[N,d] = size(x);      % N = number of data, d = dimension of x.
[N,c] = size(y);      % c = dimension of y, i.e. number of outputs.
insideBirth=1;
uB=rand(1);

% INITIALISE H AND P MATRICES:
% ===========================
invH=zeros(k(t)+1+d,k(t)+1+d,c);
P=zeros(N,N,c);
invHproposal=zeros(k(t)+2+d,k(t)+2+d,c);
Pproposal=zeros(N,N,c);
for i=1:c,
  invH(:,:,i) = M'*M;
  P(:,:,i) = eye(N) - M*inv(invH(:,:,i))*M';
end;

% PROPOSE A NEW BASIS FUNCTION:
% ============================
proposal=zeros(1,d);
for i=1:d,
  proposal(1,i)= (min(x(:,i))-walk(i)) + ((max(x(:,i))+walk(i))-(min(x(:,i))-walk(i)))*rand(1,1);
end;

% CHECK IF THE PROPOSED CENTRE ALREADY EXISTS:
% =========================================== 
match1=0;
notEnded=1;
i=1;
if k(t)>0
  while ((match1==0)&(notEnded==1)),
    if (mu{t}(i,:)==proposal),
      match1=1;
    elseif (i<k(t)),
      i=i+1;
    else
      notEnded=0;
    end;
  end;
end;
if (match1>0),
  match = match+1;
  mu{t+1} = mu{t};
  k(t+1) = k(t);
  M=M;
else
  % IF IT DOESN'T EXIST, PERFORM A BIRTH MOVE:
  % =========================================
  accepted=1;
  Mproposal = [M feval(bFunction,proposal,x)];
  for i=1:c,
    invHproposal(:,:,i) = Mproposal'*Mproposal;
    Pproposal(:,:,i) = eye(N) - Mproposal*inv(invHproposal(:,:,i))*Mproposal'; 
  end;
  small = 0; % To avoid numerical problems.
  ratio= inv(k(t)+1) * prod(walkInt) * exp(-criterion) * ((y(:,1)'*P(:,:,1)*y(:,1)+small)/(y(:,1)'*Pproposal(:,:,1)*y(:,1)+small))^(N/2);      
  for i=2:c,
    ratio= ratio * ((y(:,i)'*P(:,:,i)*y(:,i)+small)/(y(:,i)'*Pproposal(:,:,i)*y(:,i)+small))^(N/2); 
  end;
  % METROPOLIS:
  % ==========
  acceptance = min(1,ratio);   
  if (uB<acceptance),
    mu{t+1} = [mu{t}; proposal];
    k(t+1) = k(t)+1;
    M=Mproposal;
    aBirth=aBirth+1;
  else
    mu{t+1} = mu{t};
    k(t+1) = k(t);
    rBirth=rBirth+1;
    M=M;
  end;
end; 












