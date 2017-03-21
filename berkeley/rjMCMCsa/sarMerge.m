function [k,mu,M,aMerge,rMerge] = sarMerge(aMerge,rMerge,k,mu,M,x,y,t,bFunction,criterion,sigStar);
% PURPOSE : Performs the split move of the reversible jump MCMC simulated annealing.
% INPUTS  : - aMerge: Number of times the merge move has been accepted.
%           - rMerge: Number of times the merge move has been rejected.
%           - k : Number of basis functions.
%           - mu : Basis functions centres.
%           - M : Regressors matrix.
%           - x : Input data.
%           - y : Target data.
%           - t : Current time step.
%           - bFunction: Type of basis function.
%           - criterion: Model selection criterion (MDL or AIC).
%           - sigStar: Split/merge move parameter.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 21-01-99

if nargin < 11, error('Not enough input arguments.'); end
[N,d] = size(x);      % N = number of data, d = dimension of x.
[N,c] = size(y);      % c = dimension of y, i.e. number of outputs.

insideMerge=1;
uD=rand(1);

% INITIALISE H AND P MATRICES:
% ===========================
invH=zeros(k(t)+1+d,k(t)+1+d,c);
P=zeros(N,N,c);
invHproposal=zeros(k(t)+d,k(t)+d,c);
Pproposal=zeros(N,N,c);
for i=1:c,
  invH(:,:,i) = M'*M;
  P(:,:,i) = eye(N) - M*inv(invH(:,:,i))*M';
end;


position = unidrnd(length(mu{t}(:,1)),1,1);
mu1 = mu{t}(position,:);
dist = zeros(k(t),1);
for i=1:k(t),
  if i== position,
    dist(i) = inf; 
  else
    dist(i)=norm(mu1-mu{t}(i,:)); % Euclidean distance;
  end;
end; 
position2 = find(dist == min(dist));
mu2 = mu{t}(position2,:);
mumg = .5*(mu1 + mu2);

% extract components:
proposalPos1 = position + d+1;
if (proposalPos1==d+1+k(t)),
  Mproposal = [M(:,1:proposalPos1-1)];     
else
  Mproposal = [M(:,1:proposalPos1-1) M(:,proposalPos1+1:k(t)+d+1)];      
end;

if position2>position,
  proposalPos2 = position2 + d+1;
  if (proposalPos2==d+1+k(t)),
    Mproposal = [Mproposal(:,1:proposalPos2-2)];     
  else
    Mproposal = [Mproposal(:,1:proposalPos2-2) Mproposal(:,proposalPos2:k(t)+d)];      
  end;
elseif position2<position,
  proposalPos2 = position2 + d+1;
  Mproposal = [Mproposal(:,1:proposalPos2-1) Mproposal(:,proposalPos2+1:k(t)+d)];      
else
  error('Something wrong with merge move');
end;
% add merged component:
Mproposal = [Mproposal feval(bFunction,mumg,x)]; 

for i=1:c,
  invHproposal(:,:,i) = (Mproposal'*Mproposal); 
  Pproposal(:,:,i) = eye(N) - Mproposal*inv(invHproposal(:,:,i))*Mproposal'; 
end;

% PERFORM A MERGE MOVE:
% ====================
Jacobian = inv(sigStar);
small = 0; % To avoid numerical problems.
ratio= Jacobian * k(t) * inv(k(t)-1) * exp(criterion) * ((y(:,1)'*P(:,:,1)*y(:,1)+small)/(y(:,1)'*Pproposal(:,:,1)*y(:,1)+small))^(N/2);      
for i=2:c,
  ratio= ratio * ((y(:,i)'*P(:,:,i)*y(:,i)+small)/(y(:,i)'*Pproposal(:,:,i)*y(:,i)+small))^(N/2); 
end;
acceptance = min(1,ratio);  

if min(dist)<2*sigStar
  acceptance = 0;   % To ensure reversibility.
end;


if (uD<acceptance),
  previousMu = mu{t};
  if (proposalPos1==(1+d+1)),
    muTrunc = [previousMu(2:k(t),:)]; 
  elseif (proposalPos1==(1+d+k(t))),
    muTrunc = [previousMu(1:k(t)-1,:)];
  else
    muTrunc = [previousMu(1:proposalPos1-1-d-1,:); previousMu(proposalPos1-d-1+1:k(t),:)];
  end;
  if position2>position,
    if (proposalPos2==(1+d+k(t))),
      muTrunc = [muTrunc(1:k(t)-2,:)];
    else
      muTrunc = [muTrunc(1:proposalPos2-1-d-2,:); muTrunc(proposalPos2-d-1:k(t)-1,:)];
    end;
  elseif position2<position,
    if (proposalPos2==(1+d+1)),
      muTrunc = [muTrunc(2:k(t)-1,:)]; 
    else
      muTrunc = [muTrunc(1:proposalPos2-1-d-1,:); muTrunc(proposalPos2-d-1+1:k(t)-1,:)];
    end;
  else
    error('Something wrong with merge move');
  end;
  mu{t+1} = [muTrunc; mumg];
  k(t+1) = k(t)-1;
  M=Mproposal;
  aMerge=aMerge+1;
else
  mu{t+1} = mu{t};
  k(t+1) = k(t);
  rMerge=rMerge+1;
  M=M;
end;










