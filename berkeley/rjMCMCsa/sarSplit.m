function [k,mu,M,aSplit,rSplit] = sarSplit(aSplit,rSplit,k,mu,M,x,y,t,bFunction,criterion,sigStar,walk);
% PURPOSE : Performs the split move of the reversible jump MCMC simulated annealing.
% INPUTS  : - aSplit: Number of times the split move has been accepted.
%           - rSplit: Number of times the split move has been rejected.
%           - k : Number of basis functions.
%           - mu : Basis functions centres.
%           - M : Regressors matrix.
%           - x : Input data.
%           - y : Target data.
%           - t : Current time step.
%           - bFunction: Type of basis function.
%           - criterion: Model selection criterion (MDL or AIC).
%           - sigStar: Split/merge move parameter.
%           - walk: Parameter defining the compact set from which mu is sampled.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 21-05-99

if nargin < 11, error('Not enough input arguments.'); end
[N,d] = size(x);      % N = number of data, d = dimension of x.
[N,c] = size(y);      % c = dimension of y, i.e. number of outputs.
insideSplit=1;
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

% PROPOSE A BASIS FUNCTION TO BE SPLIT:
% ====================================
position = unidrnd(length(mu{t}(:,1)),1,1);
proposal = mu{t}(position,:);
uu = rand(size(proposal));
mu1 = proposal - uu*sigStar;
mu2 = proposal + uu*sigStar;

% CONSTRAIN RANDOM WALK:
% =====================
for i=1:d,
  mu1(:,i) = min(mu1(:,i),max(x(:,i))+walk(i));
  mu1(:,i) = max(mu1(:,i),min(x(:,i))-walk(i));
  mu2(:,i) = min(mu2(:,i),max(x(:,i))+walk(i));
  mu2(:,i) = max(mu2(:,i),min(x(:,i))-walk(i));
end

% Reduce the size of M by 1:
proposalPos= d+1+position;
if (proposalPos==d+1+k(t)),
  Mproposal = [M(:,1:proposalPos-1)];    
else
  Mproposal = [M(:,1:proposalPos-1) M(:,proposalPos+1:k(t)+d+1)];      
end;
% Add the new split components to m:
Mproposal = [Mproposal feval(bFunction,mu1,x) feval(bFunction,mu2,x)];
for i=1:c,
  invHproposal(:,:,i) = (Mproposal'*Mproposal); 
  Pproposal(:,:,i) = eye(N) - Mproposal*inv(invHproposal(:,:,i))*Mproposal'; 
end;

% COMPUTE THE ACCEPTANCE RATION:
% =============================
Jacobian = sigStar;
small = 0; % To avoid numerical problems.
ratio= Jacobian*inv(k(t)+1) * k(t) * exp(-criterion) * ((y(:,1)'*P(:,:,1)*y(:,1)+small)/(y(:,1)'*Pproposal(:,:,1)*y(:,1)+small))^(N/2);for i=2:c,
  ratio= ratio * ((y(:,i)'*P(:,:,i)*y(:,i)+small)/(y(:,i)'*Pproposal(:,:,i)*y(:,i)+small))^(N/2); 
end;
acceptance = min(1,ratio);   

% PERFORM DISTANCE TEST TO ENSURE REVERSIBILITY:
% =============================================
dist1 = zeros(k(t),1);
dist2 = norm(mu1-mu2); 
violation =0;
for i=1:k(t),
  if i== position,
    dist1(i) = inf; 
  else
    dist1(i)=norm(mu1-mu{t}(i,:)); % Euclidean distance;
  end;
  if dist1(i)<dist2     % Don't accept.
    violation=1;
    acceptance = 0;
  end;
end; 

% APPLY METROPOLIS-HASTINGS STEP:
% ==============================
if (uB<acceptance),
  previousMu = mu{t};
  if (proposalPos==(1+d+1)),
    muTrunc = [previousMu(2:k(t),:)]; 
  elseif (proposalPos==(1+d+k(t))),
    muTrunc = [previousMu(1:k(t)-1,:)];
  else
    muTrunc = [previousMu(1:proposalPos-1-d-1,:); previousMu(proposalPos-d-1+1:k(t),:)];
  end;
  mu{t+1} = [muTrunc; mu1; mu2];
  k(t+1) = k(t)+1;
  M=Mproposal;
  aSplit=aSplit+1;
else
  mu{t+1} = mu{t};
  k(t+1) = k(t);
  rSplit=rSplit+1;
  M=M;
end;












