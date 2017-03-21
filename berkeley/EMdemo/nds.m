function net=nds(Y,input,N,cyc,tol,s1,s2,ipar);
% PURPOSE : Estimates the weights of an MLP and the noise covariances using the EM algorithm.
% INPUTS  : - Y = The target data.
%           - input = The input data.
%           - N = Number of data.
%           - cyc = Number of EM iterations.
%           - tol = Change in likelihood stopping criterion.
%           - s1 = Number of hidden layer neurons.
%           - s2 = Number of output layer neurons.
%           - ipar = Initial parameters.
% OUTPUTS : - net.Innov = Last innovations covariance.
%           - net.Q = Process noise covariance.
%           - net.R = Measurement covariance.
%           - net.x0 = Last set of weights.
%           - net.P0 = Last weights covariance.
%           - net.LL = Log likelihood.
%           - net.output = Last network prediction.
%           - net.rms = Rms error.

% AUTHOR  : Nando de Freitas - (Re-uses bits of code by Zoubin Ghahramani)
% DATE    : 09-03-99

net=struct('type','nds','Innov',[],'Q',[],'R',[],'x0',[],'P0',[],'output',[],'LL',[],'rms',[]);
[d,N] = size(input);       % d=number of inputs, N=max time steps.
[N,p] = size(Y);           % p=number of outputs.
lik=0;
LL=[];
K=s2*(s1+1) + s1*(d+1);    % Number of parameters.
R=zeros(p,p,cyc);
Q=zeros(K,K,cyc);
R(:,:,1) = ipar.initR;
Q(:,:,1) = ipar.initQ;
x0 = ipar.initx0;
P0 = ipar.initP0;
for cycle=1:cyc  
  % E STEP:
  % ======  
  oldlik=lik;  
  [rms,lik,Xfin,Pfin,A1,A2,A3,output,Rterm,Innov]=ekfsmooth(Q(:,:,cycle),R(:,:,cycle),x0,P0,Y,input,cycle,s1,s2);
  LL=[LL lik];
  fprintf('cycle %g lik %g rms %g',cycle,lik,rms(cycle));
  if (cycle<=2)
    likbase=lik;
  elseif (lik<oldlik) 
    fprintf(' violation');
  elseif ((lik-likbase)<(1 + tol)*(oldlik-likbase)|~finite(lik)) 
    fprintf('\n');
    break;
  end;
  fprintf('\n');

  % M STEP:
  % ======
  x0=Xfin(1,:)'; 
  P0=Pfin(:,:,1);
  R(:,:,cycle+1)=0;
  for t=1:N
    R(:,:,cycle+1)= R(:,:,cycle+1) + Rterm(:,:,t) + (Y(t)-output(t))'*(Y(t)-output(t));
  end;
  R(:,:,cycle+1)=inv(N)*R(:,:,cycle+1);
  R(:,:,cycle+1)=diag(diag(R(:,:,cycle+1)));
  Q(:,:,cycle+1)=(1/(N-1))*diag(diag(A3-A1')); 
  theR=R(:,:,cycle+1)
  theQ=trace(Q(:,:,cycle+1))
  fprintf('\n');
end;
net.output=output;
net.Q=Q;
net.R=R;
net.x0=x0;
net.P0=P0;
net.LL=LL;
net.Innov=Innov;
net.rms=rms;







