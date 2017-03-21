function  [rms,lik,Xfin,Pfin,A1,A2,A3,output,Rterm,Innov]=ekfsmooth(Q,R,x0,P0,Y,input,cycle,s1,s2);
% PURPOSE : Estimates the weights of an MLP using a Rauch-Tung-Striebel smoother (E step).
% INPUTS  : - Q = Process noise covariance.
%           - R = Measurement covariance.
%           - x0 = Initial set of weights.
%           - P0 = Initial weights covariance.
%            - Y = The output data.
%           - input = The input data.
%           - N = Number of data.
%           - cycle = Current EM iterations.
%           - s1 = Number of hidden layer neurons.
%           - s2 = Number of output layer neurons.
% OUTPUTS : - rms = Rms error.
%           - lik = Log likelihood.
%           - xfin = Output set of weights.
%           - Pfin = Output weights covariance.
%           - output = Network prediction.
%           - Innov = Innovations covariance.
%           - A1, A2, A3, Rterm = Terms for evaluating R and Q in the M phase.

% AUTHOR  : Nando de Freitas - (Re-uses bits of code by Zoubin Ghahramani)
% DATE    : 09-03-99

[N p]=size(Y);
[d N]= size(input);
K=length(x0);
tiny=exp(-700);
I=eye(K);
const=(2*pi)^(-p/2);
lik=0;
K=s2*(s1+1) + s1*(d+1);
Xpre=zeros(K,1);   % P(x_t | y_1 ... y_{t-1})
Xcur=zeros(N,K);   % P(x_t | y_1 ... y_t)
Xfin=zeros(N,K);   % P(x_t | y_1 ... y_T)    given all outputs
Ppre=zeros(K,K,N);
Pcur=zeros(K,K,N);
Pfin=zeros(K,K,N); 
Pt=zeros(K,K); 
Pcov=zeros(K,K); 
Kcur=zeros(K,p);
invP=zeros(p,p);
J=zeros(K,K,N);
Innov=zeros(p,p,N);
Rterm=zeros(p,p,N);
output=zeros(N,p);
C=zeros(K,p,N);
w2 = zeros(s2,s1+1);
w1 = zeros(s1,d+1);

% FORWARD PASS:
% ============
R=R+(R==0)*tiny;
Xpre = x0';
Ppre(:,:,1)=P0;
for t=1:N
  % COMPUTE NETWORK OUTPUTS:
  % =======================
  for i = 1:s2,
    for j = 1:(s1+1),
      w2(i,j)= Xpre(1,i*(s1+1)+j-(s1+1));
    end;
  end;
  for i = 1:s1,
    for j = 1:(d+1),
      w1(i,j)= Xpre(1,s2*(s1+1) +i*(d+1)+j-(d+1));
    end;
  end;
  % Compute the network outputs for each layer:
  u1 = w1*[1 ; input(:,t)]; 
  o1 = 1./(1+exp(-u1));
  u2 = w2*[1 ; o1];
  output(t,:)=u2';  
  % FILL THE JACOBIAN MATRIX (C):
  % ============================
  for outputs = 1:p,
    % output layer:
    for i = 1:s2,
      for j = 1:(s1+1),
        if j==1
          C(i*(s1+1) + j - (s1+1) ,outputs,t)= 1;
        else
          C(i*(s1+1) + j - (s1+1) ,outputs,t)= o1(j-1,1);
        end;
      end;
    end;
    % Second layer:
    for i = 1:s1,
      for j = 1:(d+1),
        rhs = w2(outputs,i+1)*o1(i,1)*(1-o1(i,1));
        if j==1
          C(s2*(s1+1) + i*(d+1) + j - (d+1) ,outputs,t) = rhs;
        else
          C(s2*(s1+1) + i*(d+1) + j - (d+1) ,outputs,t)= rhs * input(j-1,t);
        end;
      end;
    end;
  end;
  % PERFORM KALMAN FILTERING:
  % ========================
  Innov(:,:,t) = (diag(diag(R))+C(:,:,t)'*Ppre(:,:,t)*C(:,:,t));
  invP=inv(Innov(:,:,t));
  CP=C(:,:,t)*invP;
  Kcur=Ppre(:,:,t)*CP;
  KC=Kcur*C(:,:,t)';
  Ydiff=Y(t,:)-output(t,:);
  Xcur(t,:)=Xpre+Ydiff*Kcur';
  Pcur(:,:,t)=Ppre(:,:,t)-KC*Ppre(:,:,t);
  if (t<N) % Do a random walk:
    Xpre=Xcur(t,:); 
    Ppre(:,:,t+1)=Pcur(:,:,t)+Q;
  end;
  % CALCULATE LIKELIHOOD:
  % ====================
  detiP=sqrt(det(invP));
  if (isreal(detiP) & detiP>0)
    lik=lik+N*log(detiP)-0.5*sum(sum(Ydiff.*(Ydiff*invP)));
  else
    problem=1;
  end;
end;  
lik=lik+N*log(const);
% COMPUTE RMS:
% ===========
rmsgekf=0;
for t=1:N
  rmsgekf = rmsgekf + sum((Y(t,:)-output(t,:)).^2);
end;
rmsgekf=(1/N)*rmsgekf;  
rms(cycle,1)=rmsgekf;

% BACKWARD PASS ALA RAUCH-TUNG-STRIEBEL:
% =====================================
A1=zeros(K);
A2=zeros(K);
A3=zeros(K);
Ptsum=zeros(K);  
t=N; 
Xfin(t,:)=Xcur(t,:);
Pfin(:,:,t)=Pcur(:,:,t); 
Rterm(:,:,t)=C(:,:,t)'*Pfin(:,:,t)*C(:,:,t);
Pt=Pfin(:,:,t) + Xfin(t,:)'*Xfin(t,:)/N; 
A2= -Pt;
Ptsum=Pt;
for t=(N-1):-1:1
  J(:,:,t)=Pcur(:,:,t)*inv(Ppre(:,:,t+1));
  Xfin(t,:)=Xcur(t,:)+(Xfin(t+1,:)-Xcur(t,:))*J(:,:,t)';
  Pfin(:,:,t)=Pcur(:,:,t)+J(:,:,t)*(Pfin(:,:,t+1)-Ppre(:,:,t+1))*J(:,:,t)';
  Pt=Pfin(:,:,t) + Xfin(t,:)'*Xfin(t,:)/N; 
  Ptsum=Ptsum+Pt;
  Rterm(:,:,t)=C(:,:,t)'*Pfin(:,:,t)*C(:,:,t);
end;
A3= Ptsum-Pt;
A2= Ptsum+A2;
t=N;  
Pcov=(I-KC)*Pcur(:,:,t-1);
A1=A1+Pcov+Xfin(t,:)'*Xfin(t-1,:)/N;
for t=(N-1):-1:2
  Pcov=(Pcur(:,:,t)+J(:,:,t)*(Pcov-Pcur(:,:,t)))*J(:,:,t-1)';
  A1=A1+Pcov+Xfin(t,:)'*Xfin(t-1,:)/N;
end;    











