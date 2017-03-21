% PURPOSE : To demonstrate the EM algoriths for estimating neural network
%           weights and signal noise simultaneously. We apply it to a classification
%           problem (tremor data - kindly supplied by Will Penny and Stephen Roberts).
                      
% AUTHOR  : Nando de Freitas 
% DATE    : 09-03-99

% SIMULATION PARAMETERS:
% =====================
N = 179;                  % Number of time steps.
iterations = 50;          % Number of EM iterations.
tolerance = 0.0001;       % Change in likelihood stoping criterion.
s1 = 10;                  % No of neurons in the hidden layer.
s2 = 1;                   % No of neurons in the output layer.

% LOAD TREMOR DATA:
% ================
load tremor;
data =[x_tr t_tr];
data = data(randperm(N),:);     % Order the data randomly.
xv = x_te';                     % Test set input data.
dv = t_te;                      % Test set target data.
x = data(:,1:2)';               % Train set input data.
d = data(:,3);                  % Train set target data.

% INITIALISATION:
% ==============
inputd=2;                             % Input dimension.
K=s2*(s1+1) + s1*(inputd+1);          % Total No of parameters.
ipar.initR=0.01*rand(1)*eye(s2);      % Initial measurement noise.
ipar.initQ=1*eye(K);                  % Initial process noise.
ipar.initx0=1*randn(1,K)';            % Initial weights.
ipar.initP0=100*eye(K);               % Initial weights covariance.

% PERFORM EM ESTIMATION:
% =====================
net=nds(d,x,N,iterations,tolerance,s1,s2,ipar);

% COMPUTE THE PERCENTAGE TRAIN ERROR:
% ==================================
yp = zeros(size(net.output));
for t=1:N
  if net.output(t)>0.5
    yp(t)=1;
  else
    yp(t)=0;
  end;
end;
error=yp-d;
percentageError=sum(abs(yp-d))*100/N  % Train error.
Xpre=net.x0';
inputd=2;
w2 = zeros(s2,s1+1);
w1 = zeros(s1,inputd+1);

% COMPUTE NETWORK OUTPUTS AND TEST ERROR:
% ======================================
for i = 1:s2,
  for j = 1:(s1+1),
    w2(i,j)= Xpre(1,i*(s1+1)+j-(s1+1));
  end;
end;
for i = 1:s1,
  for j = 1:(inputd+1),
    w1(i,j)= Xpre(1,s2*(s1+1) +i*(inputd+1)+j-(inputd+1));
  end;
end;
ypv=zeros(size(dv));
for t=1:N-1,
  u1 = w1*[1 ; xv(:,t)]; 
  o1 = 1./(1+exp(-u1));
  u2 = w2*[1 ; o1];
  ypv(t)=u2;
end;
yproc=ypv;
for t=1:N-1
  if ypv(t)>0.5   
    ypv(t)=1;
  else
    ypv(t)=0;
  end;
end;
errorv=ypv-dv;
percentageErrorv=sum(abs(ypv-dv))*100/(N-1) % Test error.

% PLOT SIMULATION RESULTS:
% =======================
figure(2)
clf;
subplot(221)
plot(net.LL(2:length(net.LL))/(100*log(2)));             
ylabel('Log-likelihood','fontsize',15); 
xlabel('Iterations of EM','fontsize',15); 
grid
subplot(222)
plot(diff(net.LL(2:length(net.LL))));
ylabel('Convergence rate','fontsize',15); 
xlabel('Iterations of EM','fontsize',15); 
grid
[rr,rr,zz]=size(net.R);
[qq,qq,zz]=size(net.Q);
R=zeros(zz);
trQ=zeros(zz);
for t=1:zz,
  R(t)=net.R(:,:,t);
  trQ(t)=trace(net.Q(:,:,t));
end;
subplot(223)
plot(1:zz,R); 
ylabel('R','fontsize',15); 
xlabel('Iterations of EM','fontsize',15); 
grid
subplot(224)
plot(1:zz,trQ);
ylabel('trace(Q)','fontsize',15); 
xlabel('Iterations of EM','fontsize',15); 
grid

% PLOT DECISION BOUNDARY:
% ======================
[xi,yi]=meshgrid(-0.5:.01:1,-0.5:.01:1);
outputv=zeros(length([-0.5:.01:1]),length([-0.5:.01:1]));
for t1=-0.5:.01:1,  
  for t2=-0.5:.01:1,
    xv = [t1 t2]';
    u1 = w1*[1 ; xv]; 
    o1 = 1./(1+exp(-u1));
    u2 = w2*[1 ; o1];
    outputv(round(100*t1+51),round(100*t2+51))=u2';  
  end;
end;
figure(1)
clf;
hold on
for i=1:N
  if d(i)>.5
    plot(x(1,i),x(2,i),'ro');
  else
    plot(x(1,i),x(2,i),'+');
  end;
  hold on;
end;
hold on
contour(yi,xi,outputv,[0.5 0.5],'k');
contour(yi,xi,outputv,[0.6 0.6],'k--');
contour(yi,xi,outputv,[0.4 0.4],'k--');
ylabel('Patients [o] and control [+]','fontsize',15); 
xlabel('x_{1}','fontsize',15); 
title('Decision boundary','fontsize',15); 

% PLOT ROC CURVE:
% ==============
figure(3)
clf;
Nv=N-1;
truePos=zeros(Nv,1);
falsePos=zeros(Nv,1);
ypsort=sort(yproc);
for i=1:Nv,
  yptmp = yproc;
  for t=1:Nv,
    if yproc(t)>ypsort(i)
      yptmp(t)=1;
    else
      yptmp(t)=0;
    end;
  end;
  errorv= 2*(yptmp+2*ones(size(yptmp))) - dv;
  truePos(i) = length(find(errorv==5))/89;
  falsePos(i) = length(find(errorv==6))/89;
end;
truePos=fliplr(truePos');
falsePos=fliplr(falsePos');
plot(falsePos,truePos,'r-',0:.01:1,0:.01:1,'--');
ylabel('True positives','fontsize',15); 
xlabel('False positives','fontsize',15); 
title('ROC curve','fontsize',15); 
grid;













