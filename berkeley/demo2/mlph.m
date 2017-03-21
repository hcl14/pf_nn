function [y, H] = mlp(x,theta,s1,s2);
% PURPOSE : To simulate a one hidden layer sigmoidal MLP and
%           return the Jacobian. 
% INPUTS  : - x = The network input.
%           - theta = The network weights.
%           - s1 = Number of neurons in the hidden layer.
%           - s2 = Number of neurons in the output layer (=1).
% OUTPUTS : - y = The network output.
%           - H = The Jacobian matrix.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 08-09-98

if nargin < 4, error('Not enough input arguments.'); end

% fill in weight matrices using the parameter vector: 
% ==================================================
[rows,N] = size(x);
w2 = zeros(s2,s1+1);
w1 = zeros(s1,N+1); 
T = s2*(s1+1) + s1*(N+1);
H = zeros(T,1); % Assuming one single output. 

L=0;
for i = 1:s2,
  w2(i,:)=theta(1,1,L+1:L+s1+1);
  L=L+s1+1;
end;
for i = 1:s1,
  w1(i,:)= theta(1,1,L+1:L+N+1);
  L = L+N+1;
end;

% Compute the network outputs for each layer:
% ==========================================

u1 = w1*[1 ; x']; 
o1 = 1./(1+exp(-u1));
u2 = w2*[1 ; o1];
y = u2;  

% Compute the Jacobian:
% =====================

    % output layer:
    for i = 1:s2,
      for j = 1:(s1+1),
        if j==1
          H(i*(s1+1) + j - (s1+1) ,1)= 1;
        else
          H(i*(s1+1) + j - (s1+1) ,1)= o1(j-1,1);
        end;
      end;
    end;
    
    % Second layer:
    for i = 1:s1,
      for j = 1:(N+1),
        rhs = w2(1,i+1)*o1(i,1)*(1-o1(i,1));
        if j==1
          H(s2*(s1+1) + i*(N+1) + j - (N+1) ,1) = rhs;
        else
          H(s2*(s1+1) + i*(N+1) + j - (N+1) ,1)= rhs * x(:,j-1);
        end;
      end;
    end;

