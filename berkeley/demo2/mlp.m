function y = mlp(x,theta,s1,s2);
% PURPOSE : To simulate a one hidden layer sigmoidal MLP. 
% INPUTS  : - x = The network input.
%           - theta = The network weights.
%           - s1 = Number of neurons in the hidden layer.
%           - s2 = Number of neurons in the output layer (=1).
% OUTPUTS : - y = The network output.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 08-09-98

if nargin < 4, error('Not enough input arguments.'); end

% fill in weight matrices using the parameter vector: 
% ==================================================
[rows,N] = size(x);
w2 = zeros(s2,s1+1);
w1 = zeros(s1,N+1); 
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


