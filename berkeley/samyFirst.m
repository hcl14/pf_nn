% This demo shows how the Simulated Annealing Metropolis-Hastings algorithm can be used find the maxima of 
% a mixture of 2 Gaussians if we can sample from a Gaussian proposal. It uses 1000 parallel chains, which 
% are initialised uniformly.
% Written by Christophe Andrieu and Nando de Freitas. Please refer to the NIPS MCMC tutorial for more info.


% DEFINITIONS:
% ===========

measure = 0.4;                   % Histogram correction factor (measure) to get a density.
N = 1000;                        % Number of Markov chains.
nb_iter = 9;                     % Number of iterations.
sigma = 2;                       % Standard deviation of the target components.
x = zeros(N, 1);                 % Markov chain (unknowns).
sigma_prop = 2;                 % Standard deviation of the Gaussian proposal.
N_hist = 50;                     % Number of bins in the histogram.
a = zeros(N_hist, nb_iter);
b = zeros(N_hist, nb_iter);
figure(1)
clf;


% INITIALISE THE CHAINS:
% =====================

x = zeros(N, nb_iter);
x(:, 1) = 20 * rand(N, 1);


% FIRST ITERATION: (Start with 1000 uniformly distributed chains).
% ===============

iter = 1;
subplot(3, 3, iter)
x_t = linspace(min(x(:, iter)), max(x(:, iter)), 1000);
y_t = 0.3*exp(-.5 * x_t.^2 / sigma^2)/(sqrt(2*pi)*sigma) + 0.7*exp(-.5*(x_t-10).^2/sigma^2)/(sqrt(2*pi)*sigma);
hold on
[bb, aa] = hist(x(:, iter), 50);
a(:, iter) = aa';
b(:, iter) = bb';
bar(aa, bb/(measure*N),'c')
plot(x_t, y_t, 'r','linewidth',2)
text(15,.1,'t_1');
hold off
axis('tight')


T_i = 1.0;                       % Initial temperature.


% ITERATE THE CHAIN:
% =================

for iter = 2:nb_iter

  % METROPOLIS:
  % ==========
  u = rand(N,1);
  for i = 1:N
    z = sigma_prop * randn(1, 1);
    alpha1 = 0.3*exp(-.5 * (x(i,iter-1)+z).^2 / sigma^2)/(sqrt(2*pi)*sigma) + 0.7*exp(-.5*((x(i,iter-1)+z)-10).^2/sigma^2)/(sqrt(2*pi)*sigma);
    alpha2 = 0.3*exp(-.5 * (x(i,iter-1)).^2 / sigma^2)/(sqrt(2*pi)*sigma) + 0.7*exp(-.5*((x(i,iter-1))-10).^2/sigma^2)/(sqrt(2*pi)*sigma);
    alpha = (alpha1/alpha2)^(1/T_i);
    if(u(i)<alpha)
       x(i, iter) = x(i, iter - 1) + z;
     else
       x(i, iter) = x(i, iter - 1);
     end
   end

   % PLOT THE HISTOGRAMS:
   % ===================
   subplot(3, 3, iter)
   x_t = linspace(min(x(:, iter)), max(x(:, iter)), 1000);
   y_t = 0.3*exp(-.5 * x_t.^2 / sigma^2)/(sqrt(2*pi)*sigma) + 0.7*exp(-.5*(x_t-10).^2/sigma^2)/(sqrt(2*pi)*sigma);
   hold on
   [bb, aa] = hist(squeeze(x(:, iter)), N_hist);
   a(:, iter) = aa';
   b(:, iter) = bb';
   bar(a(:, iter), b(:, iter)/(measure*N),'c')
   plot(x_t, y_t, 'r','linewidth',2)
   hold off
   axis('tight')
   if iter==2
     text(15,.1,'t_2');
   elseif iter==3
     text(15,.1,'t_3');
   elseif iter==4
     text(15,.1,'t_4');
   elseif iter==5
     text(15,.1,'t_5');
   elseif iter==6
     text(15,.1,'t_6');
   elseif iter==7
     text(15,.1,'t_7');
   elseif iter==8
     text(15,.1,'t_8');
   elseif iter==9
     text(15,.1,'t_9');
   end;

   iteration=iter
 
   T_i = T_i * .25;              % Adjust the cooling schedule.
  
end;







 
