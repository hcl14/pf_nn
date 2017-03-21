# Archieved from web archive

http://web.archive.org/web/20000917014909/http://www.cs.berkeley.edu/~jfgf/software.html



-------



# Some of my Matlab demos

## Rao Blackwellised Particle Filtering for Dynamic Bayesian Networks
In this demo, we show how to use Rao-Blackwellised particle filtering to exploit the conditional independence structure of a simple DBN.

The derivation and details are presented in: A. Doucet, J.F.G. de Freitas, K. Murphy and S. Russell. A Simple Tutorial on Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks. This detailed discussion of the ABC network should complement the UAI2000 paper .

After downloading the file, type "tar -xf demorbpfdbn.tar" to uncompress it. This creates the directory webalgorithm containing the required m files. Go to this directory, load matlab5 and type "dbnrbpf" for the demo.
Click here for program code

## Unscented Particle Filter
In these demos, we demonstrate the use of the extended Kalman filter (EKF), unscented Kalman filter (UKF), standard particle filter (a.k.a. condensation, survival of the fittest, bootstrap filter, SIR, sequential Monte Carlo, etc.), particle filter with MCMC steps, particle filter with EKF proposal and unscented particle filter (particle filter with UKF proposal) on a simple state estimation problem and on a financial time series forecasting problem. The algorithms are coded in a way that makes it trivial to apply them to other problems. Several generic routines for resampling are provided.

The derivation and details are presented in: R van der Merwe, A Doucet, JFG de Freitas and E Wan. The Unscented Particle Filter. Technical report CUED/F-INFENG/TR 380, Cambridge University Department of Engineering, May 2000.

After downloading the file, type "tar -xf upf_demos.tar" to uncompress it. This creates the directory webalgorithm containing the required m files. Go to this directory, load matlab5 and type "demo_MC" for the demo.
Click here for program code

## EM for neural networks
In this demo, I use the EM algorithm with a Rauch-Tung-Striebel smoother and an M step, which I've recently derived, to train a two-layer perceptron, so as to classify medical data (kindly provided by Steve Roberts and Will Penny from EE, Imperial College). The data and simulations are described in: J.F.G. de Freitas, M. Niranjan and A.H. Gee Nonlinear State Space Estimation with Neural Networks and the EM algorithm

After downloading the file, type "tar -xf EMdemo.tar" to uncompress it. This creates the directory EMdemo containing the required m files. Go to this directory, load matlab5 and type "EMtremor". The figures will then show you the simulation results, including ROC curves, likelihood plots, decision boundaries with error bars, etc. WARNING: Do make sure that you monitor the log-likelihood and check that it is increasing. Due to numerical errors, it might show glitches for some data sets.
Click here for program code

## On-Line MCMC Bayesian Model Selection
This demo demonstrates how to use the sequential Monte Carlo algorithm with reversible jump MCMC steps to perform model selection in neural networks. We treat both the model dimension (number of neurons) and model parameters as unknowns. The derivation and details are presented in: C. Andrieu, JFG de Freitas and A. Doucet. Sequential Bayesian Estimation and Model Selection Applied to Neural Networks . Technical report CUED/F-INFENG/TR 341, Cambridge University Department of Engineering, June 1999.

After downloading the file, type "tar -xf version2.tar" to uncompress it. This creates the directory version2 containing the required m files. Go to this directory, load matlab5 and type "smcdemo1". In the header of the demo file, one can select to monitor the simulation progress (with par.doPlot=1) and modify the simulation parameters.
Click here for program code

## Reversible Jump MCMC Bayesian Model Selection
This demo demonstrates the use of the reversible jump MCMC algorithm for neural networks. It uses a hierarchical full Bayesian model for neural networks. This model treats the model dimension (number of neurons), model parameters, regularisation parameters and noise parameters as random variables that need to be estimated. The derivations and proof of geometric convergence are presented, in detail, in: C. Andrieu, JFG de Freitas and A. Doucet. Robust Full Bayesian Learning for Neural Networks. Technical report CUED/F-INFENG/TR 343, Cambridge University Department of Engineering, May 1999.

After downloading the file, type "tar -xf rjMCMC.tar" to uncompress it. This creates the directory rjMCMC containing the required m files. Go to this directory, load matlab5 and type "rjdemo1". In the header of the demo file, one can select to monitor the simulation progress (with par.doPlot=1) and modify the simulation parameters.
Click here for program code

## Reversible Jump MCMC Simulated Annealing
This demo demonstrates the use of the reversible jump MCMC simulated annealing for neural networks. This algorithm enables us to maximise the joint posterior distribution of the network parameters and the number of basis function. It performs a global search in the joint space of the parameters and number of parameters, thereby surmounting the problem of local minima. It allows the user to choose among various model selection criteria, including AIC, BIC and MDL. The derivations and proof of convergence are presented, in detail, in: C. Andrieu, JFG de Freitas and A. Doucet. Robust Full Bayesian Learning for Neural Networks. Technical report CUED/F-INFENG/TR 343, Cambridge University Department of Engineering, May 1999.

After downloading the file, type "tar -xf rjMCMCsa.tar" to uncompress it. This creates the directory rjMCMCsa containing the required m files. Go to this directory, load matlab5 and type "rjdemo1sa". In the header of the demo file, one can select to monitor the simulation progress (with par.doPlot=1), modify the simulation parameters and select the model selection criterion.
Click here for program code

## Sequential Sampling-Importance Resampling (SIR)
The aim of this demo is to show you how to implement a generic SIR (a.k.a. particle, bootstrap, Monte Carlo) filter to estimate the hidden states of a nonlinear, non-Gaussian state space model. I've avoided fancy code and kept it very simple so that it can be easily modified. A good reference on this problem is Neil Gordon's "Novel Approach to Nonlinear/Non-Gaussian Bayesian State Estimation", IEE Proceedings-F, Vol 140, No 2, pp 107-113, 1993.

After downloading the file, type "tar -xf demo1.tar" to uncompress it. This creates the directory demo1 containing the required m files. Go to this directory, load matlab5 and type "sirdemo1". Figure 1 will then show you the simulation progress.
Click here for program code

## Hybrid SIR to train neural networks
In this simple demo, I use the hybrid SIR and EKF algorithms to train a two-layer feed-forward neural network. The data is generated from a nonlinear, non-stationary state space model. A good reference on this topic is: J.F.G. de Freitas, M. Niranjan, A.H. Gee and A. Doucet. Sequential Monte Carlo Methods for Optimisation of Neural Network Models. Technical report CUED/F-INFENG/TR 328, Cambridge University Department of Engineering, July 1998.

After downloading the file, type "tar -xf demo2.tar" to uncompress it. This creates the directory demo2 containing the required m files. Go to this directory, load matlab5 and type "sirdemo2". Figure 1 will then show you the simulation progress.
Click here for program code

## MLP-EKF sequential evidence maximisation
In this demo, I use the EKF and EKF with sequential evidence maximistion to train a neural network with data generated from a nonlinear, non-stationary state space model. The sequential evidence maximisation framework allows us to adapt the process noise covariances. This is done by matching the innovations ensemble covariance to the covariance over time so as to make the one-step-ahead predictions become white (i.e. all the information in the data is absorbed by the model). All the derivations are presented, in detail, in: J.F.G. de Freitas, M. Niranjan and A.H. Gee. Hierarchical Bayesian-Kalman models for regularisation and ARD in sequential learning. Technical report CUED/F-INFENG/TR 307, Cambridge University Department of Engineering, December 1997.

After downloading the file, type "tar -xf demo3.tar" to uncompress it. This creates the directory demo3 containing the required m files. Go to this directory, load matlab5 and type "ekfdemo1". Figure 1 will then show you the simulation results.
Click here for program code

## MLP-EKF with noisy, chaotic data
In this demo, I use the EKF and EKF with sequential evidence maximistion to train a neural network with data generated from a nonlinear, time-varying, chaotic and noisy mapping. This example is discussed in: J.F.G. de Freitas, M. Niranjan and A.H. Gee. Hierarchical Bayesian-Kalman models for regularisation and ARD in sequential learning. Technical report CUED/F-INFENG/TR 307, Cambridge University Department of Engineering, December 1997.

After downloading the file, type "tar -xf demo7.tar" to uncompress it. This creates the directory demo7 containing the required m files. Go to this directory, load matlab5 and type "ekfdemo7". Figure 1 will then show you the simulation results.
Click here for program code

## Pricing call options with EKFs and NNs
In this demo, I use the EKF and EKF with sequential evidence maximistion to train a two-layer perceptron, so as to track financial call options data on the FTSE-100 index (1994). The data and simulations are described in: J.F.G. de Freitas, M. Niranjan, A.H. Gee and A. Doucet. Sequential Monte Carlo Methods for Optimisation of Neural Network Models. Technical report CUED/F-INFENG/TR 328, Cambridge University Department of Engineering, July 1998.

After downloading the file, type "tar -xf demo5.tar" to uncompress it. This creates the directory demo5 containing the required m files. Go to this directory, load matlab5 and type "ekfdemo2". Figure 1 will then show you the simulation results.
Click here for program code 
