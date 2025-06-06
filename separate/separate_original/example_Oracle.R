# this code obtains the "Oracle" estimators for Model 7 in Table 1 in the supplementary material.
rm(list = ls())
library(Tlasso)
library(tensr)
library(glasso)
library(expm)
library(rTensor)
library(doParallel)

source("Separate.fit.R") 
source("cv.Separate.R") 
source("simulation.summary.R") 
source("Model7.R")

# Model setting
n <- 20 # sample size
dimen <- c(30, 36, 30) # dimension of tensor
nvars <- prod(dimen) # number of variables
K <- 3 # order of tensor

# set-up of precision matrices
Sigma <- array(list(), length(dimen)) # a list of covariance matrices
Omega <- array(list(), length(dimen)) # a list of precision matrices
dSigma <- array(list(), length(dimen)) # a list of square root of covariance matrices

for (i in 1:length(dimen)) {
  Omega[[i]] <- ChainOmega(dimen[i], sd = i * 100, norm.type = 2) 
  Sigma[[i]] <- solve(Omega[[i]])
  dSigma[[i]] <- t(chol(Sigma[[i]]))
}

# Number of replicates. Please set it to 100 to reproduce the simulation results.
Run <- 1
# Run <- 100

# initialize measurements
d <- 1
av.error.f <- array(0, dim = c(Run, d)) # averaged estimation error in Frobenius norm
av.error.max <- array(0, dim = c(Run, d)) # averaged estimation error in Maximum norm
av.tpr <- array(0, dim = c(Run, d)) # averaged true positive rate
av.tnr <- array(0, dim = c(Run, d)) # averaged true negative rate

d <- 3
error.f <- array(0, dim = c(Run, d)) # estimation error in Frobenius norm for each mode
error.max <- array(0, dim = c(Run, d)) # estimation error in Maximum norm for each mode
tpr <- array(0, dim = c(Run, d)) # true positive rate for each mode
tnr <- array(0, dim = c(Run, d)) # true negative rate for each mode


for (run in 1:Run) {
  # Generate training set and validation set
  data <- Model7(n, run * 123456)
  x <- data$x
  vax <- data$vax
 
  # proper candidates of tuning parameters
  lamseq <- seq(0.02, 0.09, length.out = 10)
  lambda.list <- list() # a list containing candidates of tuning parameters for each mode 
  for (i in 1:K) {
    lambda.list[[i]] <- lamseq
  }
  
  # We pass the true precision matrices to the program through Omegatilde.list and fit the model
  fit <- Separate.fit(x, vax, lambda.list = lambda.list, Omegatilde.list = Omega, scale.vec = c(1,1,1))

  # Simulation summary of estimation errors, TPR and TNR
  out <- simulation.summary(fit$Omegahat, Omega, offdiag = FALSE)
  av.error.f[run] <- out$av.error.f
  av.error.max[run] <- out$av.error.max
  av.tpr[run] <- out$av.tpr
  av.tnr[run] <- out$av.tnr

  error.f[run, ] <- out$error.f
  error.max[run, ] <- out$error.max
  tpr[run, ] <- out$tpr
  tnr[run, ] <- out$tnr
  
}

# estimation error
mean(av.error.f)
colMeans(error.f)
mean(av.error.max)
colMeans(error.max)

# TPR and TNR
mean(av.tpr)
colMeans(tpr)
mean(av.tnr)
colMeans(tnr)
