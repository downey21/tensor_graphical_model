# this code obtains the "Cyclic" estimators for Model 7 in Table 1 in the supplementary material.
rm(list = ls())
library(Tlasso)
library(tensr)
library(expm)
library(rTensor)

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
  print(run)
  
  ## Generate training set and validation set
  data <- Model7(n, run * 123456)
  x <- data$x # training set
  vax <- data$vax # validation set

  ## Tuning process
  # A sequence of candidates of tuning parameter. The form of lam1 is suggested by the Tlasso method.
  C <- seq(40, 90, length.out = 8)
  C.length <- length(C)
  lam1 <- c(
    sqrt(log(dimen[1]) / (n * prod(dimen) * dimen[1])),
    sqrt(log(dimen[2]) / (n * prod(dimen) * dimen[2])),
    sqrt(log(dimen[3]) / (n * prod(dimen) * dimen[3]))
  )
  
  loglik <- rep(0, C.length) # log-likelihood
  for (c in 1:C.length) {
    lambda <- C[c] * lam1
    fit <- Tlasso.fit(x, lambda.vec = lambda, T = 10)
    
    ## Calculate log-likelihood using the estimations from Tlasso.fit
    Omega.list.sqrt <- fit # square root of \hat\Omega
    logdet.fit <- rep(0, K) # log-determinant of \hat\Omega
    for (j in K:1) {
      Omega.list.sqrt[[j]] <- sqrtm(fit[[j]])
      logdet.fit[j] <- log(det(fit[[j]]) * 1e30) * prod(dimen) / dimen[j]
    }
    
    # Calculate \tilde S using the validation set and the estimated value of precision matrices
    S.array <- array(0, c(dimen[1], dimen[1], n))
    for (i in 1:n) {
      d <- 0
      # assign the ith observation in validation set to d
      eval(parse(text = paste("d=vax[", paste(rep(",", K), collapse = ""), "i]"))) 
      Vi <- k_unfold(as.tensor(ttl(as.tensor(d), Omega.list.sqrt,
        ms = 1:K
      )@data), m = 1)@data
      S.array[, , i] <- Vi %*% t(Vi)
    }
    S.mat <- apply(S.array, c(1, 2), mean) 
    loglik[c] <- sum(logdet.fit) - tr(S.mat)
  }
  # optimal value chosen from C
  C.best <- C[which.max(loglik)]

  ## Model fitting
  fit <- Tlasso.fit(x, lambda.vec = C.best * lam1, T = 10)
  
  ## Simulation summary of estimation errors, TPR and TNR
  out <- simulation.summary(fit, Omega, offdiag = FALSE)
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
