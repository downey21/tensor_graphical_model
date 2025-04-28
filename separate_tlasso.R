
# -*- coding: utf-8 -*-

# [Min et al. 2022]
# Min, Keqian, Qing Mai, and Xin Zhang. "Fast and separable estimation in high-dimensional tensor Gaussian graphical models." *Journal of Computational and Graphical Statistics* 31.1 (2022): 294-300.

# Section 4. Simulations in [Min et al. 2022]
# When the dimension is unbalanced, the tlasso estimator performs much worse than the proposed separate method.
# The cyclically iterative algorithm cannot improve the estimation accuracy in the iteration for partially sparse models.
# This confirms our concern that violation of sparsity assumption hurts the performance of iterative methods.
# It also indicates that the performance of the proposed method is more stable and consistent.
# The proposed method also shows superior performance in terms of support recovery.

# [Lyu et al. 2019]
# Lyu, Xiang, et al. "Tensor graphical model: Non-convex optimization and statistical inference." *IEEE transactions on pattern analysis and machine intelligence* 42.8 (2019): 2024-2037.

rm(list = ls())

# install.packages("Tlasso")
# install.packages("rTensor")
# install.packages("tensr")
# install.packages("expm")
# install.packages("glasso")
# install.packages("doParallel")

library(Tlasso)
library(rTensor)
library(tensr)
library(expm)
library(glasso)
library(doParallel)

source("/root/Project/tensor_graphical_model/separate/separate_original/Separate.fit.R")
source("/root/Project/tensor_graphical_model/separate/separate_original/cv.Separate.R")

source("/root/Project/tensor_graphical_model/separate/separate_original/Model7.R")
source("/root/Project/tensor_graphical_model/separate/separate_original/simulation.summary.R")

# Generate training set and validation set
n <- 20

# Model setting
dimen <- c(30, 36, 30) # dimension of X
nvars <- prod(dimen) # number of variables
K <- 3 # order of X

# set-up of precision matrices
Omega <- array(list(), length(dimen)) # precision matrix
for (i in 1:length(dimen)) {
    # Triangle (TR) covariance
    Omega[[i]] <- Tlasso::ChainOmega(dimen[i], sd = i * 100, norm.type = 2) # sd: seed number
}

# check
sum(Omega[[1]]^2) # 1 if norm.type = 2
Omega[[1]][1, 1] # 1 if norm.type = 1

data <- Model7(n, seed = 123456)

x <- data$x # training set
vax <- data$vax # validation set

# standardization
x_cs <- x
str(x_cs)

meantensor <- apply(x_cs, c(1, 2, 3), mean)
# for (i in 1:n) {
#     x_cs[, , , i] <- x_cs[, , , i] - meantensor
# }
x_cs <- sweep(x_cs, c(1, 2, 3), meantensor, FUN = "-")

# vec_x <- matrix(0, nrow = nvars, ncol = n)
# for (i in 1:n) {
#     vec_x[, i] <- matrix(x_cs[, , , i], nrow = nvars, ncol = 1)
# }
# vec_x <- t(vec_x)
vec_x <- matrix(x_cs, nrow = nvars, ncol = n)  # nvars × n
vec_x <- t(vec_x)                              # n × nvars

sd <- apply(vec_x, 2, sd)
sd[sd == 0] <- 1
vec_x <- scale(vec_x, center = FALSE, scale = sd)

for (i in 1:n) {
    x_cs[, , , i] <- array(vec_x[i, ], dim = dimen)
}

# proper candidates of tuning parameters
# lamseq <- seq(0.001, 0.2, length.out = 50)
lamseq <- seq(0.015, 0.1, length.out = 10)
lambda.list <- list() # a list containing candidates of tuning parameters for each mode
for (i in 1:K) {
    lambda.list[[i]] <- lamseq
}

# separate

# Separate.fit

# x: p1*p2*...*pM*n
# val: (Optional) validation set. If supplied, lambda.list should be provided
# lambda.vec: the sequence of regularization parameters for each mode in est.mode. It is used when val is missing.
# lambda.list: A list of regularization parameters that provides a lambda sequence for each mode in `est.mode`. Must be supplied with validation set. When a validation set is supplied, the optimal
# tuning parameters will be chosen from lambda.list based on the log-likelihood calculated using validation set.
# maxit: maximum number of iterations for fitting glasso. Default value is 10,000
# njobs: number of nodes used to do parallel computing

# cv.Separate

# nfolds: number of folds. Default is 5
# foldid: (Optional) a vector of values between 1 and nfolds identifying what fold each observation is in. If supplied, nfolds can be missing.

fit_separate <- Separate.fit(x, vax, lambda.list = lambda.list, maxit = 1e4, njobs = 4)

fit_separate_cv <- cv.Separate(x, lambda.list = lambda.list, maxit = 1e4, njobs = 4, nfolds = 5)

names(fit_separate)
str(fit_separate)

names(fit_separate_cv)
str(fit_separate_cv)

# Simulation summary of estimation errors, TPR and TNR
out <- simulation.summary(fit_separate$Omegahat, Omega, offdiag = FALSE) # simulation.summary is same to the est.analysis() function in the Tlasso package except for kro parts.

out$av.error.f # averaged estimation error in Frobenius norm
out$av.error.max # averaged estimation error in Maximum norm
out$av.tpr # averaged true positive rate
out$av.tnr # averaged true negative rate
out$error.f # estimation error in Frobenius norm for each mode
out$error.max # estimation error in Maximum norm for each mode
out$tpr # true positive rate for each mode
out$tnr # true negative rate for each mode

out_cv <- simulation.summary(fit_separate_cv$Omegahat, Omega, offdiag = FALSE)

out_cv$av.error.f # averaged estimation error in Frobenius norm
out_cv$av.error.max # averaged estimation error in Maximum norm
out_cv$av.tpr # averaged true positive rate
out_cv$av.tnr # averaged true negative rate
out_cv$error.f # estimation error in Frobenius norm for each mode
out_cv$error.max # estimation error in Maximum norm for each mode
out_cv$tpr # true positive rate for each mode
out_cv$tnr # true negative rate for each mode

# tlasso

# A sequence of candidates of tuning parameter. The form of lam1 is suggested by the Tlasso method.
eta <- seq(40, 90, length.out = 8)
eta.length <- length(eta)
lam1 <- c(
    sqrt(log(dimen[1]) / (n * prod(dimen) * dimen[1])),
    sqrt(log(dimen[2]) / (n * prod(dimen) * dimen[2])),
    sqrt(log(dimen[3]) / (n * prod(dimen) * dimen[3]))
)

loglik <- rep(0, eta.length)
for (c in 1:eta.length) {
    lambda <- eta[c] * lam1
    fit <- Tlasso::Tlasso.fit(x, lambda.vec = lambda, T = 10, norm.type = 2)

    ## Calculate log-likelihood using the estimations from Tlasso.fit
    Omega.list.sqrt <- fit # square root of \hat\Omega
    logdet.fit <- rep(0, K) # log-determinant of \hat\Omega
    for (j in K:1) {
        Omega.list.sqrt[[j]] <- expm::sqrtm(fit[[j]])
        logdet.fit[j] <- log(det(fit[[j]]) * 1e30) * prod(dimen) / dimen[j]
    }

    # Calculate \tilde S using the validation set and the estimated value of precision matrices
    S.array <- array(0, c(dimen[1], dimen[1], n))
    for (i in 1:n) {
        d <- 0
        # assign the ith observation in validation set to d
        eval(parse(text = paste("d=vax[", paste(rep(",", K), collapse = ""), "i]")))
        Vi <- rTensor::k_unfold(rTensor::as.tensor(rTensor::ttl(rTensor::as.tensor(d), Omega.list.sqrt, ms = 1:K)@data), m = 1)@data
        S.array[, , i] <- Vi %*% t(Vi)
    }
    S.mat <- apply(S.array, c(1, 2), mean) 
    loglik[c] <- sum(logdet.fit) - tensr::tr(S.mat)
}

# optimal value chosen from eta
eta.best <- eta[which.max(loglik)]

## Model fitting
fit_tlasso <- Tlasso::Tlasso.fit(x, lambda.vec = eta.best * lam1, T = 10, norm.type = 2)

out_tlasso <- simulation.summary(fit_tlasso, Omega, offdiag = FALSE)

out_tlasso$av.error.f # averaged estimation error in Frobenius norm
out_tlasso$av.error.max # averaged estimation error in Maximum norm
out_tlasso$av.tpr # averaged true positive rate
out_tlasso$av.tnr # averaged true negative rate
out_tlasso$error.f # estimation error in Frobenius norm for each mode
out_tlasso$error.max # estimation error in Maximum norm for each mode
out_tlasso$tpr # true positive rate for each mode
out_tlasso$tnr # true negative rate for each mode
