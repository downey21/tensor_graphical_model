# this code file implements the proposed method for the alcoholic and nonalcoholic group in EEG dataset

# EEG_dataset.RData contains the EEG dataset.
# It contains X.mat with size 256 * 64 * 122 and the label y.vec with size 1 * 122.
# The label y = 1 indicates the individual is from the alcoholic group and
# y = 0 indicates the individual is from the nonalcoholic group.

# EEG_Separate.R implements the proposed method on the alcoholic and nonalcoholic groups in EEG dataset.

# EEG_Cyclic.R implements the cyclic method using Tlasso package on the alcoholic and nonalcoholic groups in EEG dataset.

rm(list = ls())
library(tensr)
library(expm)
library(rTensor)
library(glasso)
library(doParallel)

load("EEG_dataset.RData")
source("Separate.fit.R")
source("cv.Separate.R")
source("simulation.summary.R")

x <- eegdata$X.mat # dataset
dimen <- dim(x)[1:2] # dimension of x
n <- dim(x)[3] # sample size
K <- 2 # order of tensor

# Downsize x to 64*64
xtemp <- x
x <- array(0, dim = c(64, 64, n))
for (i in 1:n) {
  xtemp[, , i] <- eegdata$X.mat[, , i]
  x[, , i] <- matrix(0, nrow = 64, ncol = 64)
  for (j in 1:64) {
    for (s in 1:64) {
      x[j, s, i] <- sum(xtemp[(j * 4 - 3):(j * 4), s, i]) / 4 # calculate average
    }
  }
}
dimen <- dim(x[, , 1]) # dimension of x after preprocessing
nvars <- prod(dimen) # number of variables

# Group1: nonalcoholic
n1 <- 45
x1 <- x[, , 78:122]

# standardization
meantensor <- apply(x1, c(1, 2), mean)
for (i in 1:n1) {
  x1[, , i] <- x1[, , i] - meantensor
}

vec_x1 <- matrix(0, nrow = nvars, ncol = n1)
for (i in 1:n1) {
  vec_x1[, i] <- matrix(x1[, , i], nrow = nvars, ncol = 1)
}
vec_x1 <- t(vec_x1)

sd <- apply(vec_x1, 2, sd)
sd[sd == 0] <- 1
vec_x1 <- scale(vec_x1, center = FALSE, scale = sd)

for (i in 1:n1) {
  x1[, , i] <- array(vec_x1[i, ], dim = dimen)
}


# Group2:alcoholic
n2 <- 77
x2 <- x[, , 1:77]

# standardization
meantensor <- apply(x2, c(1, 2), mean)
for (i in 1:n2) {
  x2[, , i] <- x2[, , i] - meantensor
}

vec_x2 <- matrix(0, nrow = nvars, ncol = n2)
for (i in 1:n2) {
  vec_x2[, i] <- matrix(x2[, , i], nrow = nvars, ncol = 1)
}
vec_x2 <- t(vec_x2)

sd <- apply(vec_x2, 2, sd)
sd[sd == 0] <- 1
vec_x2 <- scale(vec_x2, center = FALSE, scale = sd)

for (i in 1:n2) {
  x2[, , i] <- array(vec_x2[i, ], dim = dimen)
}



# Group1: nonalcoholic

#### the tuning process uses the 5-fold cross-validation. 
lambda.list=list()
lambda.list[[1]]=seq(0.001,0.2,length.out=50)
lambda.list[[2]]=seq(0.001,0.1,length.out=50)
# nfolds = 5
foldid = c(5, 1, 1, 4, 3, 5, 1, 1, 4, 1, 3, 5, 3, 5, 4, 5, 5, 2, 4, 3, 4, 2, 1, 1, 2, 3, 4, 3, 1, 3, 2, 4, 5, 2, 
           4, 2, 3, 4, 2, 5, 2, 3, 5, 2, 1)
fit <- cv.Separate(x1, lambda.list = lambda.list, foldid = foldid, normalize = FALSE)

#### Since the variation of loglik across folds is quite large, instead of using the 1 SE rule, we choose the largest lambda
#### such that  the corresponding loglik is within 1/nfolds (1/5) SE of the maximum loglik
loglik <- fit$loglik
loglik.se <- fit$loglik.se # standard error of loglik
lambda.vec <- rep(0, K)
for (i in 1:K){
  target = max(loglik[[i]]) - loglik.se[[i]][which.max(loglik[[i]])]/5
  ind2 = which(loglik[[i]] >= target)  # get indices of lambda that satisfies the requirement
  ind2 = ind2[length(ind2)]    # get the largest index
  lambda.vec[i] = lambda.list[[i]][ind2]
}

# fit model
fit2 <- Separate.fit(x1, lambda.vec = lambda.vec, normalize = FALSE)
Omegahat_nonalco <- fit2$Omegahat



# Group2: alcoholic

lambda.list=list()
lambda.list[[1]]=seq(0.001,0.1,length.out=50)
lambda.list[[2]]=seq(0.001,0.1,length.out=50)
# nfolds = 5
foldid = c(4, 5, 4, 4, 4, 3, 5, 3, 5, 5, 5, 4, 1, 2, 1, 3, 2, 3, 1, 5, 1, 3, 4, 4, 5, 1, 2, 1, 3, 3, 5, 2, 4, 1, 1, 3, 3, 2, 3, 
           5, 4, 4, 5, 3, 2, 3, 2, 5, 2, 3, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 5, 5, 4, 3, 5, 4, 2, 3, 1, 1, 5, 2, 4, 1, 4, 2, 4)
fit3 <- cv.Separate(x2, lambda.list = lambda.list, foldid = foldid, normalize = FALSE)

####  we choose the largest lambda such that the corresponding loglik is within 1/nfolds(1/5) SE of the maximum loglik
loglik <- fit3$loglik
loglik.se <- fit3$loglik.se # standard error of loglik
lambda.vec <- rep(0, K)
for (i in 1:K){
  target = max(loglik[[i]]) - loglik.se[[i]][which.max(loglik[[i]])]/5
  ind2 = which(loglik[[i]] >= target) # get indices of lambda that satisfies the requirement
  ind2 = ind2[length(ind2)]  # get the largest index
  lambda.vec[i] = lambda.list[[i]][ind2]
}

# fit model
fit4 <- Separate.fit(x2, lambda.vec = lambda.vec, normalize = FALSE)
Omegahat_alco <- fit4$Omegahat

