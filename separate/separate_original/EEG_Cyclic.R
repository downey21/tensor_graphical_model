# this code file obtains the 'cyclic' estimators for the alcoholic and nonalcoholic group in EEG dataset using Tlasso package

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
library(doParallel)
library(Tlasso)

load("EEG_dataset.RData")

x <- eegdata$X.mat # dataset
dimen <- dim(x)[1:2] # dimension of x
n <- dim(x)[3] # sample size
K <- 2 # order of tensor

# Downsize dataset to 64*64
xtemp <- x
x <- array(0, dim = c(64, 64, 122))
for (i in 1:n) {
  xtemp[, , i] <- eegdata$X.mat[, , i]
  x[, , i] <- matrix(0, nrow = 64, ncol = 64)
  for (j in 1:64) {
    for (s in 1:64) {
      x[j, s, i] <- sum(xtemp[(j * 4 - 3):(j * 4), s, i]) / 4 # calculate averages
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
x0 <- x1
n0 <- n1

# fit model
lambda.vec <- c(
  sqrt(log(dimen[1]) / (n0 * prod(dimen) * dimen[1])),
  sqrt(log(dimen[2]) / (n0 * prod(dimen) * dimen[2]))
)
fit <- Tlasso.fit(x0, lambda.vec = 1.00 * lambda.vec, T = 20, norm.type = 1)
G1_Omegahat1 <- fit[[1]]
G1_Omegahat2 <- fit[[2]]


# Group2: alcoholic
x0 <- x2
n0 <- n2

# fit model
lambda.vec <- c(
  sqrt(log(dimen[1]) / (n0 * prod(dimen) * dimen[1])),
  sqrt(log(dimen[2]) / (n0 * prod(dimen) * dimen[2]))
)
fit2 <- Tlasso.fit(x0, lambda.vec = 1.00 * lambda.vec, T = 20, norm.type = 1)
G2_Omegahat1 <- fit2[[1]]
G2_Omegahat2 <- fit2[[2]]


