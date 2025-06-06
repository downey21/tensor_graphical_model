# this file includes the model settings for Models 1-14.

library(Tlasso)

# model 1

# model setting
n <- 100
dimen <- c(30, 36, 30)
nvars <- prod(dimen)
# K: order of tensor
K <- 3

Sigma <- array(list(), length(dimen))
Omega <- array(list(), length(dimen))
dSigma <- array(list(), length(dimen))

# define Omega and Sigma
for (i in 1:length(dimen)) {
  Omega[[i]] <- ChainOmega(dimen[i], sd = i * 100, norm.type = 2)
  Sigma[[i]] <- solve(Omega[[i]])
  dSigma[[i]] <- t(chol(Sigma[[i]]))
}

# model 2

# model setting
n <- 100
dimen <- c(100, 100, 100)
nvars <- prod(dimen)
K <- 3

Sigma <- array(list(), length(dimen))
Omega <- array(list(), length(dimen))
dSigma <- array(list(), length(dimen))

# define Omega and Sigma
for (i in 1:length(dimen)) {
  Omega[[i]] <- ChainOmega(dimen[i], sd = i * 100, norm.type = 2)
  Sigma[[i]] <- solve(Omega[[i]])
  dSigma[[i]] <- t(chol(Sigma[[i]]))
}


# model 3

# model setting
n <- 100
dimen <- c(5, 5, 500)
nvars <- prod(dimen)
K <- 3

Sigma <- array(list(), length(dimen))
Omega <- array(list(), length(dimen))
dSigma <- array(list(), length(dimen))

for (i in 1:length(dimen)) {
  Omega[[i]] <- ChainOmega(dimen[i], sd = i * 100, norm.type = 2)
  Sigma[[i]] <- solve(Omega[[i]])
  dSigma[[i]] <- t(chol(Sigma[[i]]))
}

# model 4

# model setting
n <- 100
dimen <- c(30, 36, 30)
nvars <- prod(dimen)
K <- 3

Sigma <- array(list(), length(dimen))
Omega <- array(list(), length(dimen))
dSigma <- array(list(), length(dimen))

times <- 1:dimen[1]
rho <- 0.8
H <- abs(outer(times, times, "-"))
Omega[[1]] <- rho^H
Omega[[2]] <- ChainOmega(dimen[2], sd = 2000, norm.type = 2)
Omega[[3]] <- ChainOmega(dimen[3], sd = 3000, norm.type = 2)

for (i in 1:length(dimen)) {
  Omega[[i]] <- Omega[[i]] / fnorm(Omega[[i]])
  Sigma[[i]] <- solve(Omega[[i]])
  dSigma[[i]] <- t(chol(Sigma[[i]]))
}


# model 5

# model setting
n <- 100
dimen <- c(30, 36, 30)
nvars <- prod(dimen)
K <- 3

Sigma <- array(list(), length(dimen))
Omega <- array(list(), length(dimen))
dSigma <- array(list(), length(dimen))

rho <- 0.6
Omega[[1]] <- matrix(rho, nrow = dimen[1], ncol = dimen[1])
Omega[[2]] <- matrix(rho, nrow = dimen[2], ncol = dimen[2])

for (i in 1:dimen[1]) {
  Omega[[1]][i, i] <- 1
}
for (i in 1:dimen[2]) {
  Omega[[2]][i, i] <- 1
}
Omega[[3]] <- ChainOmega(dimen[3], sd = 300, norm.type = 2)

for (i in 1:length(dimen)) {
  Omega[[i]] <- Omega[[i]] / fnorm(Omega[[i]])
  Sigma[[i]] <- solve(Omega[[i]])
  dSigma[[i]] <- t(chol(Sigma[[i]]))
}

# model 6

# model setting
n <- 100
dimen <- c(5, 5, 500)
nvars <- prod(dimen)
K <- 3

Sigma <- array(list(), length(dimen))
Omega <- array(list(), length(dimen))
dSigma <- array(list(), length(dimen))

rho <- 0.6
Omega[[1]] <- matrix(rho, nrow = dimen[1], ncol = dimen[1])
Omega[[2]] <- matrix(rho, nrow = dimen[2], ncol = dimen[2])

for (i in 1:dimen[1]) {
  Omega[[1]][i, i] <- 1
}
for (i in 1:dimen[2]) {
  Omega[[2]][i, i] <- 1
}
Omega[[3]] <- ChainOmega(dimen[3], sd = 300, norm.type = 2)

for (i in 1:length(dimen)) {
  Omega[[i]] <- Omega[[i]] / fnorm(Omega[[i]])
  Sigma[[i]] <- solve(Omega[[i]])
  dSigma[[i]] <- t(chol(Sigma[[i]]))
}


# For Models 7-12, we just change n=100 to n=20 in Models 1-6.


# model 13

# model setting
n <- 20
dimen <- c(30, 30, 40, 40)
nvars <- prod(dimen)
# K: order of tensor
K <- 4

Sigma <- array(list(), length(dimen))
Omega <- array(list(), length(dimen))
dSigma <- array(list(), length(dimen))

for (i in 1:length(dimen)) {
  Omega[[i]] <- ChainOmega(dimen[i], sd = i * 100, norm.type = 2)
  Sigma[[i]] <- solve(Omega[[i]])
  dSigma[[i]] <- t(chol(Sigma[[i]]))
}


# model 14

# model setting
n <- 20
dimen <- c(20, 20, 20, 20, 20)
nvars <- prod(dimen)

K <- 5

Sigma <- array(list(), length(dimen))
Omega <- array(list(), length(dimen))
dSigma <- array(list(), length(dimen))

for (i in 1:length(dimen)) {
  Omega[[i]] <- ChainOmega(dimen[i], sd = i * 100, norm.type = 2)
  Sigma[[i]] <- solve(Omega[[i]])
  dSigma[[i]] <- t(chol(Sigma[[i]]))
}
