# this code generates data observations for Model 7
Model7 <- function(n, seed) {

# seed: random seed
# n: sample size
  
# Model setting
dimen <- c(30, 36, 30) # dimension of X
nvars <- prod(dimen) # number of variables
K <- 3 # order of X

# set-up of precision matrices
Sigma <- array(list(), length(dimen)) # covariance matrix
Omega <- array(list(), length(dimen)) # precision matrix
dSigma <- array(list(), length(dimen)) # square root of covariance matrix

for (i in 1:length(dimen)) {
  Omega[[i]] <- ChainOmega(dimen[i], sd = i * 100, norm.type = 2) #Triangle (TR) covariance
  Sigma[[i]] <- solve(Omega[[i]])
  dSigma[[i]] <- t(chol(Sigma[[i]]))
}

set.seed(seed) 

# Generate data observation
# training set
vec_x <- matrix(rnorm(nvars * n), ncol = n) 
x <- array(0, dim = c(dimen, n))
for (i in 1:n) {
  x[, , , i] <- array(vec_x[, i], dimen)
  x[, , , i] <- atrans(x[, , , i], dSigma)
}

# validation set
vec_vax <- matrix(rnorm(nvars * n), ncol = n) 
vax <- array(0, dim = c(dimen, n))
for (i in 1:n) {
  vax[, , , i] <- array(vec_vax[, i], dimen)
  vax[, , , i] <- atrans(vax[, , , i], dSigma)
}

result <- list()
result$x <- x
result$vax <- vax

return(result)

}


