#### This function estimates the precision matrices in the tensor graphical model by choosing tuning parameters using cross-validation.  
# x: p1*p2*...*pM*n
# est.mode: index set of precision matrices to be estimated. If not specified, all precision matrices will be estimated. Default is c(1,...,M).
# lambda.list: a list of regularization parameters that provides a lambda sequence for each mode in `est.mode` for cross validation. 
# Omegatilde.list: (Optional) a list of M matrices
# nfolds: number of folds. Default is 5
# foldid: (Optional) a vector of values between 1 and nfolds identifying what fold each observation is in. If supplied, nfolds can be missing.
# scale.vec: constants to scale the log-likelihood to avoid infinite value. Default is 1 for all modes
# normalize: indicates whether $\widetilde{\boldsymbol{\Omega}}_m$ should be normalized to have unit Frobenius norm. Default is TRUE.
# thres: threshold for convergence. Default value is 1e-4
# maxit: maximum number of iterations for fitting glasso. Default 10,000
# njobs: number of nodes used to do parallel computing

cv.Separate = function(x, est.mode = NULL, lambda.list = NULL, Omegatilde.list = NULL, scale.vec = NULL, normalize = TRUE, nfolds = 5, foldid = NULL, thres = 1.0e-4, maxit = 1e4, njobs = 4) {

  if (!(is.null(est.mode) | (length(lambda.list) == length(est.mode)))) {
    stop("the length of lambda.list should be the same as the length of est.mode")
  }
  dimen = dim(x) # dimension of x
  K = length(dimen) - 1 # order of tensor
  n = dimen[K + 1] # sample size

  if (is.null(est.mode) == TRUE) {
    est.mode = c(1:K)
  }
  if (!(is.null(Omegatilde.list) | length(Omegatilde.list) == K)) {
    stop("argument Omegatilde.list should be a list of M matrices")
  }

  if (is.null(scale.vec)) {
    scale.vec = rep(1, length(est.mode))
  }
  if (nfolds < 3) {
    stop("nfolds must be bigger than 3; nfolds=5 recommended")
  }
  if (nfolds > n) {
    stop("The number of folds should be smaller than the sample size.")
  }

  ##### Cross-validation #####
  if (is.null(foldid)) {
    foldid = sample(rep(seq(nfolds), length = n))
  }
  # record log-likelihood for each mode and each lambda in the lambda.list
  loglik = list() # log-likelihood
  for (i in 1:length(lambda.list)) {
    loglik[[i]] = matrix(0, ncol = length(lambda.list[[i]]), nrow = nfolds)
  }
  for (i in seq(nfolds)) {
    which = foldid == i
    # divide dataset into training set and validation set
    eval(parse(text = paste("xtr=x[", paste(rep(",", K), collapse = ""), "!which]")))
    eval(parse(text = paste("xval=x[", paste(rep(",", K), collapse = ""), "which]")))
    fit = Separate.fit(x = xtr, val = xval, est.mode = est.mode, lambda.list = lambda.list, Omegatilde.list = Omegatilde.list, scale.vec = scale.vec, normalize = normalize, thres = 1.0e-4, maxit = 1e4, njobs = 4)
    for (j in 1:length(lambda.list)) {
      loglik[[j]][i,] = fit$loglik[[j]]
    }
  }
  
  # Calculate the averaged log-likelihoods and stardard errors across folds
  loglik.se = list()
  loglik.mean = list()
  for (i in 1:length(lambda.list)) {
    loglik.se[[i]] = apply(loglik[[i]],2,sd)/sqrt(nfolds)
    loglik.mean[[i]] = colMeans(loglik[[i]])
  }
  
  # optimal lambda corresponding to the maximum log-likelihood
  lam.best = rep(0, length(lambda.list)) 
  for (i in 1:length(lambda.list)) {
    ind = which.max(loglik.mean[[i]])
    lam.best[i] = lambda.list[[i]][ind]
  }
  
  ##### Model fitting #####
  # fit model with the best lambdas
  fit = Separate.fit(x = x, val = NULL, est.mode = est.mode, lambda.vec = lam.best, Omegatilde.list = Omegatilde.list, scale.vec = scale.vec, normalize = normalize, thres = 1.0e-4, maxit = 1e4, njobs = 4)
  
  # output
  result = list()
  result$Omegahat = fit$Omegahat
  result$lambda = lam.best
  result$loglik = loglik.mean
  result$loglik.se = loglik.se
  
  return(result)
}
