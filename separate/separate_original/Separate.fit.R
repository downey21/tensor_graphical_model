#### This function estimates the precision matrices in the tensor graphical model using the proposed parallel scheme.
# x: p1*p2*...*pM*n
# val: (Optional) validation set. If supplied, lambda.list should be provided
# est.mode: index set of precision matrices to be estimated. If not specified, all precision matrices will be estimated
# lambda.vec: the sequence of regularization parameters for each mode in est.mode. It is used when val is missing.
# lambda.list: A list of regularization parameters that provides a lambda sequence for each mode in `est.mode`. Must be supplied with validation set. When a validation set is supplied, the optimal
# tuning parameters will be chosen from lambda.list based on the log-likelihood calculated using validation set.
# Omegatilde.list: (Optional) a list of M matrices
# scale.vec: constants to scale the log-likelihood to avoid infinite value. Default is 1 for all modes
# normalize: indicates whether $\widetilde{\boldsymbol{\Omega}}_m$ and $\widehat{\boldsymbol{\Omega}}_m$ should be normalized to have unit Frobenius norm. Default is TRUE.
# thres: threshold for convergence. Default value is 1e-4
# maxit: maximum number of iterations for fitting glasso. Default value is 10,000
# njobs: number of nodes used to do parallel computing


Separate.fit = function(x, val = NULL, est.mode = NULL, lambda.vec = NULL, lambda.list = NULL, Omegatilde.list = NULL, scale.vec = NULL, normalize = TRUE, thres = 1.0e-4, maxit = 1e4, njobs = 4) {
  
  if (is.null(est.mode) == TRUE) {
    est.mode = c(1:K)
  }
  if (is.null(val)) {
    if (is.null(lambda.vec) | length(lambda.vec) != length(est.mode)) {
      stop("lambda.vec is missing or does not have the correct length")
    }
  } else {
    if (is.null(lambda.list) | length(lambda.list) != length(est.mode)) {
      stop("lambda.list is missing or does not have the correct length")
    }
  }

  dimen = dim(x) # dimension of dataset
  K = length(dimen) - 1 # order of tensor
  n = dimen[K + 1] # sample size of training set
  n_val = dim(val)[K + 1] # sample size of validation set
  nvars = prod(dimen) # number of variables
  m.vec = dimen[1:K] # dimension of each observation

  if (!(is.null(Omegatilde.list) | length(Omegatilde.list) == K)) {
    stop("argument Omegatilde.list should be a list of M matrices")
  }

  if (is.null(scale.vec)) {
    scale.vec = rep(1, length(est.mode))
  }

  ##### Calculate \tilde\Omega #####
  if (is.null(Omegatilde.list) == FALSE) {
    fit1 = Omegatilde.list  # user-specified value for \tilde\Omega
  }
  else {
    # Calculate \tilde\Omega by the definition in the paper
    c1 = makeCluster(njobs)
    registerDoParallel(c1)
    fit1 = foreach(k = 1:K, .export = c("x"), .combine = list, .multicombine = TRUE) %dopar% {
      # when sample size is small, use the identity matrix
      if (n * nvars < ((dimen[k]**2) * (dimen[k] - 1) / 2)) {
        Omega_tilde = diag(dimen[k])
      }
      else {
        # when sample size is large, calculate the sample estimator of the precision matrices
        S.array = array(0, c(m.vec[k], m.vec[k], n))
        for (i in 1:n) {
          d = 0
          eval(parse(text = paste("d=x[", paste(rep(",", K), collapse = ""), "i]"))) # assign the ith observation to d
          Vi = rTensor::k_unfold(rTensor::as.tensor(d), m = k)@data  # unfold tensor
          S.array[, , i] = Vi %*% t(Vi)
        }
        S.mat = apply(S.array, c(1, 2), mean) * m.vec[k] / prod(m.vec) # sample estimation of \Sigma_k
        Omega_tilde = solve(S.mat)
        # normalization
        if (normalize) {
        Omega_tilde = Omega_tilde / norm(Omega_tilde, type = "F")}
      }
      Omega_tilde
    }
    stopCluster(c1)
  }

  K1 = length(est.mode) # number of precision matrices to be estimated
  lam.best = rep(0, K1)
  loglik = list() 

  ###### Tuning process ######
  # When validation set is supplied, the lambdas with the maximum log-likelihood will be chosen
  if (!(is.null(val))) {
    Omega.list = list() # list of \tilde\Omega
    Omega.list.sqrt = list() # list of square root of \tilde\Omega
    for (k in 1:K) {
      Omega.list[[k]] = fit1[[k]]
      Omega.list.sqrt[[k]] = sqrtm(Omega.list[[k]])
    }
    Omega.sqrt.copy = Omega.list.sqrt

    for (mode_index in 1:K1) {
      k = est.mode[mode_index]
      
      # Calculate \tilde S_k using the training set
      S.array = array(0, c(m.vec[k], m.vec[k], n))
      Omega.list.sqrt[[k]] = diag(m.vec[k]) # set \tilde\Omega_k to identity matrix
      for (i in 1:n) {
        d = 0
        eval(parse(text = paste("d=x[", paste(rep(",", K), collapse = ""), "i]"))) # assign the ith observation to d
        Vi = k_unfold(as.tensor(ttl(as.tensor(d), Omega.list.sqrt,
          ms = 1:K
        )@data), m = k)@data
        S.array[, , i] = Vi %*% t(Vi)
      }
      S.mat = apply(S.array, c(1, 2), mean) * m.vec[k] / prod(m.vec) # \tilde S_k

      # Calculate \tilde S_k using the validation set
      testS.array = array(0, c(m.vec[k], m.vec[k], n_val))
      for (i in 1:n_val) {
        d = 0
        eval(parse(text = paste("d=val[", paste(rep(",", K), collapse = ""), "i]")))
        Vi = k_unfold(as.tensor(ttl(as.tensor(d), Omega.list.sqrt,
          ms = 1:K
        )@data), m = k)@data
        testS.array[, , i] = Vi %*% t(Vi)
      }
      testS.mat = apply(testS.array, c(1, 2), mean) * m.vec[k] / prod(m.vec) # \tilde S_k
      Omega.list.sqrt[[k]] = Omega.sqrt.copy[[k]]

      # fit model with a sequence of lambdas
      lamk = lambda.list[[mode_index]] # a sequence of candidates for lambda_k
      lam.length = length(lamk)
      loglik2 = rep(0, lam.length)
      for (i in 1:lam.length) {
        Out1 = glasso(S.mat, rho = lamk[i], penalize.diagonal = FALSE, maxit = 1e4, thr = 1.0e-4)
        hat_Omega = Out1$wi
        loglik2[i] = -tr(testS.mat %*% hat_Omega) + log(det(hat_Omega * scale.vec[mode_index]))
        if (loglik2[i] == Inf) {
          stop(paste("Infinite value! Please choose a smaller scale for mode", mode_index))
        }
        if (loglik2[i] == -Inf) {
          stop(paste("Negative infinite value! Please choose a larger scale for mode", mode_index))
        }
      }
      ind = which.max(loglik2)
      lam.best[mode_index] = lamk[ind] # get the optimal lambda that maximizes the log-likelihood
      loglik[[mode_index]] = loglik2
    }
  } else {
    # if validation set is not provided, directly use lambda.vec to fit model
    lam.best = lambda.vec
  }


  ##### Model fitting using parallel computing #####
  # register cluster for parallel computing
  c1 = makeCluster(njobs)
  registerDoParallel(c1)
  K1 = length(est.mode)
  fit_result = foreach(mode_ind = 1:K1, .packages = c("glasso", "rTensor", "expm"), .export = c("x"), .combine = list, .multicombine = TRUE) %dopar% {
    k = est.mode[mode_ind]
    Omega.list.sqrt = list()
    for (i in 1:K) {
      Omega.list.sqrt[[i]] = sqrtm(fit1[[i]])
    }
    # Calculate \tilde S_k
    S.array = array(0, c(m.vec[k], m.vec[k], n))
    Omega.list.sqrt[[k]] = diag(m.vec[k])
    for (i in 1:n) {
      d = 0
      eval(parse(text = paste("d=x[", paste(rep(",", K), collapse = ""), "i]")))
      Vi = k_unfold(as.tensor(ttl(as.tensor(d), Omega.list.sqrt,
        ms = 1:K
      )@data), m = k)@data
      S.array[, , i] = Vi %*% t(Vi)
    }
    S.mat = apply(S.array, c(1, 2), mean) * m.vec[k] / prod(m.vec) # \tilde S_k
    # fit model
    Out1 = glasso(S.mat, rho = lam.best[mode_ind], penalize.diagonal = FALSE, maxit = maxit, thr = thres)
    hat_Omega = as.matrix(Out1$wi)
    # normalization
    if (normalize) {
      hat_Omega = hat_Omega / norm(hat_Omega, type = "F")
    }
    hat_Omega
  }
  stopCluster(c1)

  result = list()
  result$Omegahat = fit_result
  result$lambda = lam.best
  result$loglik = loglik
  return(result)
}
