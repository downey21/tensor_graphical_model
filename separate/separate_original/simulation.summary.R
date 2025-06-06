# This function computes the estimation errors, TPR and TNR.
# Omega.hat.list: list of estimation of precision matrices
# Omega.true.list: list of true precision matrices 
# offdiag: logical; indicate if excludes diagonal when computing performance measures

simulation.summary <- function(Omega.hat.list, Omega.true.list, offdiag = TRUE) {
  
  if (!is.list(Omega.hat.list)) {
    stop("argument Omega.hat.list should be a list")
  }
  else if (!is.list(Omega.true.list)) {
    stop("argument Omega.true.list should be a list")
  }
  else if (any(!sapply(Omega.hat.list, is.matrix))) {
    stop("argument Omega.hat.list should be a list of precision matrices")
  }
  else if (any(!sapply(Omega.true.list, is.matrix))) {
    stop("argument Omega.true.list should be a list of precision matrices")
  }
  else if (length(Omega.hat.list) != length(Omega.true.list)) {
    stop("arguments Omega.hat.list and Omega.true.list should share the same length")
  }
  else if (any(!(sapply(Omega.hat.list, dim)[1, ] == sapply(
    Omega.true.list,
    dim
  )[1, ]))) {
    stop("dimension of elements in argument Omega.hat.list should match argument Omega.true.list")
  }
  else if (!is.logical(offdiag)) {
    stop("argument offdiag should be a logical TRUE or FALSE ")
  }
  
  K <- dim(as.array(Omega.hat.list))
  # estimation error in Frobenius norm and Maximum norm
  error.f <- rep(0, K) 
  error.max <- rep(0, K)
  # true positive rate and true negative rate
  tpr <- rep(0, K)
  tnr <- rep(0, K)
  
  # Calculate estimation error, TPR, TNR
  if (offdiag == FALSE) {
    for (i in 1:K) {
      error.f[i] <- norm(Omega.hat.list[[i]] - Omega.true.list[[i]],
        type = "F"
      )
      error.max[i] <- norm(Omega.hat.list[[i]] - Omega.true.list[[i]],
        type = "M"
      )
      tpr[i] <- length(intersect(which(Omega.hat.list[[i]] !=
        0), which(Omega.true.list[[i]] != 0))) / length(which(Omega.true.list[[i]] !=
        0))
      tnr[i] <- length(intersect(which(Omega.hat.list[[i]] ==
        0), which(Omega.true.list[[i]] == 0))) / length(which(Omega.true.list[[i]] ==
        0))
    }
  }
  else {
    Omega.hat.list.off <- Omega.hat.list
    Omega.true.list.off <- Omega.true.list
    for (i in 1:K) {
      diag(Omega.hat.list.off[[i]]) <- 0
      diag(Omega.true.list.off[[i]]) <- 0
      error.f[i] <- norm(Omega.hat.list.off[[i]] - Omega.true.list.off[[i]],
        type = "F"
      )
      error.max[i] <- norm(Omega.hat.list.off[[i]] - Omega.true.list.off[[i]],
        type = "M"
      )
      diag(Omega.hat.list.off[[i]]) <- NA
      diag(Omega.true.list.off[[i]]) <- NA
      tpr[i] <- length(intersect(which(Omega.hat.list.off[[i]] !=
        0), which(Omega.true.list.off[[i]] != 0))) / length(which(Omega.true.list.off[[i]] !=
        0))
      tnr[i] <- length(intersect(which(Omega.hat.list.off[[i]] ==
        0), which(Omega.true.list.off[[i]] == 0))) / length(which(Omega.true.list.off[[i]] ==
        0))
    }
  }
  
  # output
  Out <- list()
  Out$av.error.f <- mean(error.f)
  Out$av.error.max <- mean(error.max)
  Out$av.tpr <- mean(tpr)
  Out$av.tnr <- mean(tnr)
  Out$error.f <- error.f
  Out$error.max <- error.max
  Out$tpr <- tpr
  Out$tnr <- tnr
  return(Out)
}
