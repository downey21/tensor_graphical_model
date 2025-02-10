
# -*- coding: utf-8 -*-

rm(list = ls())

# install.packages("Tlasso")

# https://cran.r-project.org/web/packages/Tlasso/vignettes/Tlasso.html

library(Tlasso)

m.vec <- c(5,5,5)  # dimensionality of a tensor 
# m1, m2, m3
n <- 5   # sample size 

Omega.true.list = list()
for (k in 1:length(m.vec)) {
    Omega.true.list[[k]] <- Tlasso::ChainOmega(m.vec[k], sd = k, norm.type = 1) # sd: seed number
}

Omega.true.list

sum(Omega.true.list[[1]]^2) # 1 if norm.type = 2
Omega.true.list[[1]][1, 1] # 1 if norm.type = 1

Sigma.true.list = list()
for (k in 1:length(m.vec)) {
    Sigma.true.list[[k]] <- solve(Omega.true.list[[k]])
}

# Separable Tensor Normal Distribution
DATA <- Tlasso::Trnorm(n = n, m.vec = m.vec, mu = array(0, m.vec), Sigma.list = Sigma.true.list, sd = 1) 

dim(DATA)
DATA

# Tlasso algorithm

# lambda.thm is regularization parameter
lambda.thm <- 20 * c(
    sqrt(log(m.vec[1])/(n*prod(m.vec))), 
    sqrt(log(m.vec[2])/(n*prod(m.vec))), 
    sqrt(log(m.vec[3])/(n*prod(m.vec)))
)

out.tlasso <- Tlasso::Tlasso.fit(DATA, T = 1, lambda.vec = lambda.thm, norm.type = 2, thres = 1e-05) # lambda.vec; Defalut is NULL, s.t. it is tuned via HUGE package directly.

# output is a list of estimation of precision matrices
str(out.tlasso)
out.tlasso[[1]]
out.tlasso

Tlasso::est.analysis(out.tlasso, Omega.true.list, offdiag = FALSE) # If offdiag = TRUE, diagnoal in each matrix is ingored when comparing two matrices. Default is TRUE.
# error.kro: error in Frobenius norm of kronecker product
# tnr.kro: TPR of kronecker product
# tnr.kro: TNR of kronecker product
# av.error.f: averaged Frobenius norm error across all modes
# av.error.max: averaged Max norm error across all modes
# av.tpr: averaged TPR across all modes
# av.tnr: averaged TNR across all modes
# error.f: error in Frobenius norm of each mode
# error.max: error in Max norm of each mode
# tpr: TPR of each mode
# tnr: TNR of each mode

mat.list <- list() # list of matrices of test statistic value  
for (k in 1:length(m.vec)) {
    rho <- Tlasso::covres(DATA, out.tlasso, k = k) 
    # To compute test statistic, we first need to compute sample covariance of residuals
    # rho: sample covariance matrix of residuals, including diagnoal

    bias_rho <- Tlasso::biascor(rho, out.tlasso, k = k)
    # bias corrected sample covariance of residuals, excluding diagnoal
    # bias_rho: for bias correction

    varpi2 <- Tlasso::varcor(DATA, out.tlasso, k = k)
    # variance correction term for kth mode's sample covariance of residuals
    # varpi2: variance correction

    tautest <- matrix(0, m.vec[k], m.vec[k])
    for (i in 1:(m.vec[k]-1)) {
        for (j in (i+1):m.vec[k]){
            tautest[j,i] = tautest[i,j] = sqrt((n-1) * prod(m.vec[-k])) * bias_rho[i,j]/sqrt(varpi2*rho[i,i]*rho[j,j])
            # compute final test statistic 
        }
    }

    mat.list[[k]] <- tautest
}
mat.list[[1]]
mat.list

# inference measures (off-diagnoal), critical value is 0.975 quantile of standard normal
Tlasso::infer.analysis(mat.list, qnorm(0.975), Omega.true.list, offdiag = TRUE) # If offdiag = TRUE, diagnoal in each matrix is ingored when comparing two matrices. Default is TRUE.
# fp: number of false positive of each mode
# fn: number of false negative of each mode
# d: number of all discovery of each mode
# nd: number of all non-discovery of each mode
# t: number of all true non-zero entries of each mode

k <- 1 # interested mode 
upsilon <- 0.1  # control level

# compute the difference between FDP and upsilon
fun <- function(varsigma, mk, upsilon, tautest) {
    return((2*(1-pnorm(varsigma))*mk*(mk-1))/max(1,sum(sign(abs(tautest) > varsigma))) - upsilon)
}

# select a critical value in (0,6) that has the samllest difference 
diff <- c(); ind <- 1; inter <- seq(0,6,0.0001)
for (varsigma in inter) {
    diff[ind] <- fun(varsigma,mk=m.vec[k],upsilon=upsilon,tautest=mat.list[[k]])
    ind <- ind + 1
}
# the smallest critical value that constrains FDP under upsilon
critical <- inter[min(which(diff < 0))]

# testing hypothesis with the critcal value 
# FDR will converge to the limit proved in Lyu et al. 2019.
inference.FDR <- Tlasso::infer.analysis(mat.list, critical, Omega.true.list, offdiag = TRUE)
inference.FDR

k <- 1 # interested mode
# true graph structure. 
# set thres=0 in case true edge is eliminated
Tlasso::graph.pattern(Omega.true.list[[1]], main = 'True graph of mode 1', thres = 0)

inf.mat <- mat.list[[k]] > qnorm(0.975)
# set thres=0 (<1) since inf.mat is logical
Tlasso::graph.pattern(inf.mat, main = 'Inference of mode 1', thres = 0)
