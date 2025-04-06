
# -*- coding: utf-8 -*-

# [Min et al. 2022]
# Min, Keqian, Qing Mai, and Xin Zhang. "Fast and separable estimation in high-dimensional tensor Gaussian graphical models." *Journal of Computational and Graphical Statistics* 31.1 (2022): 294-300.

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
