# -*- coding: utf-8 -*-
"""
@author: H-divergence: A Decision-Theoretic Discrepancy Measure for Two Sample Tests
@Implementation of H-Divergence two sample test in our paper on Blob dataset

BEFORE USING THIS CODE:
Numpy and Sklearn are required. Users can install Python via Anaconda to obtain both packages.
Anaconda can be found in https://www.anaconda.com/distribution/#download-section.

This code is based on the two sample test experiment in "Learning Deep Kernels for Non-Parametric Two-Sample Tests".
The original github repo is https://github.com/fengliu90/DK-for-TST, please also follow their instruction to reproduce the baselines in our paper.
"""
import numpy as np
from sklearn.utils import check_random_state
from utils import JSV_Gaussian
import argparse
import os
join=os.path.join

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--ncomp', type=int, default=10, help="number of components in the GMM (default: 10)")
parser.add_argument('--ntrial', type=int, default=10, help="number of trials (default: 10)")
parser.add_argument('--exptype', type=str, default="power", help="type of experiment (power or typei) (default: power)")
args = parser.parse_args()

def sample_blobs(n, rows=3, cols=3, sep=1, rs=None):
    """Generate Blob-S for testing type-I error."""
    rs = check_random_state(rs)
    correlation = 0
    # generate within-blob variation
    mu = np.zeros(2)
    sigma = np.eye(2)
    X = rs.multivariate_normal(mu, sigma, size=n)
    corr_sigma = np.array([[1, correlation], [correlation, 1]])
    Y = rs.multivariate_normal(mu, corr_sigma, size=n)
    # assign to blobs
    X[:, 0] += rs.randint(rows, size=n) * sep
    X[:, 1] += rs.randint(cols, size=n) * sep
    Y[:, 0] += rs.randint(rows, size=n) * sep
    Y[:, 1] += rs.randint(cols, size=n) * sep
    return X, Y

def sample_blobs_Q(N1, sigma_mx_2, rows=3, cols=3, rs=None):
    """Generate Blob-D for testing type-II error (or test power)."""
    rs = check_random_state(rs)
    mu = np.zeros(2)
    sigma = np.eye(2) * 0.03
    X = rs.multivariate_normal(mu, sigma, size=N1)
    Y = rs.multivariate_normal(mu, np.eye(2), size=N1)
    # assign to blobs
    X[:, 0] += rs.randint(rows, size=N1)
    X[:, 1] += rs.randint(cols, size=N1)
    Y_row = rs.randint(rows, size=N1)
    Y_col = rs.randint(cols, size=N1)
    locs = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    for i in range(9):
        corr_sigma = sigma_mx_2[i]
        L = np.linalg.cholesky(corr_sigma)
        ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
        ind2 = np.concatenate((ind, ind), 1)
        Y = np.where(ind2, np.matmul(Y,L) + locs[i], Y)
    return X, Y

# Setup seeds
np.random.seed(1102)
# Setup for all experiments
N_per = 100 # permutation times
alpha = 0.05 # test threshold
n_list = [10,20,40,50,70,80,90,100] # number of samples in per mode
K = args.ntrial # number of trials
N = 100 # # number of test sets
N_f = 100.0 # number of test sets (float)
n_components = args.ncomp
# Generate variance and co-variance matrix of Q
sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2 = np.zeros([9,2,2])
for i in range(9):
    sigma_mx_2[i] = sigma_mx_2_standard
    if i < 4:
        sigma_mx_2[i][0 ,1] = -0.02 - 0.002 * i
        sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
    if i==4:
        sigma_mx_2[i][0, 1] = 0.00
        sigma_mx_2[i][1, 0] = 0.00
    if i>4:
        sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i-5)
        sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i-5)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

fname = join(args.log_dir, 'blob_gmm' + '.txt')

# For each n in n_list, run two-sample test
for n in n_list:
    N1 = 9 * n
    N2 = 9 * n
    Results = np.zeros([1, K])
    # Repeat experiments K times (K = 10) and report average test power/typeI error
    for kk in range(K):
        # Compute test power/typeI error
        H_u = np.zeros(N)
        T_u = np.zeros(N)
        M_u = np.zeros(N)
        count_u = 0
        for k in range(N):
            # Generate Blob
            np.random.seed(seed=11 * k + 10 + n)
            if args.exptype == "power":
                s1,s2 = sample_blobs_Q(N1, sigma_mx_2)
            elif args.exptype == "typei":
                s1,s2 = sample_blobs(N1) # for validating type-I error (s1 ans s2 are from the same distribution)
            else:
                raise NotImplementedError("Please choose either power or typei experiment")

            S = np.concatenate((s1, s2), axis=0)
            # Run two sample test on generated data
            h_u, threshold_u, jsv_u = JSV_Gaussian(S, N_per, N1, alpha, n_components)
            # Gather results
            count_u = count_u + h_u
            # if k % 10 == 0:
            #     print(n, k, "JSV Gaussian:", count_u)
            H_u[k] = h_u
            T_u[k] = threshold_u
            M_u[k] = jsv_u
        # Print test power of MMD-D
        print("n =",str(n),"--- Test Power of V-Div GMM on Blob: ", H_u.sum()/N_f)
        Results[0, kk] = H_u.sum() / N_f
        print("n =",str(n),"--- Test Power of V-Div GMM on Blob (K times): ",Results[0])
        print("n =",str(n),"--- Average Test Power of V-Div GMM on Blob: ",Results[0].sum()/(kk+1))
    with open(fname, 'a') as f:
        f.write("Blob_" + args.exptype + "_" + str(n) + "_ncomp_" + str(n_components) + "_GMM:" + str(Results) + '\n')