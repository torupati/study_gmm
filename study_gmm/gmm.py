import numpy as np
import pickle
from numpy import ndarray, random
from numpy import load, array, dot, sum
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def gmm(x:ndarray, pi:ndarray, mu:ndarray, sigma:ndarray) -> ndarray:
    N, D = x.shape
    K = len(pi)
    return sum(pi[k] * multivariate_normal(mu[k, :], sigma[k, :, :]).pdf(x) for k in range(K))


def gmm_e_step(x:ndarray, w:ndarray, mu:ndarray, sigma:ndarray) -> ndarray:
    """
    Calculate gamma of GMM (E step of GMM training)

    Parameters
    ----------
    x(N,D), samples
    w(K),
    mu(K,D), mean vector
    sigma(K,D,D), covaraince matrices
    Returns
    -------
    g(N,K), probability of x(n) in k-th Gaussian
    """
    N, D, K = x.shape[0], x.shape[1], w.shape[0]
    #y = np.zeros((N, K))
    #for k in range(K):
    #    y[:, k] = gauss(x, mu[k, :], sigma[k, :, :])  # KxN
    #y = array([ gauss(x, mu[k, :], sigma[k,:,:]) for k in range(K)]).transpose() # (N,K)
    #gamma = np.zeros((N, K))
    #for n in range(N):
    #    wk = np.zeros(K)
    #    for k in range(K):
    #        wk[k] = pi[k] * y[n, k]
    #    gamma[n, :] = wk / np.sum(wk)
    g = array([ w[k] * multivariate_normal(mu[k, :], sigma[k,:,:]).pdf(x) for k in range(K)]).transpose() # (N,K)
    for n in range(N):
        g[n,:] = g[n,:] / sum(g[n,:])
    return g

# メイン ----------------------------------
def test_e_step():
    Gamma = gmm_e_step(X, Pi, Mu, Sigma)
    
    # 表示 ----------------------------------
    plt.figure(1, figsize=(4, 4))
    show_mixgauss_prm(X, Gamma, Pi, Mu, Sigma)

def gmm_m_step(x:ndarray, gamma:ndarray) -> (ndarray, ndarray, ndarray):
    """
    Update Gaussian and weights of GMM (M step of GMM training)

    Parameters
    ----------
    x(N,D), samples
    g(N,K), probability of x(n) in k-th Gaussian

    Returns
    -------
    w(K),
    mu(K,D), mean vector
    sigma(K,D,D), covaraince matrix
    """
    N, D, K = x.shape[0], x.shape[1], gamma.shape[1]
    # update pi
    pi = sum(gamma, axis=0) / sum(gamma)
    # update mu
    mu = dot(gamma.transpose(), x)
    for k in range(K):
        if sum(gamma[:,k]) < 1.0E-10:
            continue
        mu[k,:] = mu[k,:] / sum(gamma[:,k])
    # update sigma
    sigma = np.zeros((K, D, D))
    for k in range(K):
        for n in range(N):
            wk = x - mu[k, :]
            wk = wk[n, :, np.newaxis]
            sigma[k, :, :] = sigma[k, :, :] + gamma[n, k] * np.dot(wk, wk.T)
        sigma[k, :, :] = sigma[k, :, :] / np.sum(gamma[:, k])
    return pi, mu, sigma

# メイン ----------------------------------
def test_m_stemp():
    Pi, Mu, Sigma = m_step_mixgauss(X, Gamma)
    
    # 表示 ----------------------------------
    plt.figure(1, figsize=(4, 4))
    show_mixgauss_prm(X, Gamma, Pi, Mu, Sigma)


infile = "out.pickle"
with open(infile, 'rb') as f:
    indata = pickle.load(f)
    print(indata)
X = indata['sample']
Param = indata['model_param']
print(X.shape)
#print(Param.__dict__.keys())
max_it = 16 # 繰り返しの回数
N, D = X.shape[0], X.shape[1]
K = Param.K
Gamma = np.c_[np.ones((N, 1)), np.zeros((N, K-1))]
Pi, Mu, Sigma = Param.Pi, Param.Mu, Param.Sig
if len(Sigma.shape) == 2:
    new_sigma = np.zeros((K, D, D))
    for k in range(K):
        for d in range(D):
            new_sigma[k, d, d] = Sigma[k, d]
    Sigma = new_sigma

assert(Gamma.shape[1] == Param.Mu.shape[0])

# 混合ガウスの目的関数 ----------------------
def log_likelihood_gmm(x, pi, mu, sigma):
    # x: NxD
    # pi: Kx1
    # mu: KxD
    # sigma: KxDxD
    # output lh: NxK
    N, D = x.shape
    K = len(pi)
    y = np.zeros((N, K))
    for k in range(K):
        y[:, k] = multivariate_normal(mu[k, :], sigma[k, :, :]).pdf(x)  # (K,N)
    lh = 0
    for n in range(N):
        wk = 0
        for k in range(K):
            wk = wk + pi[k] * y[n, k]
        lh = lh + np.log(wk)
    return lh

#
#
#
#Pi, Mu, Sigma = Param["Pi"], Param["Mu"], Param["Sigma"]
#Gamma = np.c_[np.ones((N, 1)), np.zeros((N, 2))]

max_it =12 
it = 0
loglikelihood_history = []  # distortion measure
for it in range(0, max_it):
    Gamma = gmm_e_step(X, Pi, Mu, Sigma)
    ll = log_likelihood_gmm(X,Pi,Mu,Sigma)
    loglikelihood_history.append(ll)
    print("itr={} log-likelihood={}".format(it, ll))
    Pi, Mu, Sigma = gmm_m_step(X, Gamma)

#print(np.round(Err, 2))
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(range(0, len(loglikelihood_history)), loglikelihood_history, color='k', linestyle='-', marker='o')
ax.set_xlim([0,len(loglikelihood_history)])
ax.set_ylabel("log liklihood")
ax.set_xlabel("iteration step")
#plt.ylim([40, 80])
ax.grid(True)
plt.savefig("loglikelihood-gmm.png")
plt.show(block=False)
input("wait")