import numpy as np
import pickle
from numpy import ndarray, random
from numpy import load, array, dot, sum
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt



class GaussianMixtureModel():
    """Definition of clusters for K-means altorithm
    """

    def __init__(self, M:int, D:int):
        """_summary_
        Each Gaussian is initialized as zero means, unit covariance.
        Weight is initailized as equal probability.
        Args:
            M (int): number of mixtures
            D (int): dimension of smaples
        """
        self._M = M
        self._D = D
        self.Mu = np.random.randn(M, D) #centroid
        self.Sigma = np.zeros((M, D, D))
        for m in range(self._M):
            self.Sigma[m,:,:] = np.eye(D)
        self.Pi = np.ones(M) / M # equal probability for initial condition

    @property
    def M(self) -> int:
        """Number of Gaussians

        Returns:
            int: number of Gaussians
        """
        return self._M

    @property
    def D(self) -> int:
        """Dimension of input data

        Returns:
            int: dimension of input data
        """
        return self._D
    
    def __repr__(self):
        return '{n} (M={k} D={d})'.format(n=self.__class__.__name__, k=self._M, d= self._D)

    def set_Mu(self, m, val):
        """Set m-th Gaussian's mean vector

        Args:
            m (_type_): _description_
            val (_type_): _description_
        """
        assert m < self._M
        assert len(val) == self._D
        self.Mu[m,:] = val

    def probability(self, x:ndarray) -> ndarray:
        """Calculate probability of this GMM at given sample points

        Args:
            x (ndarray): _description_

        Returns:
            ndarray: _description_
        """
        return sum(self._Pi[k] * multivariate_normal(self.Mu[k, :], self.Sigma[k, :, :]).pdf(x) \
            for k in range(self._K))

    def log_likelihood(self, x: np.ndarray):
        N, D = x.shape
        y = np.zeros((N, self._M)) #keep all Gaussian pdf (nost smart)
        for k in range(self._M):
            y[:, k] = multivariate_normal(self.Mu[k, :], self.Sigma[k, :, :]).pdf(x)  # (K,N)
        lh = 0
        for n in range(N):
            wk = 0
            for k in range(self._M):
                wk = wk + self.Pi[k] * y[n, k]
            lh = lh + np.log(wk)
        return lh

    def update_e_step(self, x:ndarray) -> ndarray:
        """Calculate gamma of GMM (E step of GMM training)

        Args:
            x(N,D), trainig samples (n-th sample, d dimensional vector)

        Returns:
            gam(N,K), probability of x(n) in k-th Gaussian
        """
        N = x.shape[0]
        gam = array([ self.Pi[m] * multivariate_normal(self.Mu[m, :], self.Sigma[m,:,:]).pdf(x) \
            for m in range(self._M)]).transpose() # (N,M)
        print("new gaussian = ", gam[0,0], self.Pi[0])
        for n in range(N):
            gam[n,:] = gam[n,:] / sum(gam[n,:])
        return gam

    def update_m_step(self, x:ndarray, gamma:ndarray) -> bool:
        """Update parameter of GMM with given allocation.

        Args:
            X(N,D), training samples. (N is number of samples, D is dimension)
        """
        N, D, M = x.shape[0], x.shape[1], gamma.shape[1]
        self.Pi = sum(gamma, axis=0) / sum(gamma)
        self.Mu = dot(gamma.transpose(), x)
        for m in range(self.M):
            if sum(gamma[:,m]) < 1.0E-10:
                continue
            self.Mu[m,:] = self.Mu[m,:] / sum(gamma[:,m])
            self.Sigma = np.zeros((M, D, D))
        for m in range(self._M):
            for n in range(N):
                wk = x - self.Mu[m, :] # distance between x and m-th Gaussian's mean
                wk = wk[n, :, np.newaxis]
                self.Sigma[m, :, :] = self.Sigma[m, :, :] + gamma[n, m] * np.dot(wk, wk.T)
            self.Sigma[m, :, :] = self.Sigma[m, :, :] / np.sum(gamma[:, m])
        return True


def _gmm(x:ndarray, pi:ndarray, mu:ndarray, sigma:ndarray) -> ndarray:
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
    print("old gaussian = ", g[0,0], w[0])
    return g

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
Pi, Mu, Sigma = Param.Pi, Param.Mu, Param.Sig

gmm = GaussianMixtureModel(K, D)
gmm.Mu = Param.Mu
Gamma = np.c_[np.ones((N, 1)), np.zeros((N, K-1))]

if len(Sigma.shape) == 2:
    new_sigma = np.zeros((K, D, D))
    for k in range(K):
        for d in range(D):
            new_sigma[k, d, d] = Sigma[k, d]
    gmm.Sigma = new_sigma
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


max_it =12 
it = 0
loglikelihood_history = []  # distortion measure
for it in range(0, max_it):
    ll = gmm.log_likelihood(X)
    ll2 = log_likelihood_gmm(X,Pi,Mu,Sigma)
    loglikelihood_history.append(ll)
    print("itr={} log-likelihood={} {}".format(it, ll, ll2))

    Gamma = gmm_e_step(X, Pi, Mu, Sigma)
    print('oirg', Gamma[0,0])
    _gamma = gmm.update_e_step(X)
    print(_gamma[0,0])

    Pi, Mu, Sigma = gmm_m_step(X, Gamma)
    gmm.update_m_step(X, _gamma)

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

# メイン ----------------------------------
def test_e_step():
    Gamma = gmm_e_step(X, Pi, Mu, Sigma)
    
    # 表示 ----------------------------------
    plt.figure(1, figsize=(4, 4))
    show_mixgauss_prm(X, Gamma, Pi, Mu, Sigma)

# メイン ----------------------------------
def test_m_stemp():
    Pi, Mu, Sigma = m_step_mixgauss(X, Gamma)
    
    # 表示 ----------------------------------
    plt.figure(1, figsize=(4, 4))
    show_mixgauss_prm(X, Gamma, Pi, Mu, Sigma)

