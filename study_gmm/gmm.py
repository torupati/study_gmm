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

    def update_e_step(self, x:ndarray) -> (np.ndarray, float):
        """Calculate gamma of GMM (Expectation step of GMM training)

        Args:
            x(N,D), trainig samples (n-th sample, d dimensional vector)

        Returns:
            gam(np.ndarray): probability of x(n) in k-th Gaussian, P(k|x[n]), shape=(N,K)
            llh(float): log-likelihood
        """
        N = x.shape[0]
        # caluclate P(x[n], k) = P(k)P(x[n]|k) and hold all as array (n,m)
        gam = array([ self.Pi[m] * multivariate_normal(self.Mu[m, :], self.Sigma[m,:,:]).pdf(x) \
            for m in range(self._M)]).transpose() # (N,M)
        llh = 0.0 # log likelihood of all training data
        for n in range(N):
            _s = sum(gam[n,:]) # P(x[n]) = sum_k P(k,x[n])
            gam[n,:] = gam[n,:] / _s
            llh += np.log(_s)
        return gam, llh

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


def train_gmm(gmm:GaussianMixtureModel, X:np.ndarray):
    max_it =12 
    loglikelihood_history = []  # distortion measure
    for it in range(0, max_it):
        #_ll = gmm.log_likelihood(X)
        _gamma, _ll = gmm.update_e_step(X)
        loglikelihood_history.append(_ll)
        gmm.update_m_step(X, _gamma)
        print('GMM EM training: step={_i} E[log(P(X)]={_l}'.format(_i=it, _l=_ll/N))
    return gmm, loglikelihood_history

gmm, loglikelihood_history = train_gmm(gmm, X)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(range(0, len(loglikelihood_history)), loglikelihood_history, color='k', linestyle='-', marker='o')
ax.set_xlim([0,len(loglikelihood_history)])
ax.set_ylabel("log liklihood")
ax.set_xlabel("iteration step")
#plt.ylim([40, 80])
ax.grid(True)
out_pngfile = "loglikelihood-gmm.png"
plt.savefig(out_pngfile)
plt.show(block=False)
print(out_pngfile)
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

