"""
EM algorithm implementation to train Gaussian Mixture Models.
"""

import numpy as np
import pickle
from numpy import ndarray, random
from numpy import load, array, dot, sum
from scipy.stats import multivariate_normal

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import logging

logger = logging.getLogger(__name__)

class GaussianMixtureModel():
    """Definition of Gaussian Mixture Model (GMM)

    Update with given training data by EM algorithm is implemented.
    """

    def __init__(self, M:int, D:int):
        """
        Each Gaussian is initialized as zero means, unit covariance.
        Weight is initailized as equal probability.
        Args:
            M (int): number of mixtures
            D (int): dimension of smaples
        """
        self._M = M
        self._D = D
        self.Mu = np.random.randn(M, D)
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
            x (ndarray): random variable of GMM

        Returns:
            ndarray: probability density of GMM. Note this is NOT in log scale.
        """
        return sum(self.Pi[k] * multivariate_normal(self.Mu[k, :], self.Sigma[k, :, :]).pdf(x) \
            for k in range(self._M))

    def log_likelihood(self, x: np.ndarray) -> float:
        """Calculate log-likelihood of vectors.

        Args:
            x (np.ndarray): D-dimesional vectors. Its number is N. (N, D)

        Returns:
            float: Total log-likelihood
        """
        N, D = x.shape
        y = np.zeros((N, self._M)) #keep all Gaussian pdf (not smart)
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
        try:
            gam = array([ self.Pi[m] * multivariate_normal(self.Mu[m, :], self.Sigma[m,:,:], allow_singular=True).pdf(x) \
                for m in range(self._M)]).transpose() # (N,M)
        except np.linalg.LinAlgError as e:
            print(self.Pi)
            print(self.Sigma)
            raise e
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
            #self.Sigma = np.zeros((M, D, D))
        #for m in range(self._M):
        #    if sum(gamma[:,m]) < 1.0E-5:
        #        continue
            for n in range(N):
                wk = x - self.Mu[m, :] # distance between x and m-th Gaussian's mean
                wk = wk[n, :, np.newaxis]
                self.Sigma[m, :, :] = self.Sigma[m, :, :] + gamma[n, m] * np.dot(wk, wk.T)
            #print(self.Sigma[m,:,:])
            tmp_sig = self.Sigma[m, :, :] / np.sum(gamma[:, m])
            if not (tmp_sig > 1E4).any():
                self.Sigma[m, :, :] = tmp_sig
            else:
                print('too large: self.Sigma=', self.Sigma)
                print('gamma = ', np.sum(gamma[:,m]), ' covariance is not updated.')
                #input()
        return True

    def split_Gaussian(self):
        """
        Not implemented yet.
        """
        raise NotImplementedError('not implemented')
        #gauss_vars = {}
        #for m in self._M:
        #    var = np.sum(np.diag(self.Sigma[m,:,:]))
        #    gauss_vars[m] = var
        #max_m = max(guass_vars, key=gauss_vars.get)
        #self._M += 1
        #
        #self.Mu = np.random.randn(self._M, self._D)
        #self.Sigma = np.zeros((self._M, self._D, self._D))


def train_gmm(gmm:GaussianMixtureModel, X:np.ndarray, max_it:int =12):
    """Train GMM

    Args:
        gmm (GaussianMixtureModel): _description_
        X (np.ndarray): _description_
        max_it (int, optional): number of iteration. Defaults to 12.

    Returns:
        _type_: _description_
    """
    N = X.shape[0]
    loglikelihood_history = []  # distortion measure
    with logging_redirect_tqdm(loggers=[logger]):
        pbar = tqdm(range(max_it), desc=f"gmm-train(M={gmm.M})", postfix="postfix", ncols=80)
        for it in pbar:
    #for it in range(0, max_it):
        #_ll = gmm.log_likelihood(X)
            _gamma, _ll = gmm.update_e_step(X)
            loglikelihood_history.append(_ll)
            gmm.update_m_step(X, _gamma)
            pbar.write('GMM EM training: step={_i} E[log(P(X)]={_l}'.format(_i=len(loglikelihood_history), _l=_ll/N))
    return gmm, loglikelihood_history




def load_from_file(input_file:str):
    with open(input_file, 'rb') as f:
        indata = pickle.load(f)
        #print(indata)
    X = indata['sample'] # data points.
    Param = indata['model_param'] # model parameters.
    return X, Param

def mixutre_number_test(X, Param, max_mixnum:int = 10):
    N, D = X.shape[0], X.shape[1]
    K = Param.K
    Pi, Mu, Sigma = Param.Pi, Param.Mu, Param.Sigma

    mixnum_likelihood = {}
    L = np.linalg.cholesky(np.cov(X.T))
    for mix_num in range(2, max_mixnum):
        np.random.seed(0)
        gmm = GaussianMixtureModel(mix_num, D)
        gmm.Sigma = np.zeros([mix_num, D, D])
        gmm.Mu = np.zeros([mix_num, D])
        for m in range(mix_num):
            gmm.Mu[m, :] = X.mean() + np.dot(L, np.random.randn(D))
            gmm.Sigma[m, :, :] = np.eye(D)
        gmm, loglikelihood_history = train_gmm(gmm, X, max_it=100)
        print('GMM EM training: mixture={i} E[log(P(X)]={_l}'.format(i=mix_num, _l=loglikelihood_history[-1]))
        mixnum_likelihood[mix_num] = loglikelihood_history[-1]
    print(mixnum_likelihood)




