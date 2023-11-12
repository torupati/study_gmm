"""
K-means clustering Implementation
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pickle
from tqdm import tqdm
import argparse
import logging

import kmeans_plot

logger = logging.getLogger(__name__)

class KmeansCluster():
    """Definition of clusters for K-means altorithm
    """
    COV_NONE = 0
    COV_FULL = 1
    COV_DIAG = 2

    TRAIN_VAR_OUTSIDE = 0
    TRAIN_VAR_INSIDE = 1

    def __init__(self, K:int, D:int, **kwargs):
        """_summary_

        Args:
            K (int): number of cluster. Fixed.
            D (int): dimension of smaples. Fixed.

            covariance_mode(str): "none"(default), "diag" or "full" (optional)
            trainvars(str): "outside"(default) or "inside" (optional)

        Raises:
            Exception: wrong covariance mode input
            Exception: wrong training variable mode input

        Note:
        If you want to change the number of clusters, new instance with desired cluster numbers should be created.
        """

        self._K = K
        self._D = D
        self.Mu = np.random.randn(K, D) #centroid
        #self.Pi = np.ones(K) / K # Cumulative probability
        if kwargs.get('covariance_mode') == None:
            self._cov_mode = KmeansCluster.COV_DIAG
            self.Sigma = np.ones((K, D))
        elif kwargs.get('covariance_mode') == 'full':
            self._cov_mode = KmeansCluster.COV_FULL
            self.Sigma = np.ones((K, D, D))
        elif kwargs.get('covariance_mode') == 'diag':
            self._cov_mode = KmeansCluster.COV_DIAG
            self.Sigma = np.ones((K, D))
        elif kwargs.get('covariance_mode') == 'none':
            self._cov_mode = KmeansCluster.COV_NONE
            self.Sigma = None
        else:
            raise Exception('invalid covariance mode')

        # define training variables
        print(kwargs.get('trainvars','outside') == 'inside')
        if kwargs.get('trainvars','outside') == 'outside':
            self._train_mode = KmeansCluster.TRAIN_VAR_OUTSIDE
        elif kwargs.get('trainvars','outside') == 'inside':
            self._train_mode = KmeansCluster.TRAIN_VAR_INSIDE
            self._loss = 0.0
            self._X0 = np.zeros([K])
            self._X1 = np.zeros([K, D])
            if self._cov_mode == KmeansCluster.COV_NONE:
                self._X2 = None
            elif self._cov_mode == KmeansCluster.COV_FULL:
                self._X2 = np.zeros([K,D,D])
            elif self._cov_mode == KmeansCluster.COV_DIAG:
                self._X2 = np.zeros([K, D])
        else:
            raise Exception('invalid training variable mode.')

    @property
    def K(self) -> int:
        """Number of clusters

        Returns:
            int: cluster count
        """
        return self._K

    @property
    def D(self) -> int:
        """Dimension of input data

        Returns:
            int: dimension of input data
        """
        return self._D

    def __repr__(self):
        return '{n} (K={k} D={d})'.format(n=self.__class__.__name__, k=self._K, d= self._D)

    @property
    def covariance_mode(self) -> str:
        return self._cov_mode.name

    @property
    def train_vars_mode(self) -> str:
        return list(["outside", 'inside'])[self._train_mode]

    def distortion_measure(self, x:np.ndarray, r:np.ndarray) -> float:
        """Calculate distortion measure.

        Args:
            x (ndarray): input samples (N,D)
            r (ndarray): sample alignment to clusters (N,K)

        Returns:
            float: disotrion (average per sample). If N = 0, 0.0 is returned.
        """
        J = 0.0
        n_sample = x.shape[0]
        if n_sample == 0:
            return 0.0
        for n in range(n_sample):
            dist = [sum(v*v for v in x[n, :] - self.Mu[k, :]) for k in range(self._K)]
            J = J + np.dot(r[n, :], dist)
        return J/n_sample

    def get_alignment(self, x: np.ndarray) -> np.ndarray:
        """
        Hard allocation of each sample to a cluster (or Gaussians)

        Args:
            x (ndarray): input samples (N,D)
        Returns:
            r (ndarray): sample alignment to clusters (N,K)
        """
        if len(x.shape) != 2:
            raise Exception("dimension error")
        N = x.shape[0]
        r = np.zeros((N, self._K))
        for n in range(N):
            r[n, np.argmin([ sum(v*v for v in x[n, :] - self.Mu[k, :]) for k in range(self._K)])] = 1
            r[n, :] = r[n,:]/r[n,:].sum()
            #wk = [(x[n, 0] - mu[k, 0])**2 + (x[n, 1] - mu[k, 1])**2 for k in range(K)]
            #r[n, argmin(wk)] = 1
        return r

    def get_soft_alignment(self, x: np.ndarray) -> np.ndarray:
        """
        Soft alocation of each sample to Gaussians. E-step of EM algorithm to GMM.

        Args:
            x (ndarray): input samples (N,D)
        Returns:
            r (ndarray): sample alignment to clusters (N,K)
        """
        if len(x.shape) != 2:
            raise Exception("dimension error")
        N = x.shape[0]
        r = np.zeros((N, self._K))
        g = array([ self.Pi[k] * multivariate_normal(self.Mu[k, :], self.Sigma[k,:,:]).pdf(x) \
            for k in range(self._K)]).transpose() # (N,K)
        for n in range(N):
            r[n,:] = r[n,:] / sum(r[n,:])
        return r

    def get_centroid(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Calculate centroid from given alignment.

        Args:
            x (ndarray): input samples (N,D)
            r (ndarray): sample alignment to clusters (N,K)

        Returns:
            ndarray: updated centroid(K,D)
        """
        #  mu[k, :] = np.sum_n(r[n, k] * x[n,:]) / sum_{n,k'}(r[n, k'])
        _mu = np.dot(r.transpose(), x)
        for k in range(self._K):
            if sum(r[:,k]) < 0.01:
                # this cluster has no sample aligned.
                continue
            _mu[k,:] = _mu[k,:] / sum(r[:,k])
            # divied by number of aligned sample to the k-th cluster
        return _mu

    def PushSample(self, x: np.ndarray) -> (int, float):
        """Push one training sample to inner training variables

        Args:
            x (ndarray): traiing sample (D,)

        Returns:
            int: aligned cluster's index (between 0 and K-1)
        """
        if self._train_mode != self.TRAIN_VAR_INSIDE:
            raise Exception('train mode is not TRAIN_VAR_INSIDE.')
        costs = [ sum([v*v for v in (x - self.Mu[k, :])]) for k in range(self._K)]
        k_min = np.argmin(costs)
        self._loss += costs[k_min]
        self._X0[k_min] += 1
        self._X1[k_min,:] += x
        if self._cov_mode == KmeansCluster.COV_DIAG:
            self._X2[k_min,:] = (x * x)
        elif self._cov_mode == KmeansCluster.COV_FULL:
            self._X2[k_min,:,:] = (x.reshape(self._D,1) * x.reshape(1, self._D)) # not checked.
        return k_min

    def ClearTrainigVariables(self):
        """Reset inside statistics for training

        Raises:
            Exception: invalid triaing setting
        """
        if self._train_mode != self.TRAIN_VAR_INSIDE:
            raise Exception('train mode is not TRAIN_VAR_INSIDE.')
        self._loss = 0.0
        self._X0 = np.zeros([self._K])
        self._X1 = np.zeros([self._K, self._D])
        if self._cov_mode == KmeansCluster.COV_NONE:
            self._X2 = None
        elif self._cov_mode == KmeansCluster.COV_FULL:
            self._X2 = np.zeros([self._K, self._D, self._D])
        elif self._cov_mode == KmeansCluster.COV_DIAG:
            self._X2 = np.zeros([self._K, self._D])

    def UpdateParameters(self) -> (float, list):
        if self._train_mode != self.TRAIN_VAR_INSIDE:
            raise Exception('train mode is not TRAIN_VAR_INSIDE.')
        for k in range(self._K):
            if self._X0[k] == 0:
                continue
            self.Mu[k,:] = self._X1[k,:] / self._X0[k]
            if self._cov_mode in [KmeansCluster.COV_FULL, KmeansCluster.COV_DIAG]:
                self.Sigma = self._X2 /self._X0[k]
        return self._loss, self._X0


def kmeans_clustering(X:np.ndarray, mu_init:np.ndarray, max_it:int = 20, save_ckpt:bool = False):
    """Run k-means clustering.

    Args:
        X (np.ndarray): vector samples (N, D)
        mu_init (np.ndarray): initial mean vectors(K, D)
        max_it (int, optional): Iteration steps. Defaults to 20.

    Returns:
        _type_: _description_
    """
    K, Dim = mu_init.shape
    N, Dim2 = X.shape
    if Dim != Dim2:
        raise Exception('dimmension is not compatible.')
    kmeansparam = KmeansCluster(K, Dim, trainvars='inside')
    kmeansparam.Mu = mu_init
    cost_history = []
    alignment_history = []
    pbar = tqdm(range(max_it), desc="kmeans", postfix="postfix", ncols=80)
    for it in pbar:
        for n in range(N):
            kmeansparam.PushSample(X[n,:])
        loss, align_dist = kmeansparam.UpdateParameters()
        kmeansparam.ClearTrainigVariables()
        #print(loss/N, kmeansparam.Mu)
        logger.info("iteration %d loss %f", it, loss/N)
        cost_history.append(loss)
        alignment_history.append(align_dist)
    return kmeansparam, cost_history


def main(args):
    print(args.input_file)
    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)
        print(data.keys())
        X = data['sample']
        print('model type: ', data.get('model_type'))
        param = data['model_param']
        mu_init = param.Mu
        
    if args.num_cluster > 0:
        Dim = X.shape[1]
        np.random.seed(args.random_seed)
        mu_init = np.random.randn(args.num_cluster, Dim)

    kmeansparam, cost_history = kmeans_clustering(X, mu_init)
    #kmeansparam, cost_history = train_light(X, mu_init)
    print('Mu:', kmeansparam.Mu)
    print('Sigma:', kmeansparam.Sigma)

    out_pngfile = "distortion.png"
    fig = kmeans_plot.plot_distortion_history(cost_history)
    fig.savefig(out_pngfile)
    logger.info('out: %s', out_pngfile)

    R = kmeansparam.get_alignment(X)
    out_file = 'out_kmeans.pickle'
    with open(out_file, 'wb') as f:
        pickle.dump({'model': kmeansparam,
                    'history': cost_history,
                    'iteration': len(cost_history),
                    'alignment': R},
                    f)
    print(sum(R==1))
    logger.info('out: %s', out_file)


def set_parser():
    parser = argparse.ArgumentParser(
                    prog='K-means clustering',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('input_file', type=str, help='sample data in pickle')
    parser.add_argument('--random_seed', type=int, help='random seed', default=0)
    parser.add_argument('-k', '--num_cluster', type=int,
                        help='number of cluster. centroid are initialized by randn', default=0)
    return parser

if __name__ == '__main__':
    logging.basicConfig(filename='kmeans.log',\
        format='%(asctime)s [%(levelname)s] %(name)s %(message)s',\
        datefmt='%m/%d/%Y %I:%M:%S', \
        level=logging.INFO)

    parser = set_parser()
    args = parser.parse_args()
    main(args)
