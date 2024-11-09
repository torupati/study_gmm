"""
K-means clustering Implementation
"""
import logging

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class KmeansException(Exception):
    def __init__(self, arg=""):
        self.arg = arg

class InvalidParameterSetting(KmeansException):
    def __str__(self):
        return (
            f"{self.arg=}"
        )

class InsideError(KmeansException):
    def __str__(self):
        return (
            f"{self.arg=}"
        )

class InputSampleError(KmeansException):
    def __str__(self):
        return (
            f"{self.arg=}"
        )

class KmeansCluster():
    """class
    Definition of clusters for K-means altorithm
    """
    COV_NONE = 0
    COV_FULL = 1
    COV_DIAG = 2

    TRAIN_VAR_OUTSIDE = 0
    TRAIN_VAR_INSIDE = 1

    DISTANCE_LINEAR_SCALE = 0
    DISTANCE_LOG_SCALE = 1
    DISTANCE_KL_DIVERGENCE = 2

    def __init__(self, K: int, D: int, **kwargs):
        """Initialize instance.

        Args:
            K (int): number of cluster. Fixed.
            D (int): dimension of smaples. Fixed.

            covariance_mode(str): "none"(default), "diag" or "full" (optional)
            trainvars(str): "outside"(default) or "inside" (optional)

        Raises:
            Exception: wrong covariance mode input
            Exception: wrong training variable mode input

        Note:
        If you want to change the number of clusters, new instance with desired
        cluster numbers should be created.
        """
        self._K = K
        self._D = D
        self.Mu = np.random.randn(K, D)  # centroid
        if kwargs.get('covariance_mode', 'diag') == 'diag':
            self._cov_mode = KmeansCluster.COV_DIAG
            self.Sigma = np.ones((K, D))
        elif kwargs['covariance_mode'] == 'full':
            self._cov_mode = KmeansCluster.COV_FULL
            self.Sigma = np.ones((K, D, D))
        elif kwargs['covariance_mode'] == 'none':
            self._cov_mode = KmeansCluster.COV_NONE
            self.Sigma = None
        else:
            raise InvalidParameterSetting('covariance mode is wrong in constructor.')

        # define training variables
        #print(kwargs.get('trainvars', 'outside') == 'outside')
        if kwargs.get('trainvars', 'outside') == 'outside':
            self._train_mode = KmeansCluster.TRAIN_VAR_OUTSIDE
        elif kwargs['trainvars'] == 'inside':
            self._train_mode = KmeansCluster.TRAIN_VAR_INSIDE
            self._loss = 0.0
            self._X0 = np.zeros([K], dtype=np.uint32)
            self._X1 = np.zeros([K, D])
            if self._cov_mode == KmeansCluster.COV_NONE:
                self._X2 = None
            elif self._cov_mode == KmeansCluster.COV_FULL:
                self._X2 = np.zeros([K, D, D])
            elif self._cov_mode == KmeansCluster.COV_DIAG:
                self._X2 = np.zeros([K, D])
        else:
            raise InvalidParameterSetting('training variable mode.')

        if kwargs.get('dist_mode', 'linear') == 'linear':
            self._dist_mode = KmeansCluster.DISTANCE_LINEAR_SCALE
        elif kwargs['dist_mode'] == 'log':
            self._dist_mode = KmeansCluster.DISTANCE_LOG_SCALE
        elif kwargs['dist_mode'] == 'kldiv':
            self._dist_mode = KmeansCluster.DISTANCE_KL_DIVERGENCE
        else:
            raise InvalidParameterSetting('distance mode')

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

    @property
    def DistanceType(self) -> str:
        name_list =  {self.DISTANCE_LINEAR_SCALE: "linear", 
                      self.DISTANCE_LOG_SCALE: "log",
                      self.DISTANCE_KL_DIVERGENCE: "kldiv"}
        return name_list[self._dist_mode]

    def __repr__(self):
        return '{n} K:{k} D:{d}'.format(n=self.__class__.__name__,
                                          k=self._K, d=self._D)\
                + ' dist={d2}'.format(d2=self.DistanceType)

    @property
    def covariance_mode(self):
        return self._cov_mode

    @property
    def train_vars_mode(self) -> str:
        return list(["outside", 'inside'])[self._train_mode]

    def distortion_measure(self, x: np.ndarray, r: np.ndarray) -> float:
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
            dist = None
            if self._dist_mode == KmeansCluster.DISTANCE_LINEAR_SCALE:
                dist = [sum(v*v for v in x[n, :] - self.Mu[k, :]) for k in range(self._K)]
            elif self._dist_mode == KmeansCluster.DISTANCE_LOG_SCALE:
                dist = [sum(v*v for v in np.log(x[n, :]) - np.log(self.Mu[k, :])) for k in range(self._K)]
            elif self._dist_mode == KmeansCluster.DISTANCE_KL_DIVERGENCE:
                dist = [KmeansCluster.KL_divergence(x[n, :], self.Mu[k, :]) for k in range(self._K)]
            if dist is not None:
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
            raise InputSampleError(f"input shape is wrong {x.shape=}")
        N = x.shape[0]
        r = np.zeros((N, self._K), dtype=np.uint16)
        for n in range(N):
            costs = None
            if self._dist_mode == KmeansCluster.DISTANCE_LINEAR_SCALE:
                costs = [sum([v*v for v in (x[n, :] - self.Mu[k, :])]) for k in range(self._K)]
            elif self._dist_mode == KmeansCluster.DISTANCE_LOG_SCALE:
                costs = [sum([v*v for v in (np.log(x[n, :]) - np.log(self.Mu[k, :]))]) for k in range(self._K)]
            elif self._dist_mode == KmeansCluster.DISTANCE_KL_DIVERGENCE:
                costs = [KmeansCluster.KL_divergence(x[n, :], self.Mu[k, :]) for k in range(self._K)]
            if costs is not None:
                r[n, np.argmin(costs)] = 1
            #r[n, np.argmin([ sum(v*v for v in x[n, :] - self.Mu[k, :]) for k in range(self._K)])] = 1
            r[n, :] = r[n,:]/r[n,:].sum()
            #wk = [(x[n, 0] - mu[k, 0])**2 + (x[n, 1] - mu[k, 1])**2 for k in range(K)]
            #r[n, argmin(wk)] = 1
        return r

    def PushSample(self, x: np.ndarray) -> (int, float):
        """Push one training sample to inner training variables

        Args:
            x (ndarray): traiing sample (D,)

        Returns:
            int: aligned cluster's index (between 0 and K-1)
            float: loss of this sample
        """
        if self._train_mode != self.TRAIN_VAR_INSIDE:
            raise Exception('train mode is not TRAIN_VAR_INSIDE.')
        if self._dist_mode == KmeansCluster.DISTANCE_LINEAR_SCALE:
            costs = [sum([v*v for v in (x - self.Mu[k, :])]) for k
                     in range(self._K)]
        elif self._dist_mode == KmeansCluster.DISTANCE_LOG_SCALE:
            costs = [sum([v*v for v in (np.log(x) - np.log(self.Mu[k, :]))])
                     for k in range(self._K)]
        elif self._dist_mode == KmeansCluster.DISTANCE_KL_DIVERGENCE:
            costs = [KmeansCluster.KL_divergence(x, self.Mu[k, :]) for k
                     in range(self._K)]
        else:
            raise InvalidParameterSetting("wrong distance model")
        if np.isinf(costs).any():
            if self._dist_mode == KmeansCluster.DISTANCE_LOG_SCALE:
                raise Exception(f'log(x)={np.log(x)}'
                                + f' log(mu)={np.log(self.Mu)} costs={costs}')
            raise Exception(f'wrong input in distance computation x={x} mu={self.Mu}'
                      + f'costs={costs}')

        k_min = np.argmin(costs)
        self._loss += costs[k_min]
        self._X0[k_min] += 1
        self._X1[k_min, :] += x
        if self._cov_mode == KmeansCluster.COV_DIAG:
            self._X2[k_min, :] = (x * x)
        elif self._cov_mode == KmeansCluster.COV_FULL:
            # NOT checked.
            self._X2[k_min, :, :] = (x.reshape(self._D, 1)
                                     * x.reshape(1, self._D))
        return k_min, costs[k_min]

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
            self.Mu[k, :] = self._X1[k, :] / self._X0[k]
            if self._cov_mode in [KmeansCluster.COV_FULL,
                                  KmeansCluster.COV_DIAG]:
                self.Sigma = self._X2 / self._X0[k]
            if self._dist_mode == KmeansCluster.DISTANCE_KL_DIVERGENCE:
                self.Mu[k, :] = self.Mu[k, :] / np.sum(self.Mu[k, :])
        return self._loss, self._X0

    @property
    def loss(self) -> float:
        return self._loss

    @staticmethod
    def KL_divergence(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """Calculate KL divergence beteen two vectors.

        Args:
            x (np.ndarray): _description_
            y (np.ndarray): _description_
        Note:
            input vectors must be non negative.

        Returns:
            float: _description_
        """
        _x = x / np.sum(x)
        _y = y / np.sum(y)
        xy_diff = np.log(_x) - np.log(_y)
        _kl_div = np.sum(_x * xy_diff)
        return _kl_div


def kmeans_clustering(X: np.ndarray, mu_init: np.ndarray, **kwargs):
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
        raise InsideError(f"Wrong dim. M: {Dim}  X: {Dim2}")

    max_it = kwargs.get('max_it', 20)
    #save_ckpt = kwargs.get('save_ckpt', False)
    dist_mode = kwargs.get('dist_mode', 'linear')

    kmeansparam = KmeansCluster(K, Dim, trainvars='inside',
                                dist_mode=dist_mode)
    logger.info('initialize: %s', str(kmeansparam))
    kmeansparam.Mu = mu_init
    cost_history = []
    align_history = []
    with logging_redirect_tqdm(loggers=[logger]):
        pbar = tqdm(range(max_it), desc="kmeans", postfix="postfix", ncols=80)
        for it in pbar:
            for n in range(N):
                _, _ = kmeansparam.PushSample(X[n, :])
            loss, align_dist = kmeansparam.UpdateParameters()
            kmeansparam.ClearTrainigVariables()
            #logger.info("iteration %d loss %f", it, loss/N)
            pbar.write(f"iteration {it} loss {loss/N}")
            cost_history.append(loss)
            align_history.append(align_dist)
            # Convergence validation
            if len(cost_history) > 1:
                cost_diff = cost_history[-2] - cost_history[-1]
                align_diff = [x - y for x, y in
                              zip(align_history[-1], align_history[-2])]
                assert cost_diff >= 0.0
                if 0.0 <= cost_diff < 1.0E-6 and np.sum(align_diff) < 1.0E-6:
                    logger.debug('iteration step=%d cost_diff = %f'
                                + 'alignment change=%s',
                                it, cost_diff, align_diff)
                    break
    return kmeansparam, cost_history
