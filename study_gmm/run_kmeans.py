"""
K-means clustering

todo: full covariance
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, random, uint8 # definition, modules
from numpy import array, argmin, dot, ones, zeros, cov, savez # functions
from math import floor
from scipy.stats import multivariate_normal
import pickle
import argparse

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
            covariance_mode(str): "none"(default), "diag" or "full"
        Note if you want to change the number of clusters, new instance with desired cluster numbers should be created.
        """
        self._K = K
        self._D = D
        self.Mu = np.random.randn(K, D) #centroid
        #self.Pi = np.ones(K) / K # Cumulative probability
        print('kwargs', kwargs)

        if kwargs.get('covariance_mode','none') == 'full':
            self._cov_mode = self.COV_FULL
            self.Sigma = np.ones((K, D, D))
        elif kwargs.get('covariance_mode','none') == 'diag':
            self._cov_mode = self.COV_DIAG
            self.Sigma = np.ones((K,D))
        elif kwargs.get('covariance_mode','none') == 'none':
            self._cov_mode = self.COV_NONE
            self.Sigma = None
        else:
            raise Exception('invalid covariance mode')

        print(kwargs.get('trainvars','outside') == 'inside')
        if kwargs.get('trainvars','outside') == 'outside':
            self._train_mode = KmeansCluster.TRAIN_VAR_OUTSIDE
        elif kwargs.get('trainvars','outside') == 'inside':
            self._train_mode = KmeansCluster.TRAIN_VAR_INSIDE
            self._loss = 0.0
            self._X0_sum = np.zeros([K])   
            self._X1_sum = np.zeros([K, D])
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
        print('train_var_mode=', KmeansCluster.TRAIN_VAR_INSIDE, self._train_mode)
        self._train_mode

    def update_Mu(self, k, val):
        assert k < self._K
        assert len(val) == self._D
        self.Mu[k,:] = val

    def distortion_measure(self, x:ndarray, r:ndarray) -> float:        
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
            J = J + dot(r[n, :], dist)
        return J/n_sample

    def get_alignment(self, x:ndarray) -> ndarray:
        """
        Hard alocation of each sample to a cluster (or Gaussians)

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
            r[n, argmin([ sum(v*v for v in x[n, :] - self.Mu[k, :]) for k in range(self._K)])] = 1
            r[n, :] = r[n,:]/r[n,:].sum()
            #wk = [(x[n, 0] - mu[k, 0])**2 + (x[n, 1] - mu[k, 1])**2 for k in range(K)]
            #r[n, argmin(wk)] = 1
        return r

    def get_soft_alignment(self, x:ndarray) -> ndarray:
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

    def get_centroid(self, x:ndarray, r:ndarray) -> ndarray:
        """Calculate centroid from given alignment.

        Args:
            x (ndarray): input samples (N,D)
            r (ndarray): sample alignment to clusters (N,K)

        Returns:
            ndarray: updated centroid(K,D)
        """
        #  mu[k, :] = np.sum_n(r[n, k] * x[n,:]) / sum_{n,k'}(r[n, k'])
        _mu = dot(r.transpose(), x)
        for k in range(self._K):
            if sum(r[:,k]) < 0.01:
                # this cluster has no sample aligned.
                continue
            _mu[k,:] = _mu[k,:] / sum(r[:,k])
            # divied by number of aligned sample to the k-th cluster
        return _mu

    def PushSample(self, x: np.ndarray) -> bool:
        """_summary_

        Args:
            x (ndarray): traiing sample (D,)

        Returns:
            bool: _description_
        """
        #print('train_var_mode=', KmeansCluster.TRAIN_VAR_INSIDE, self._train_mode)
        if self._train_mode != self.TRAIN_VAR_INSIDE:
            return False
        costs = [ sum(v*v for v in x - self.Mu[k, :]) for k in range(self._K)]
        k_min = argmin(costs)
        self._loss += costs[k_min]
        self._X0_sum[k_min] += 1
        self._X1_sum[k_min,:] += x
        #print(k_min, self._X0_sum)
        return True

    def ClearTrainigVariables(self):
        if self._train_mode != self.TRAIN_VAR_INSIDE:
            return
        self._loss = 0.0
        self._X0_sum = np.zeros([self._K])
        self._X1_sum = np.zeros([self._K, self._D])

    def UpdateParameters(self) -> float:
        if self._train_mode != self.TRAIN_VAR_INSIDE:
            return
        for k in range(self._K):
            if self._X0_sum[k] == 0:
                continue
            self.Mu[k,:] = self._X1_sum[k,:] / self._X0_sum[k]
        return self._loss


def plot_training_samples(ax, x):
    X_range_x1 = [min(x[:,0]), max(x[:,0])]
    X_range_x2 = [min(x[:,1]), max(x[:,1])]
    ax.plot(x[:, 0], x[:, 1], marker='.',
                #markerfacecolor=X_col[k],
                markeredgecolor='k',
                markersize=6, alpha=0.5, linestyle='none')
    ax.set_xlim(X_range_x1)
    ax.set_ylim(X_range_x2)
    ax.grid("True")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")


def plot_distorion_history(cost_history: list):
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    ax.plot(range(0, len(cost_history)), cost_history, color='black', linestyle='-', marker='o')
    ax.set_yscale("log")
    ax.set_xlim([0,len(cost_history)])
    ax.grid(True)
    ax.set_xlabel("Iteration Step")
    ax.set_ylabel("Loss")
    return fig


def show_prm(ax, x, r, mu, kmeans_param_ref:dict = {}):
    #X_col = ['cornflowerblue', "orange", 'black', 'white', 'red']
    K = mu.shape[0]
    for k in range(K):
        ax.plot(x[r[:, k] == 1, 0], x[r[:, k] == 1, 1],
                marker='.',
                #markerfacecolor=X_col[k],
                markeredgecolor='k',
                markersize=6, alpha=0.5, linestyle='none')
    mu_ref = kmeans_param_ref.get("Mu")
    if mu_ref is not None:
        for k in range(K):
            ax.plot(mu_ref[k, 0], mu_ref[k, 1], marker='*',
                #markerfacecolor=X_col[k],
                markersize=8, markeredgecolor='k', markeredgewidth=2)
    for k in range(K):
        ax.plot(mu[k, 0], mu[k, 1], marker='o',
                #markerfacecolor=X_col[k],
                markersize=10,
                markeredgecolor='k', markeredgewidth=1)
    X_range_x1 = [min(x[:,0]), max(x[:,0])]
    X_range_x2 = [min(x[:,1]), max(x[:,1])]
    ax.set_xlim(X_range_x1)
    ax.set_ylim(X_range_x2)
    ax.grid(True)
    ax.set_aspect("equal")



def train_light(X:np.ndarray, mu_init:np.ndarray, max_it:int = 20, save_ckpt:bool = False):
    """Run k-means clustering.

    Args:
        X (np.ndarray): vector samples (N, D)
        mu_init (np.ndarray): initial mean vectors(K, D)
        max_it (int, optional): Iteration steps. Defaults to 20.

    Returns:
        _type_: _description_
    """
    K = mu_init.shape[0]
    N, Dim = X.shape
    kmeansparam = KmeansCluster(K, Dim, trainvars='inside')
    kmeansparam.Mu = mu_init
    cost_history = []
    for it in range(0, max_it):
        for n in range(N):
            kmeansparam.PushSample(X[n,:])
        loss = kmeansparam.UpdateParameters()
        kmeansparam.ClearTrainigVariables()
        #loss = kmeansparam.distortion_measure(X, R)
        print(loss/N)
        print(kmeansparam.Mu)
        cost_history.append(loss)
        input()
    return kmeansparam, cost_history


def kmeans_clustering(X:np.ndarray, mu_init:np.ndarray, max_it:int = 20, save_ckpt:bool = False):
    """Run k-means clustering.

    Args:
        X (np.ndarray): vector samples (N, D)
        mu_init (np.ndarray): initial mean vectors(K, D)
        max_it (int, optional): Iteration steps. Defaults to 20.

    Returns:
        _type_: _description_
    """
    K = mu_init.shape[0]
    Dim = X.shape[1]
    kmeansparam = KmeansCluster(K, Dim)
    kmeansparam.Mu = mu_init
    R = None # alignmnet
    cost_history = []
    fig0, axes0 = plt.subplots(1, 4, figsize=(16, 4))
    for it in range(0, max_it):
        R_new = kmeansparam.get_alignment(X) # alignment to all sample to all clusters
        #R_new = kmeansparam.get_soft_alignment(X) # not implemented yet
        #if it > 2:
        #    num_updated = (R_new - R) == np.zeros([N,K])
        #    print('num_updated = ', num_updated)
            #if num_updated == 0:
            #    break
        #print(R_new)
        R = R_new
        loss = kmeansparam.distortion_measure(X, R)
        cost_history.append(loss)
        if it < (2*2):
            print(int(floor(it/2)), it%2)
            ax =axes0[it]
            #ax =axes0[floor(it/2), it%2]
            #show_prm(ax, X, R, mu, X_col, KmeansParam)
            show_prm(ax, X, R, kmeansparam.Mu)
            ax.set_title("iteration {0:d}".format(it + 1))
        print(f'iteration={it} distortion={loss}')
        kmeansparam.Mu = kmeansparam.get_centroid(X, R)
        loss = kmeansparam.distortion_measure(X, R)
        cost_history.append(loss)
        if save_ckpt:
            ckpt_file = f"kmeans_itr{it}.ckpt"
            with open(ckpt_file, 'wb') as f:
                pickle.dump({'model': kmeansparam,
                    'loss': loss,
                    'iteration': it,
                    'alignment': R}, f)
            print(ckpt_file)
        # convergence check must be done
    fig0.savefig("iteration.png")

    print("centroid:", np.round(kmeansparam.Mu, 5))
    print(kmeansparam.distortion_measure(X, kmeansparam.get_alignment(X)))
    return kmeansparam, cost_history


def main(args):
    
    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)
        print(data.keys())
        X = data['sample']
        print('model type: ', data.get('model_type'))
        param = data['model_param']
        mu_init = param.Mu
        
    if args.num_cluster > 0:
        Dim = X.shape[1]
        random.seed(args.random_seed)
        mu_init = random.randn(args.num_cluster, Dim)

    kmeansparam, cost_history = kmeans_clustering(X, mu_init)
    #kmeansparam, cost_history = train_light(X, mu_init)

    out_pngfile = "distortion.png"
    fig = plot_distorion_history(cost_history)
    fig.savefig(out_pngfile)

    R = kmeansparam.get_alignment(X)
    out_file = 'out_kmeans.pickle'
    with open(out_file, 'wb') as f:
        pickle.dump({'model': kmeansparam,
                    'history': cost_history,
                    'iteration': len(cost_history),
                    'alignment': R},
                    f)
    print(out_file)
    print(sum(R==1))


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
    parser = set_parser()
    args = parser.parse_args()
    main(args)
