"""
K-means clustering
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, random, uint8 # definition, modules
from numpy import array, argmin, dot, ones, zeros, cov, savez # functions
from math import floor
import pickle

class KmeansCluster():
    """Definition of clusters for K-means altorithm
    """
    def __init__(self, K:int, D:int):
        """_summary_

        Args:
            K (int): number of cluster
            D (int): dimension of smaples
        """
        self._K = K
        self._D = D
        self.Mu = np.random.randn(K, D) #centroid
        self.Sig = np.ones((K, D))
        self.Pi = np.ones(K) / K # Cumulative probability

    @property
    def K(self) -> int:
        """Number of clusters

        Returns:
            int: _description_
        """
        return self._K

    @property
    def D(self) -> int:
        return self._D

    def update_Mu(self, k, val):
        assert k < self._K
        assert len(val) == self._D
        self.Mu[k,:] = val

    def distortion_measure(self, x:ndarray, r:ndarray) -> float:
        """
        Calculate distortion measure.

        Args
        ----
        x, training samples(N,D)
        r, alignment of sample(N, K)
        """
        J = 0.0
        n_sample = x.shape[0]
        if n_sample == 0:
            return 0.0
        for n in range(n_sample):
            dist = [sum(v*v for v in x[n, :] - self.Mu[k, :]) for k in range(self._K)]
            J = J + dot(r[n, :], dist)
        #   J = J + r[n, k] * ((x[n, 0] - mu[k, 0])**2 + (x[n, 1] - mu[k, 1])**2)
        return J/n_sample

    def get_alignment(self, x:ndarray) -> ndarray:
        """
        Args
        ----
        x[N, D]: samples
        mu[K, D]: centroids
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

    def get_centroid(self, x:ndarray, r:ndarray) -> ndarray:
        """_summary_

        Args:
            x (ndarray): input samples (N,D)
            r (ndarray): sample alignment to clusters (N,K)

        Returns:
            ndarray: updated centroid(K,D)
        """
    #mu = np.zeros((K, x.shape[1])) # (K, D)
    #for k in range(K):
    #    mu[k, 0] = np.sum(r[:, k] * x[:, 0]) / np.sum(r[:, k])
    #    mu[k, 1] = np.sum(r[:, k] * x[:, 1]) / np.sum(r[:, k])
        _mu = dot(r.transpose(), x)
        for k in range(self._K):
            if sum(r[:,k]) < 0.01:
                # this cluster has no sample aligned.
                continue
            _mu[k,:] = _mu[k,:] / sum(r[:,k])
        return _mu

#X_col = ['cornflowerblue', "orange", 'black', 'white', 'red']

def generate_samples(n_sample: int, kmeans_param) -> ndarray:
    """
    シミュレーション用のデータを作成する.
    Parameters
    ----------
    num, サンプル数
    dim, 特徴量の次元
    """
    X = np.zeros((n_sample, kmeans_param.D))
    counts = np.random.multinomial(n_sample, kmeans_param.Pi)
    i = 0
    k = 0
    while i < n_sample:
        for j in range(counts[k]):
            X[i+j,:] = kmeans_param.Mu[k,:] \
                + np.dot(np.diag(kmeans_param.Sig[k,:]), random.randn(kmeans_param.D))
        i += counts[k]
        k += 1
    return X

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

def show_prm(ax, x, r, mu, kmeans_param_ref:dict = {}):
    K = mu.shape[0]
    for k in range(K):
        # データ分布の描写
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


def generate_sample_kmeans_cluster():
    K, D = 4, 2
    param = KmeansCluster(K, D)

    #param.Mu = np.array([[-0.5, -0.5], [0.5, 1.0], [1.0, -0.5]])
    #param.Sig = array([[.7, .7], [.8, .3], [.3, .8]])
    #param.Pi = array([0.4, 0.8, 1])

    param.Mu = array([[3.0, 3.0], [0.0, 2.0], [2.0, -3.5], [-3.0, 0.0]])
    param.Sig = array([[1.0, 1.0], [0.3, 0.1], [0.6, 0.5], [1.0, 0.5]])
    param.Pi = array([1/K]*K)
    return param


def main():

    random.seed(1)
    N = 300

    #kmeansparam = KmeansCluster(6, 20)
    kmeansparam_true = generate_sample_kmeans_cluster()
    X = generate_samples(N, kmeansparam_true)
    K = kmeansparam_true.K
    Dim = kmeansparam_true.D

    fig0, ax0 = plt.subplots(1,1, figsize=(8,8))
    plot_training_samples(ax0, X)
    fig0.savefig("samples.png")


    X_range_x1 = [min(X[:,0]), max(X[:,0])]
    X_range_x2 = [min(X[:,1]), max(X[:,1])]

    npz_file = f"new_sample_data_D{kmeansparam_true.D}_K{kmeansparam_true.K}.npz"
    np.savez(npz_file, X=X, ModelParam = kmeansparam_true)
    print('save data in file: ', npz_file)


    # Mu と R の初期化
    mu_init = random.randn(K+1, Dim)
    kmeansparam = KmeansCluster(K+1, Dim)
    kmeansparam.Mu = mu_init
    max_it = 10
    R = None
    cost_history = []
    fig0, axes0 = plt.subplots(1, 4, figsize=(16, 4))
    for it in range(0, max_it):
        R_new = kmeansparam.get_alignment(X) # alignment to all sample to all clusters
        if it > 2:
            num_updated = int((R_new - R).sum())
            print('num_updated = ', R_new - R)
            #if num_updated == 0:
            #    break
        print(R_new)
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
        ckpt_file = f"kmeans_itr{it}.ckpt"
        with open(ckpt_file, 'wb') as f:
            pickle.dump({'model': kmeansparam,
                    'loss': loss,
                    'iteration': it,
                    'alignment': R}, f)
            print(ckpt_file)
    fig0.savefig("iteration.png")

    print("centroid:", np.round(kmeansparam.Mu, 5)),

    print(kmeansparam.distortion_measure(X, kmeansparam.get_alignment(X)))

    fig1, ax = plt.subplots(1, 1, figsize=(8,4))
    ax.plot(range(0, len(cost_history)), cost_history, color='black', linestyle='-', marker='o')
    ax.set_yscale("log")
    ax.set_xlim([0,len(cost_history)])
    ax.grid(True)
    ax.set_xlabel("Iteration Step")
    ax.set_ylabel("Loss")
    fig1.savefig("distortion.png")

if __name__ == '__main__':
    main()

