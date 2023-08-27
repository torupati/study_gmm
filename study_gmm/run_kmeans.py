import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, random, uint8 # definition, modules
from numpy import array, argmin, dot, ones, zeros, cov, savez # functions
from math import floor

random.seed(1)
N = 300
#X_range0 = [-3, 3]
#X_range1 = [-3, 3]
KmeansParam = {\
    "Mu": array([[-0.5, -0.5], [0.5, 1.0], [1.0, -0.5]]),  # 分布の中心(K, D)
    "Sig": array([[.7, .7], [.8, .3], [.3, .8]])  # 分布の分散(K, D)
}
Pi = array([0.4, 0.8, 1])  # 累積確率

KmeansParam = {\
    "Mu": array([[3.0, 3.0], [0.0, 2.0], [2.0, -3.5], [-3.0, 0.0]]),  # 分布の中心(K, D)
    "Sig": array([[1.0, 1.0], [0.3, 0.1], [0.6, 0.5], [1.0, 0.5]])  # 分布の分散(K, D)
}

class KmeansCluster():
    def __init__(self, K, D):
        self._K = K
        self._D = D
        self.Mu = np.random.randn(K, D) #centroid
        self.Sig = np.ones((K, D))
        self.Pi = np.ones(K) / K # Cumulative probability

    @property
    def K(self) -> int:
        return self._K

    @property
    def D(self) -> int:
        return self._D

    def update_Mu(self, k, val):
        assert k < self._K
        assert len(val) == self._D
        self.Mu[k,:] = val
#X_col = ['cornflowerblue', "orange", 'black', 'white', 'red']
#K = KmeansParam["Mu"].shape[0]
#assert(KmeansParam.get("Mu").shape == (K, Dim))
#assert(KmeansParam.get("Sig").shape == (K, Dim))
#assert(len(X_col) == K)


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


kmeansparam = KmeansCluster(6, 20)
X = generate_samples(N, kmeansparam)

#X = generate_samples(N, Dim, Pi, KmeansParam.get("Mu"), KmeansParam.get("Sig"))
X_range_x1 = [min(X[:,0]), max(X[:,0])]
X_range_x2 = [min(X[:,1]), max(X[:,1])]

npz_file = f"new_sample_data_D{kmeansparam.D}_K{kmeansparam.K}.npz"
np.savez(npz_file, X=X, ModelParam = KmeansParam)
print('save data in file: ', npz_file)
#savez("sampledata.npz", X=X, ModelParam = KmeansParam)

def plot_training_samples(ax, x):
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

fig0, ax0 = plt.subplots(1,1, figsize=(8,8))
plot_training_samples(ax0, X)
fig0.savefig("samples.png")

def kmeans_update_alignment(x:ndarray, mu:ndarray) -> ndarray:
    """
    parameters
    ----------
    x[N, D]: samples
    mu[K, D]: centroids
    """
    if len(x.shape) != 2:
        raise Exception("dimension error")
    N, D = x.shape
    r = np.zeros((N, K))
    for n in range(N):
        r[n, argmin([ sum(v*v for v in x[n, :] - mu[k, :]) for k in range(K)])] = 1
        #wk = [(x[n, 0] - mu[k, 0])**2 + (x[n, 1] - mu[k, 1])**2 for k in range(K)]
        #r[n, argmin(wk)] = 1
    return r


def kmeans_update_centroid(x:ndarray, r:ndarray) -> ndarray:
    """
    各クラスタの平均値を求める
    Parameters
    ----------
    x, training data (N, D)
    r, alignment information(K, D)
    """
    #mu = np.zeros((K, x.shape[1])) # (K, D)
    #for k in range(K):
    #    mu[k, 0] = np.sum(r[:, k] * x[:, 0]) / np.sum(r[:, k])
    #    mu[k, 1] = np.sum(r[:, k] * x[:, 1]) / np.sum(r[:, k])
    mu = dot(r.transpose(), x)
    for k in range(K):
        if sum(r[:,k]) < 0.01:
            continue
        mu[k,:] = mu[k,:] / sum(r[:,k])
    return mu


def distortion_measure(x:ndarray, r:ndarray, mu:ndarray) -> float:
    """
    Calculate distortion measure
    Parameters
    ----------
    x, training samples(N,D)
    r, alignment of sample(N, K)
    mu, cluster centroid(K, D)
    """
    J = 0.0
    for n in range(x.shape[0]):
        dist = [sum(v*v for v in x[n, :] - mu[k, :]) for k in range(K)]
        J = J + dot(r[n, :], dist)
    #J = 0
    #for n in range(x.shape[0]):
        #for k in range(K):
        #   J = J + r[n, k] * ((x[n, 0] - mu[k, 0])**2 + (x[n, 1] - mu[k, 1])**2)
    return J

# データの図示関数 ---------------------------
def show_prm(ax, x, r, mu, kmeans_param_ref:dict = {}):
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
    ax.set_xlim(X_range_x1)
    ax.set_ylim(X_range_x2)
    ax.grid(True)
    ax.set_aspect("equal")

K = kmeansparam.K
Dim = kmeansparam.D

# Mu と R の初期化
#mu_init = np.array([[-2, 1], [-2, 0], [-2, -1]])
#mu_init = np.array([[0, 1], [0, 0], [1, 0]])
#mu_init = np.array([[2, 0], [0, 0], [1, 0]])
#mu_init = ndarray( (kmeansparam.K, kmeansparam.D)) * 0.5
#mu_init = X.mean(0) + ndarray(K, Dim), cov(X.T))
mu_init = random.randn(K, Dim)

mu = mu_init
print("mu=", mu)
max_it = 10
cost_history = []
fig0, axes0 = plt.subplots(1, 4, figsize=(8, 16))
for it in range(0, max_it): # K-means 法
    R = kmeans_update_alignment(X, mu)
    loss = distortion_measure(X, R, mu)
    cost_history.append(loss)
    if it < (2*2):
        print(int(floor(it/2)), it%2)
        ax =axes0[it]
        #ax =axes0[floor(it/2), it%2]
        #show_prm(ax, X, R, mu, X_col, KmeansParam)
        show_prm(ax, X, R, mu)
        ax.set_title("iteration {0:d}".format(it + 1))
    #ax.set_xticks(range(X_range0[0], X_range0[1]), "")
    #ax.set_yticks(range(X_range1[0], X_range1[1]), "")
    print(f'iteration={it} distortion={loss}')
    mu = kmeans_update_centroid(X, R)
    loss = distortion_measure(X, R, mu)
    cost_history.append(loss)
fig0.savefig("iteration.png")

print("simultion mean:", np.round(KmeansParam.get("Mu"), 5))
print("centroid:", np.round(mu, 5)),

print("Training")
#print(np.round(DM, 5))
#print(distortion_measure(X, kmeans_update_alignment(X, KmeansParam.get("Mu")), KmeansParam.get("Mu")))
print(distortion_measure(X, kmeans_update_alignment(X, kmeansparam.Mu), kmeansparam.Mu))

fig1, ax = plt.subplots(1, 1, figsize=(8,4))
ax.plot(range(0, len(cost_history)), cost_history, color='black', linestyle='-', marker='o')
#ax.set_ylim(40, 80)
ax.set_yscale("log")
ax.set_xlim([0,len(cost_history)])
ax.grid(True)
ax.set_xlabel("Iteration Step")
ax.set_ylabel("Distortion")
fig1.savefig("distortion.png")

