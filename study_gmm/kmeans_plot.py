import numpy as np
import matplotlib.pyplot as plt

from kmeans import KmeansCluster

def kmeans_clustering_orig(X:np.ndarray, mu_init:np.ndarray, max_it:int = 20, save_ckpt:bool = False):
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
#        if it < (2*2):
#            print(int(floor(it/2)), it%2)
#            ax =axes0[it]
            #ax =axes0[floor(it/2), it%2]
            #show_prm(ax, X, R, mu, X_col, KmeansParam)
#            show_prm(ax, X, R, kmeansparam.Mu)
#            ax.set_title("iteration {0:d}".format(it + 1))
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


def plot_distortion_history(cost_history: list):
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
