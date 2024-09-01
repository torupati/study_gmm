import numpy as np
import matplotlib.pyplot as plt

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
