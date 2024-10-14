import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_gamma(ax, _gamma, state_labels: list = []):
    
    ax.imshow(_gamma.transpose(), cmap='Reds', vmin=0, vmax=1)
    ax.set_xlabel('time index')

    T, M = _gamma.shape
    if len(state_labels) == M:
        ax.set_yticks([0, 1, 2], labels=state_labels)
    else:
        ax.set_ylabel('state index')
    ax.invert_yaxis()  # labels read top-to-bottom
    #ax.set_ylabel('state index')
    return ax

def plot_likelihood(ax, steps:list, log_likelihood:list, ylabel:str):
    ax.plot(steps, log_likelihood)
    ax.grid(True)
    ax.set_xlabel('iteration steps')


def plot_checkpoint_dir(ckpt_file):
    """not implmeneted yet.

    Args:
        ckpt_file (_type_): _description_
    """

    state_name = ['A dominant', 'B dominant', 'Transient']
    names = ['A', 'B', 'C', 'D']
    with open(ckpt_file, ) as f:
        model = pickle.load(f)
        hmm = model.get('model', None)
        model_type = model.get('model_type', '')
        #                     'total_likelihood': total_likelihood,
        #                     'total_sequence_num': len(obss_seqs),
        #                     'total_obs_num': total_obs_num,
        #                     'iteration': itr_count},
    M = len(state_name)
    fig, axs = plt.subplots(1, M, figsize=(9, 3), sharey=True)
    for b, st_name, ax in zip(hmm.obs_prob, state_name, axs):
        ax.bar(names, b, alpha=0.75)
        ax.set_title(st_name)
        ax.set_ylim([0, 1.0])
    fig.savefig('hmm_outprob_dist.png')


def plot_categorical(ax, values: list[int], names: list[str] = None):
    D = len(values)
    t = np.arange(0, D)
    ax.bar(names, values, 
            bins=D, range=(0, D), density=True, align='left', rwidth=0.75, label="Sample", alpha=0.75)
    ax.xticks(t)
    
#plt.savefig('bino.png')
#plt.show()


