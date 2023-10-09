from study_gmm.hmm import HMM, print_state_obs, hmm_baum_welch
from study_gmm.sample_generator import sampling_from_hmm
from study_gmm.hmm_plot import plot_gamma
import matplotlib.pyplot as plt
import copy

import numpy as np

def test_forward_backward():
    """

    """
    # Assume there are 3 observation states, 2 hidden states, and 5 observations
    M = 3
    D = 4
    hmm = HMM(M, D)
    hmm.init_state = np.array([0.34, 0.33, 0.33])
    hmm._log_init_state = np.log(hmm.init_state)
    hmm.state_tran = np.array([[0.60, 0.35, 0.05],\
                               [0.01, 0.60, 0.39],
                               [0.30, 0.00, 0.70]])
    hmm.obs_prob = np.array([\
        [0.70, 0.10, 0.10, 0.10],
        [0.01, 0.09, 0.80, 0.10],
        [0.1, 0.45, 0.00, 0.45]])
    hmm_orig = copy.deepcopy(hmm)
    print(f"hmm={hmm}")

#    N = 200
#    T = 50
    N = 100
    T = 90
    sample_lengths = [T] * N
    states_orig, obs_seqs = sampling_from_hmm(sample_lengths, hmm)
    #print(obs_seqs)

    hmm.obs_prob = np.array([\
        [0.4, 0.2, 0.20, 0.20],
        [0.20, 0.20, 0.4, 0.20],
        [0.25, 0.25, 0.25, 0.25]])
    hmm_init = copy.deepcopy(hmm)
    ll_history = hmm_baum_welch(hmm, obs_seqs)
    print(ll_history)
    import json
    import matplotlib.pyplot as plt
    with open('out_test_hmm_baumwelch.json', 'w') as f:
        json.dump(ll_history, f, indent=2)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ave_ll_history = [ll/n for ll, n in zip(ll_history['log_likelihood'], ll_history['total_obs_num'])]
    ax.plot(ll_history['step'], ave_ll_history)
    ax.set_ylabel(r'$E[\log P(X)]$')
    ax.set_xlabel('iteration steps')
    ax.grid(True)
    ax.set_xlim((-2, len(ll_history['step'])))
    fig.set_tight_layout(True)
    out_figname = 'out_hmm_training_loglikelihood.png'

    fig.savefig(out_figname)

    #
    state_name = ['A dominant', 'B dominant', 'Transient']
    state_labels = ['A boom', 'B boom', 'Trans.']
    names = ['A', 'B', 'C', 'D']
    fig, axs = fig, axs = plt.subplots(1, M, figsize=(9, 3), sharey=True)
    for b, b0, st_name, ax in zip(hmm.obs_prob, hmm_init.obs_prob, state_name, axs):
        print(b)
        ax.bar(names, b, alpha=0.75)
        ax.scatter(np.arange(0, len(names)), b0, alpha=0.75)
        ax.set_title(st_name)
        ax.set_ylim([0, 1.0])
    fig.savefig('out_hmm_training_outprob_dist.png') 
