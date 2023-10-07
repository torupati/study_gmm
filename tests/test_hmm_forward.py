from study_gmm.hmm import HMM, print_state_obs
from study_gmm.sample_generator import sampling_from_hmm
from study_gmm.hmm_plot import plot_gamma
import matplotlib.pyplot as plt

import numpy as np

def test_forward_backward():
    """
    test viterbi_search
    """
    # Assume there are 3 observation states, 2 hidden states, and 5 observations
    M = 3
    D = 4
    hmm = HMM(M, D)
    hmm.init_state = np.array([0.5, 0.5, 0.0])
    hmm._log_init_state = np.log(hmm.init_state)
    hmm.state_tran = np.array([[0.60, 0.35, 0.05],\
                               [0.01, 0.60, 0.39],
                               [0.30, 0.00, 0.70]])
    hmm.obs_prob = np.array([\
        [0.70, 0.10, 0.10, 0.10],
        [0.01, 0.09, 0.80, 0.10],
        [0.1, 0.45, 0.00, 0.45]])
    print(f"hmm={hmm}")

    obs = [0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 3, 2, 2, 2, 3, 2, 2, 1, 3, 3, 1, 1, 3, 0, 0, 0, 0]
    #np.random.seed(0)
    #st_orig, obs = sampling_from_hmm([10], hmm)
    print("observations({})={}".format(len(obs), obs))
    
    fig, axes = plt.subplots(2, 1)

    _gamma, _qsi, _logprob = hmm.forward_backward_algorithm_linear(obs)
    print('gamma', _gamma)
    print('qsi', _qsi)
    print(_logprob)
    #assert -11.524219471588987 
    plot_gamma(axes[0], _gamma)
    axes[0].set_title('Forward-Backward')

    _gamma, _qsi, _logprob = hmm.forward_viterbi(obs)
    print('gamma', _gamma)
    print('qsi', _qsi)
    print(_logprob)
    #assert -11.524219471588987
    axes[1].set_title('Viterbi search') 
    plot_gamma(axes[1],_gamma)
    
    fig.savefig('test_gamma.png')
 
#    st, ll = hmm.viterbi_search(obs)
#    print('st (', len(st), ')= ', st )
#    print_state_obs(obs, st)
#    print(st_orig)

