from study_gmm.hmm import HMM, print_state_obs
import numpy as np

def test_viterbi_search():
    """
    test viterbi_search
    """
    # Assume there are 3 observation states, 2 hidden states, and 5 observations
    M = 2
    D = 4
    hmm = HMM(M, D)
    hmm._log_init_state = np.log(np.array([0.5, 0.5]))
    hmm.state_tran = np.array([[0.1, 0.9],
                               [0.5, 0.5]])
    hmm.obs_prob = np.array([[0.5, 0.5, 0.0, 0.0],
            [0.00, 0.00, 0.5, 0.5]])
    print(f"hmm={hmm}")

    observations = [0, 1, 2, 3, 2, 0, 2, 0]
    print("observations({})={}".format(len(observations), observations))
    st, ll = hmm.viterbi_search(observations)
    print('st (', len(st), ')= ', st )
    print_state_obs(observations, st)
#    print( viterbi_path(priors, transmat, obslik, scaled=False, ret_loglik=True) )#=> (array([0, 1, 1, 1, 0]), -8.0120386579275227)
