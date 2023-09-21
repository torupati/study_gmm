from study_gmm.hmm import HMM, print_state_obs
from study_gmm.sample_generator import sampling_from_hmm
import numpy as np

def test_viterbi_search():
    """
    test viterbi_search
    """
    # Assume there are 3 observation states, 2 hidden states, and 5 observations
    M = 2
    D = 4
    hmm = HMM(M, D)
    hmm.init_state = np.array([0.5, 0.5])
    hmm._log_init_state = np.log(hmm.init_state)
    hmm.state_tran = np.array([[0.1, 0.9],
                               [0.5, 0.5]])
    hmm.obs_prob = np.array([[0.5, 0.5, 0.0, 0.0],
            [0.00, 0.00, 0.5, 0.5]])
    print(f"hmm={hmm}")

    #observations = [0, 1, 2, 3, 2, 0, 2, 0]
    st_orig, obs = sampling_from_hmm([10], hmm)
    print("observations({})={}".format(len(obs), obs))
    st, ll = hmm.viterbi_search(obs)
    print('st (', len(st), ')= ', st )
    print_state_obs(obs, st)
    print(st_orig)
#    print( viterbi_path(priors, transmat, obslik, scaled=False, ret_loglik=True) )#=> (array([0, 1, 1, 1, 0]), -8.0120386579275227)

def test_viterbi_search2():
    """
    test viterbi_search
    """
    # Assume there are 3 observation states, 2 hidden states, and 5 observations
    M = 3
    D = 4
    hmm = HMM(M, D)
    hmm.init_state = np.array([0.5, 0.3, 0.2])
    hmm.state_tran = np.array([[0.7, 0.2, 0.1],
                               [0.2, 0.0, 0.8],
                               [0.3, 0.6, 0.1]])
    hmm.obs_prob = np.array([[0.5, 0.3, 0.1, 0.1],
            [0.10, 0.10, 0.30, 0.50],
            [0.30, 0.30, 0.30, 0.10]])
    print(f"hmm={hmm}")

    counts = 0
    total_counts = 0
    for samp_idx in range(10):
        st_orig, obs = sampling_from_hmm([200], hmm)
        #print("observations({})={}".format(len(obs), obs))
        st, ll = hmm.viterbi_search(obs)
        #print('st (', len(st), ')= ', st )
        #print_state_obs(obs, st)
        #print(st_orig)
        print('idx   org  est')
        for _i, (s0, s1, o) in enumerate(zip(st_orig, st, obs)):
            print(f'i={_i:03d} {s0}    {s1}  o={o}')
            if s0 == s1:
                counts += 1
            total_counts += 1
        print(counts, total_counts)
#    print( viterbi_path(priors, transmat, obslik, scaled=False, ret_loglik=True) )#=> (array([0, 1, 1, 1, 0]), -8.0120386579275227)

#test_viterbi_search()
test_viterbi_search2()