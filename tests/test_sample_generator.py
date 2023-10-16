from study_gmm.sample_generator import sampling_from_hmm
from study_gmm.sample_generator import sample_markov_process
from study_gmm.hmm import HMM
import numpy as np


def test_generate_markov_process():

    M = 2
    D = 5
    hmm = HMM(M, D)
    _init_state = np.array([0.9, 0.1])
    state_tran = np.array([[0.1, 0.9],
                            [0.1, 0.9]])
    np.random.seed(1)
    s = sample_markov_process(10, _init_state, state_tran)
    assert s == [0, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    print(s)
