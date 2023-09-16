"""Hidden Markov Models implementaton for learning purpose
"""

import numpy as np
from numpy import zeros, log, array
from numpy import ndarray
from typing import List, Tuple
import json

eps = 1.0E-128

class HMM:
    """
    HMM parameter definition
    """

    def __init__(self, M: int, D:int):
        """Define HMM parameter.

        Args:
            M (int): Number of hidden state
            D (int): Category number (dimension) of observation
        """
        self._M = M
        self._D = D
        self.state_priori \
            = np.array([1/M]*M) # initial state probability Pr(s[t=0] == i)
        self.state_tran \
            = np.ones((M,M)) * (1/M) # state transition probability, Pr(s[t+1]=j | s[t]=i)
        self.obs_prob \
            = np.zeros((M,D)) # state emission probability, Pr(y|s[t]=i)
        for m in range(M):
            self.obs_prob[m,:] = np.random.uniform(0,1,D)

        # training variables (keep sufficient statistics for parameter update)
        self._ini_state_stat = np.zeros(M)
        self._state_tran_stat = np.zeros((M,M))
        self._obs_count = np.zeros((M, D))
        self._training_count = 0
        self._training_total_log_likelihood = 0.0

    def __repr__(self) -> str:
        return f"HMM (#state={self._M}, #dim={self._D}) \n" \
            + f" initial state probability={self.state_priori}\n" \
            + f" transition probability={self.state_tran}" \
            + f" obs_prob={self.obs_prob}"

    def viterbi_search(self, obss:List[int]):
        """Viterbi search of discrete simble HMM

        - Finds the most-probable (Viterbi) path through the HMM states given observation.
        - Trellis (search space) is allocated in this method and release after the computation.

        Args:
            obss (List[int]): given observation sequence(descreat signal), y[t]

        Returns:
            _type_: _description_
        """
        # Note that this implementation (keeping all observation of each state, each time step) is naieve because
        # it is not necessary to keep at the same time and memory exhasting.
        # (1) log P(x[t]|s[t]) is only required at time step t in viterbi search
        # (2) Probability can be stored in log scale in advance.
        _log_obsprob = array([ log(self.obs_prob[:,z]+eps) for z in obss]).T # log likelihood of each state, each time step

        T = len(obss)
        _trellis_prob = np.zeros((self._M, T), dtype=float)
        _trellis_bp = np.zeros((self._M,T), dtype=int)

        _trellis_prob[:,0] = log(self.state_priori + eps) + _log_obsprob[:,0]
        _trellis_bp[:,0] = 0
        for t in range(1, T):# for each time step t
            for j in range(self._M): # for each state s[t]
                _probs \
                    = _trellis_prob[:,t-1] + log(self.state_tran[:,j]+eps) + _log_obsprob[j,t]
                # calculate path from each s[t-1]. _probs is array
                _trellis_bp[j,t] = _probs.argmax()
                _trellis_prob[j,t] = _probs[_trellis_bp[j,t]]
        # back traincing
        best_path:List[int] = []
        t, s = T-1, _trellis_prob[:,-1].argmax()
        while t >= 0:
            #print("back trace t={} s={}".format(t,s))
            best_path.append(s)
            s = _trellis_bp[s, t]
            t = t-1
        best_path.reverse()
        log_prob = _trellis_prob[:,-1].max() # max log P(x[1:T]|s[1:T]|) P(s[1:T])
        return best_path, log_prob

    def push_viterbi_trainining_statistics(self, obss:List[int]):
        """Add sufficient statistics from given observation.

        Args:
            obss (List[int]): _description_
        """
        T = len(obss)
        best_path, log_likelihood = self.viterbi_search(obss)
        #print("best path:", best_path)
        assert len(best_path) == len(obss)
        self._training_total_log_likelihood += log_likelihood

        _gamma1 = zeros([T, self._M]) # element is binary
        _gamma2 = zeros([T-1, self._M, self._M]) # g(t, s, s') element is binary
        for t, s in enumerate(best_path):
            _gamma1[t, s] = 1.0 # P(S[t]=s|X)
            if t < T-1:
                _gamma2[t,s, best_path[t+1]] = 1.0 # P(S[t]=s, s[t+1]=s'|X)
        #
        self._ini_state_stat = self._ini_state_stat + _gamma1[0]
        for t in range(T-1):
#            print('add\n ', _tran_count_time_step[t,:,:])
            self._state_tran_stat = self._state_tran_stat + _gamma2[t,:,:]
        for t in range(T):
            # make observation vector. 
            o = np.zeros(self._D)
            o[obss[t]] = 1
            #print('o.shape=', o.shape)
            #print('gamma=', _gamma1[t])
            #print('obs_count=', self._obs_count.shape)
            #print(_gamma1[t,0] * o)
            for _m in range(self._M):
                self._obs_count[_m,:] = self._obs_count[_m,:] + _gamma1[t,_m] *  o
        self._training_count += 1
        return True

    def update_parameters(self):
        self.state_priori = self._ini_state_stat / sum(self._ini_state_stat)
        #print('self._state_tran_stat=', self._state_tran_stat)
        for m in range(self._M): # normalize each state
            self.state_tran[m,:] = self._state_tran_stat[m,:]/sum(self._state_tran_stat[m,:])
            self.obs_prob[m,:] = self._obs_count[m,:]/sum(self._obs_count[m,:])
        #print('self.state_tran_stat=', self.state_tran)
        # training variables
        self._ini_state_stat = np.zeros(self._M)
        self._state_tran_stat = np.zeros((self._M,self._M))
        self._obs_coutn = np.zeros((self._M, self._D))
        self._training_count = 0
        ll = self._training_total_log_likelihood
        self._training_total_log_likelihood = 0.0
        return ll


def hmm_viterbi_training(hmm, obss_seqs):
    """

    Args:
        hmm (_type_): _description_
        obss_seqs (_type_): _description_
    """
    itr_count = 0
    while itr_count < 4:
        for x in obss_seqs:
            hmm.push_viterbi_trainining_statistics(x)
        total_likelihood = hmm.update_parameters()
        print("itr {} log-likelihood {}".format(itr_count, total_likelihood))
        print(f'hmm={hmm}')
        itr_count += 1
        #break

def print_state_obs(x, st):
    for i, (_x, _s) in enumerate(zip(x, st)):
        print(f't={i} s={st[i]} x={x[i]}')

def test_viterbi_search():
    """
    test viterbi_search
    """
    # Assume there are 3 observation states, 2 hidden states, and 5 observations
    M = 2
    D = 4
    hmm = HMM(M, D)
    hmm.state_priori = np.array([0.5, 0.5])
    hmm.state_tran = np.array([[0.5, 0.5],
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

def test_viterbi_training():
    training_data = [\
        [0, 1, 0, 0, 1, 2, 1, 1, 2, 2, 2, 3],
        [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
        [1, 0, 1, 1, 2, 4, 3, 0, 1],
        [2, 0, 1, 1, 1, 2, 3, 2, 1, 1]]
    print("N={}".format(len(training_data))) 
    M = 2
    D = 5
    hmm = HMM(M, D)
    hmm.state_priori = np.array([0.5, 0.5])
    hmm.state_tran = np.array([[0.5, 0.5],
                               [0.5, 0.5]])
    hmm.obs_prob = np.array([[0.5, 0.2, 0.2, 0.1, 0.0],
            [0.00, 0.1, 0.4, 0.4, 0.1]])
    print("emission probability: {}".format(hmm.obs_prob))
    #for i, x in enumerate(training_data):
    #    print(f'n={i} x={x}')
    #    print( hmm.viterbi_search(x) )
        #break
    ###
    hmm_viterbi_training(hmm, training_data)


if __name__=='__main__':
    #test_viterbi_search()
    test_viterbi_training()
