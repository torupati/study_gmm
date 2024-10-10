"""Hidden Markov Models implementaton for learning purpose
"""

import numpy as np
from numpy import zeros, log, array
from numpy import ndarray
from typing import List, Tuple
import json
import pickle
from os import path, makedirs

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
        self.init_state = np.array([1/M]*M) # initial state probability Pr(s[t=0]==i)
        self.state_tran \
            = np.ones((M,M)) * (1/M) # state transition probability, Pr(s[t+1]=j | s[t]=i)
        self.obs_prob \
            = np.zeros((M,D)) # state emission probability, Pr(y|s[t]=i)
        for m in range(M):
            self.obs_prob[m,:] = np.random.uniform(0,1,D)
            self.obs_prob[m,:] = self.obs_prob[m,:] / self.obs_prob[m,:].sum()

        # training variables (keep sufficient statistics for parameter update)
        self._ini_state_stat = np.zeros(M)
        self._state_tran_stat = np.zeros((M,M))
        self._obs_count = np.zeros((M, D))
        self._training_count = 0
        self._training_total_log_likelihood = 0.0

    def __repr__(self) -> str:
        return f"HMM (#state={self._M}, #dim={self._D}) \n" \
            + " [initial state probability]\n" \
            + f"{self.init_state}\n" \
            + " [transition probability]\n" \
            + f"{self.state_tran}\n" \
            + "observation probability\n" \
            + f" {self.obs_prob}"

    @property
    def M(self) -> int:
        """Number of hidden states

        Returns:
            int: number of hidden states
        """
        return self._M

    @property
    def D(self) -> int:
        """Number of category or dimension of observation

        Returns:
            int: dimension of observation
        """
        return self._D

    def randomize_parameter(self):
        _vals = np.random.uniform(size=self._M)
        self.init_state = _vals / sum(_vals) # _vals > 0 is garanteered.

        self.state_tran = np.random.uniform(size=(self._M, self._M))
        self.obs_prob = np.random.uniform(size=(self._M, self._D))
        for m in range(self._M):
            self.state_tran[m,:] = self.state_tran[m,:]/sum(self.state_tran[m,:])
            self.obs_prob[m,:] = self.obs_prob[m,:]/sum(self.obs_prob[m,:])

    def viterbi_search(self, obss):
        """Viterbi search of discrete simble HMM

        - Finds the most-probable (Viterbi) path through the HMM states given observation.
        - Trellis (search space) is allocated in this method and release after the computation.

        Args:
            obss (List[int]): given observation sequence(descreat signal), y[t]

        Returns:
            _type_: _description_
        """

        T = len(obss)
        # Note that this implementation (keeping all observation of each state, each time step) is naieve because
        # it is not necessary to keep at the same time and memory exhasting.
        # (1) log P(x[t]|s[t]) is only required at time step t in viterbi search
        # (2) Probability can be stored in log scale in advance.
        _log_obsprob = np.zeros((T, self._M))
        for t in range(T):
            x_t = np.zeros(self._D)
            x_t[obss[t]] = 1.0
            for s in range(self._M):
                _obs_prob = self.obs_prob[s,:]
                _obs_prob[_obs_prob < 1.0E-100] = 1.0E-100
                _log_obsprob[t,s] = np.dot(x_t, np.log(_obs_prob))

        _trellis_prob = np.ones((self._M, T), dtype=float) * np.log(eps) # log scale
        _trellis_bp = np.zeros((self._M,T), dtype=int)

        _log_init_state = []
        for x in self.init_state:
            v = -10000.0
            if x > 1.0E-300:
                v = np.log(x)
            _log_init_state.append(v) 
        _trellis_prob[:,0] = _log_init_state + _log_obsprob[0,:]
        _trellis_bp[:,0] = 0
        for t in range(1, T):# for each time step t
            for j in range(self._M): # for each state s[t-1]=i to s[t]=j
                # _probs[i] = P(x[1:t]|s[t-1]=i,s[t]=j)
                _probs \
                    = _trellis_prob[:,t-1] + log(self.state_tran[:,j]+eps) + _log_obsprob[t,j]
                #print('probs=', _probs)
                # calculate path from each s[t-1]. _probs is array
                _trellis_bp[j,t] = _probs.argmax()
                _trellis_prob[j,t] = _probs[_trellis_bp[j,t]]
                #print(f'back pointer[t={t}, s={j}]={_trellis_bp[j,t]}')
            #print('t=', t)
            #print('trellis=', _trellis_prob[:,t])
        #print(_trellis_prob)
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

    def forward_viterbi(self, obss:List[int]):
        """Push training sequence to get probability of latent varialble condition by input.

        Args:
            obss (List[int]): _description_
        Returns: Probability of latent state.
            gamma_1: g(t,s) = P(S[t]=s|X)
            gamma_1: g(t,s,s') = P(S[t]=s,S[t+1]=s'|X)
        """
        T = len(obss)
        best_path, log_likelihood = self.viterbi_search(obss)
        assert len(best_path) == len(obss)
        self._training_total_log_likelihood += log_likelihood

        _gamma1 = zeros([T, self._M]) # element is binary
        _gamma2 = zeros([T-1, self._M, self._M]) # g(t, s, s') element is binary
        for t, s in enumerate(best_path):
            _gamma1[t, s] = 1.0 # gamma(t,s)=P(S[t]=s|X)
            if t < T-1:
                _gamma2[t,s, best_path[t+1]] = 1.0 # gamma(t,s,s')=P(S[t]=s, s[t+1]=s'|X)
        #
        return _gamma1, _gamma2, log_likelihood


    def calc_logobss(self, obss):
        T = len(obss)
        _log_obsprob = np.zeros((T, self._M))
        for t in range(T):
            x_t = np.zeros(self._D)
            x_t[obss[t]] = 1.0
            for s in range(self._M):
                _log_obsprob[t,s] = np.dot(x_t, np.log(self.obs_prob[s,:]))
        return _log_obsprob

    def forward_algorithm(self, obsprob) -> (np.ndarray, np.ndarray):
        """HMM forward algorithm
        alpha[t,s] = P(s[t]|x[1],...,x[t])
        alpha_scale[t] = Sum_s[t] P(x[1],...,x[t], s[t])

        Args:
            obsprob (_type_): observation probabilities at time t, state s
        """
        T, _M = obsprob.shape
        assert _M == self._M
        _alpha_scale = np.ones(T) * np.nan
        _alpha = np.zeros((T, self._M), dtype=float) # linear scale, not log scale
        _alpha[0,:] = self.init_state * obsprob[0,:] # alpha[t=0,s] = pi[s] * b(x[t=0],s)
        _alpha_scale[0] = _alpha[0,:].sum()
        _alpha[0,:] = _alpha[0,:]/_alpha_scale[0]
        for t in range(1, T):# for each time step t = 1, ..., T-1
            for i in range(self._M):
                _alpha[t,i] = 0.0
                for _i0 in range(self._M):
                    _alpha[t,i] += _alpha[t-1,_i0] * self.state_tran[_i0,i] * obsprob[t,i]
#            _alpha[t,:] = (_alpha[t-1,:] @ self.state_tran) * _obsprob[t,:]
            _alpha_scale[t] = _alpha[t,:].sum()
            _alpha[t,:] = _alpha[t,:] / _alpha_scale[t]
            # P(s[t]=s | X[t]=x[t], S)
        return _alpha, _alpha_scale

    def calculate_prob(self, obss) -> np.ndarray:
        """Caluclate log probabilities of given observation

        b(t, m) = P(x[t] | State=m)

        Args:
            obss (_type_): _description_

        Returns:
            np.ndarray: (T, M)-shape array, log observation probabilities
        """
        T = len(obss)
        _obsprob = np.zeros((T, self._M))
        for t in range(T):
            for s in range(self._M):
                _obsprob[t,s] = self.obs_prob[s,obss[t]]
        return _obsprob

    def forward_backward_algorithm_linear(self, obss):
        """Push training sequence to get probability of latent varialble condition by input.

        Args:
            obss (List[int]): _description_
        Returns: Probability of latent state.
            gamma_1: g(t,s) = P(S[t]=s|X)
            gamma_1: g(t,s,s') = P(S[t]=s,S[t+1]=s'|X)
        """
        T = len(obss)
        #_obsprob = np.exp(self.calc_logobss(obss))
        _obsprob = self.calculate_prob(obss)
        _alpha, _alpha_scale = self.forward_algorithm(_obsprob)

        _log_prob = 0.0
        for t in range(T):
            _log_prob +=  np.log(_alpha_scale[t])# sum_s log P(x[1:T],s[T]=s)
        self._training_total_log_likelihood += _log_prob
        #print('alpha=', _alpha)

        _beta = np.ones((T, self._M)) * np.nan # linear scale, not log scale
        _beta[T-1,:] = np.ones(self._M)
        _g1 = np.ones((T, self._M)) * np.nan
        _g2 = np.ones((T-1, self._M, self._M)) * np.nan
        _g1[T-1,:] = _alpha[T-1,:] * _beta[T-1,:]
        #print('gamma=', _g1[T-1,:])
        #print(f'sum(gamma[t={T-1}])=', _g1[T-1,:].sum(), ' (last)')
        for t in range(T-1,0,-1):# for each time step t = T-1, ..., 0
            # P(s[t-1],s[t]=s,X[t:T]=x[t:T])
            for i in range(self._M):
                _beta[t-1,i] = 0.0
                for _j in range(self._M):
                    _beta[t-1,i] += self.state_tran[i,_j] * _obsprob[t,_j] * _beta[t,_j]
            #_b = (self.state_tran @ np.diag(_obsprob[t,:])) @ _beta[t,:]
            _beta[t-1,:] = _beta[t-1,:] / _alpha_scale[t]

            # merge forward probability and backward probability
            _g1[t-1,:] = _alpha[t-1,:] * _beta[t-1,:]
            assert (_g1[t-1,:].sum() - 1.0) < 1.0E-9
            #print(f'sum(gamma[t={t-1}])=', _g1[t-1,:].sum())
            #print(f'gamma[t={t-1}]=', _g1[t-1,:])
            for i in range(self._M):
                for j in range(self._M): # transition s[t-1] to s[t]
                    _g2[t-1,i,j] = _alpha[t-1,i] * self.state_tran[i,j] * _obsprob[t,j] * _beta[t,j]
            _g2[t-1,:,:] = _g2[t-1,:,:]  / _alpha_scale[t]
            #print('value=', (np.dot(_alpha[t-1,:], self.state_tran) * _obsprob[t-1,:]).shape)
            #print('gzi=', _g2[t-1,:,:])
            #print(f'sum(gzai[t={t-1}])', _g2[t-1,:,:].sum(axis=1))
            #input()
            for i in range(self._M):
                assert (_g1[t-1,i] - _g2[t-1,i,:].sum()) < 1.0E-06
        return _g1,_g2, _log_prob


    def push_sufficient_statistics(self, obss: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> int:
        """Update suffience statistics for parameters by given observation and latent state probability.
        This function is used in both Viterbi traning and Baum-Welch algorithm.

        Args:
            obss (numpy.ndarray): A shape-(T,D) array, observation X given
            g1 (numpy.ndarray): A shape-(T, M) array, gamma(t, s, s') = P(S[t]=s, S[t+1]=s'|X)
            g2 (numpy.ndarray): A shape-(T, M, M) array, gamma(t, s) = P(S[t]=s|X)
        """
        T = len(obss)
        self._ini_state_stat = self._ini_state_stat + g1[0]
        for t in range(T-1):
            self._state_tran_stat = self._state_tran_stat + g2[t,:,:]
        for t in range(T):
            # make observation vector.
            o_t = np.zeros(self._D)
            o_t[obss[t]] = 1
            for _m in range(self._M):
                self._obs_count[_m,:] = self._obs_count[_m,:] + g1[t,_m] *  o_t
        self._training_count += 1

    def update_parameters(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        _init_state = self._ini_state_stat / sum(self._ini_state_stat)
        _init_state[_init_state < eps] = eps # if probability is lower then eps, set eps to void log(0)
        self.init_state = _init_state
        #print('self._state_tran_stat=', self._state_tran_stat)
        for m in range(self._M): # normalize each state
            if sum(self._state_tran_stat[m,:]) > 0.0:
                self.state_tran[m,:] = self._state_tran_stat[m,:]/sum(self._state_tran_stat[m,:])
            if sum(self._obs_count[m,:]) > 0.0:
                self.obs_prob[m,:] = self._obs_count[m,:]/sum(self._obs_count[m,:])
        #print('self.state_tran_stat=', self.state_tran)

        # reset training variables
        self._ini_state_stat = np.zeros(self._M)
        self._state_tran_stat = np.zeros((self._M,self._M))
        self._obs_coutn = np.zeros((self._M, self._D))
        self._training_count = 0

        tll = self._training_total_log_likelihood
        self._training_total_log_likelihood = 0.0
        return tll


def hmm_viterbi_training(hmm, obss_seqs):
    """

    Args:
        hmm (_type_): _description_
        obss_seqs (_type_): _description_
    """
    itr_count = 0
    training_history={'step':[], 'log_likelihood':[]}
    prev_likelihood = np.nan
    while itr_count < 10:
        for x in obss_seqs:
            g1, g2, ll = hmm.forward_viterbi(x)
            hmm.push_sufficient_statistics(x,g1,g2)
        total_likelihood = hmm.update_parameters()
        print("itr {} E[logP(X)]={}".format(itr_count, total_likelihood/len(obss_seqs)))
        training_history['step'].append(itr_count)
        training_history['log_likelihood'].append(total_likelihood/len(obss_seqs))

        if itr_count > 0:
            assert prev_likelihood <= total_likelihood
        prev_likelihood = total_likelihood
        itr_count += 1
    return training_history

def hmm_baum_welch(hmm, obss_seqs, itr_limit:int = 100) -> dict:
    """HMM training using EM algorithm.

    Args:
        hmm (_type_): _description_
        obss_seqs (_type_): _description_
    """
    itr_count = 0
    _save_model = True
    outdir = 'models/checkpoints/'
    if _save_model:
        makedirs(outdir, exist_ok=True)
    ll_history={'step':[], 'log_likelihood':[], 'total_obs_num': [], 'total_seq_num': []}
    prev_likelihood = np.nan
    while itr_count < itr_limit:
        total_obs_num = 0
        for x in obss_seqs:
            _gamma, _xi, tll = hmm.forward_backward_algorithm_linear(x)
            hmm.push_sufficient_statistics(x, _gamma, _xi)
            total_obs_num += len(x)
        total_likelihood = hmm.update_parameters()
        print("itr {} E[logP(X)]={}".format(itr_count, total_likelihood/len(obss_seqs)))
        ll_history['step'].append(itr_count)
        ll_history['log_likelihood'].append(total_likelihood)
        ll_history['total_obs_num'].append(total_obs_num)
        ll_history['total_seq_num'].append(len(obss_seqs))
        # save model
        if _save_model and itr_count % 30 == 0:
            ckpt_file = path.join(outdir, f'hmm_checkpoint_{itr_count:06d}.ckpt')
            with open(ckpt_file, 'wb') as f:
                pickle.dump({'model': hmm,
                             'model_type': 'HMM',
                             'total_likelihood': total_likelihood,
                             'total_sequence_num': len(obss_seqs),
                             'total_obs_num': total_obs_num,
                             'iteration': itr_count},
                        f)
                print(ckpt_file)

        #print('------ after Baum welch trianing ------')
        #print(f'hmm={hmm}')
        if itr_count > 0:
            assert prev_likelihood <= total_likelihood
        prev_likelihood = total_likelihood
        itr_count += 1
    return ll_history


def print_state_obs(x, st):
    for i, (_x, _s) in enumerate(zip(x, st)):
        print(f't={i} s={st[i]} x={x[i]}')





#if __name__=='__main__':
#    ...

