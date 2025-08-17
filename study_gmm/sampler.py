import logging

logger = logging.getLogger(__name__)

import numpy as np

try:
    from .hmm import HMM
    logger.debug("Using relative import")
except ImportError:
    from  hmm import HMM
    logger.debug("Using absoloute import")
logger.debug(f"{__package__=}  {__name__=}  {__file__=}")

#from  kmeans import KmeansCluster
from .kmeans import KmeansCluster

np.random.seed(1)


def generate_sample_parameter(K:int = 4, D:int = 2, **kwargs):
    """Create D-dimensional K component mean and covariance.

    Returns:
        _type_: _description_
    """
    logger.info(f'{K=} {D=}')
    param = KmeansCluster(K, D)

    param.Pi = np.array([1/K]*K)

    if kwargs.get('PRESET', False):
        param.Mu = np.array([[3.0, 3.0],\
        [0.0, 2.0],\
        [2.0, -3.5],\
        [-3.0, 0.0]])

        param.Sigma = np.array([\
        [[1.0, 0.0],[0.0, 1.0]],\
        [[0.3, 0.1],[0.1, 0.1]], \
        [[0.6, -0.3],[-0.3,0.5]],
        [[1.0, 0.8],[0.8, 0.8]]])
        return param
    param.Mu = np.random.randn(K, D)
    param.Sigma = np.zeros((K, D, D))
    for k in range(K):
        param.Sigma[k, :, :] = np.eye(D)
    return param


def generate_samples(n_sample: int, kmeans_param) -> np.ndarray:
    """_summary_

    Args:
        n_sample (int): number of samples to be generated.
        kmeans_param (_type_): mean and covaraince

    Returns:
        np.ndarray: generated samples.
    """
    logger.info(f'{n_sample=} {kmeans_param=}')
    X = np.zeros((n_sample, kmeans_param.D))
    counts = np.random.multinomial(n_sample, kmeans_param.Pi)
    i = 0
    k = 0
    if len(kmeans_param.Sigma.shape) == 2:
        S = np.diag(kmeans_param.Sigma[k,:])
    elif len(kmeans_param.Sigma.shape) == 3:
        S = kmeans_param.Sigma
    # Prepare matrix L such as L*L = S
    L = [np.linalg.cholesky(S[k,:,:]) for k in range(kmeans_param.K)]
    while i < n_sample:
        for j in range(counts[k]):
            X[i+j,:] = kmeans_param.Mu[k,:] \
                + np.dot(L[k], np.random.randn(kmeans_param.D))
        i += counts[k]
        k += 1
    return X


def sample_markov_process(length:int, init_prob, tran_prob):
    """_summary_

    Args:
        length (int): length of state sequence
        init_prob (_type_): initial state probability, pi[i] = P(s[t=0]=i)
        tran_prob (_type_): state transition probability, a[i][j] = P(s[t]=j|s[t-1]=i)
    """
    #print(f'init_prob.shape={init_prob.shape} tran_prob.shape={tran_prob.shape}')
    if length < 1:
        raise Exception('Length must be larger than 1.')
    n_states = len(init_prob)
    assert tran_prob.shape == (n_states, n_states)
    s = [np.nan] * length
    s[0] = np.random.choice(n_states, p=init_prob)
    for t in range(1,length):
        s[t] = np.random.choice(n_states, p=tran_prob[s[t-1],:])
    return s


def sample_lengths(ave_len:int, num: int):
    """Determin lengths of sequences by possion distribution.

    Args:
        ave_len (int): average of sample length
        num (int): number of sequence to be generated
    """
    lengths = np.random.poisson(ave_len, num)
    if len(lengths[lengths == 0]) > 0:
        for i in np.where(lengths == 0):
            while True:
                v = np.random.poisson(ave_len, 1)[0]
                if v > 0:
                    lengths[i] = v
                    break
    return lengths

def sample_multiple_markov_process(num:int, init_prob, tran_prob):
    """
    Sampling multiple Markov processes with given model paremeters.

    Args:
        num (int): _description_
        init_prob (_type_): _description_
        tran_prob (_type_): _description_
    """
    assert num > 0
    lengths = sample_lengths(10, num)
    print('lengths=', lengths)
    x = []
    for _l in lengths:
        x1 = sample_markov_process(_l, init_prob, tran_prob)
        x.append(x1)
    return x

def sampling_from_hmm(sequence_lengths, hmm:HMM):
    """sample HMM output and its hidden states from given parameters

    Args:
        n_sequence (int): _description_
        hmm (HMM): _description_
    """
    out = []
    outdim_ids = hmm.D
    for _l in sequence_lengths:
        states = sample_markov_process(_l, hmm.init_state, hmm.state_tran)
        obs = []
        for s_t in states:
            # sample x from p(x|s[t])
            x = np.random.choice(outdim_ids, p=hmm.obs_prob[s_t,:])
            obs.append(x)
        out.append(obs)
    return states, out


