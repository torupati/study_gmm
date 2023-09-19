import argparse
import pickle
import numpy as np
from run_kmeans import KmeansCluster
from hmm import HMM

def generate_sample_parameter():
    K, D = 4, 2
    param = KmeansCluster(K, D)

    param.Mu = np.array([[3.0, 3.0],\
        [0.0, 2.0],\
        [2.0, -3.5],\
        [-3.0, 0.0]])

    param.Sigma = np.array([\
        [[1.0, 0.0],[0.0, 1.0]],\
        [[0.3, 0.1],[0.1, 0.1]], \
        [[0.6, -0.3],[-0.3,0.5]],
        [[1.0, 0.8],[0.8, 0.8]]])
    param.Pi = np.array([1/K]*K)
    return param


def generate_samples(n_sample: int, kmeans_param) -> np.ndarray:
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


def generate_markov_process(length:int, init_prob, tran_prob):
    """_summary_

    Args:
        length (int): length of state sequence
        init_prob (_type_): initial state probability, pi[i] = P(s[t=0]=i)
        tran_prob (_type_): state transition probability, a[i][j] = P(s[t]=j|s[t-1]=i)
    """
    print(f'init_prob.shape={init_prob.shape} tran_prob.shape={tran_prob.shape}')
    if length < 1:
        raise Exception('Length must be larger than 1.')
    n_states = len(init_prob)
    assert tran_prob.shape == (n_states, n_states)
    s = [np.nan] * length
    s[0] = np.random.choice(n_states, p=init_prob)
    for t in range(1,length):
        s[t] = np.random.choice(n_states, p=tran_prob[s[t-1],:])
    return s

def generate_length(ave_len:int, num: int):
    """_summary_

    Args:
        ave_len (int): _description_
        num (int): _description_
    """
    lengths = np.random.poisson(ave_len, num)
    if len(lengths[lengths == 0]) > 0:
        for i in np.where(length == 0):
            while True:
                v = np.random.poisson(ave_len, 1)[0]
                if v > 0:
                    lengths[i] = v
                    break
    return lengths

def generate_samples_hmm(sequence_lengths, hmm:HMM) -> np.ndarray:
    """_summary_

    Args:
        n_sequence (int): _description_
        hmm (HMM): _description_

    Returns:
        np.ndarray: _description_
    """
    X = None
    return X


def main(args):
    np.random.seed(1)

    kmeans_param = generate_sample_parameter()
    X = generate_samples(args.N, kmeans_param)

    with open(args.out_file, 'wb') as f:
        pickle.dump({'model_param': kmeans_param,
                    'sample': X,
                    'model_type': 'KmeansClustering'}, f)
        print(args.out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    #parser.add_argument('filename')
    parser.add_argument('N', type=int, help='number of sample')
    parser.add_argument('-o', '--out_file', type=str, \
        help='output file name(out.pickle)', default='out.pickle')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
