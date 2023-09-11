import argparse
import pickle
import numpy as np
from run_kmeans import KmeansCluster

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
    L = [np.linalg.cholesky(S[k,:,:]) for k in range(kmeans_param.K)]
    while i < n_sample:
        for j in range(counts[k]):
            X[i+j,:] = kmeans_param.Mu[k,:] \
                + np.dot(L[k], np.random.randn(kmeans_param.D))
        i += counts[k]
        k += 1
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
