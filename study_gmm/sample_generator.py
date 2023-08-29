import argparse
import pickle
import numpy as np
from run_kmeans import generate_sample_kmeans_cluster

def generate_samples(n_sample: int, kmeans_param) -> np.ndarray:
    X = np.zeros((n_sample, kmeans_param.D))
    counts = np.random.multinomial(n_sample, kmeans_param.Pi)
    i = 0
    k = 0
    while i < n_sample:
        for j in range(counts[k]):
            X[i+j,:] = kmeans_param.Mu[k,:] \
                + np.dot(np.diag(kmeans_param.Sig[k,:]), np.random.randn(kmeans_param.D))
        i += counts[k]
        k += 1
    return X

def main(args):
    random.seed(1)

    kmeans_param = generate_sample_kmeans_cluster()
    X = generate_samples(args.N, kmeans_param)

    with open(args.out_file, 'wb') as f:
        pickle.dump({'model_param': kmeans_param,
                    'sample': X,
                    'model_type': 'KmeansClustering'}, f)
        print(args.out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='K-means clustering',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    #parser.add_argument('filename')
    parser.add_argument('N', type=int, help='number of sample')
    parser.add_argument('-o', '--out_file', type=str, \
        help='output file name(out.pickle)', default='out.pickle')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
