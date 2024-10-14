import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import matplotlib.pyplot as plt
from study_gmm.gmm import GaussianMixtureModel, train_gmm, mixutre_number_test, load_from_file
from study_gmm.gmm_plot import plot_loglikelihood_history

def main(args):
    X, Param = load_from_file(args.input_file)
    print(X.shape, Param.__dict__.keys())
    if args.max_mixnum > 0:
        mixutre_number_test(X, Param, args.max_mixnum)
        return

    N, D = X.shape[0], X.shape[1]
    K = Param.K
    Pi, Mu, Sigma = Param.Pi, Param.Mu, Param.Sigma
    max_it = 10
    gmm = GaussianMixtureModel(K, D)
    gmm.Mu = np.random.randn(K, D) # Param.Mu
    Gamma = np.c_[np.ones((N, 1)), np.zeros((N, K-1))]

    gmm.Sigma = Sigma

    assert(Gamma.shape[1] == Param.Mu.shape[0])

    gmm, loglikelihood_history = train_gmm(gmm, X)

    fig = plot_loglikelihood_history(loglikelihood_history)
    out_pngfile = "loglikelihood-gmm.png"
    fig.savefig(out_pngfile)
#fig.show(block=False)
    print(out_pngfile)

import argparse
def set_parser():
    parser = argparse.ArgumentParser(
                    prog='GMM',
                    description='Training by EM algorithm',
                    epilog='Text at the bottom of help')
    parser.add_argument('input_file', type=str, help='sample data in pickle')
    parser.add_argument('--random_seed', type=int, help='random seed', default=0)
    parser.add_argument('--max-mixnum', type=int, default = 0, \
        help='maximum mixture number (0 is not use this option)')
    return parser

if __name__ == '__main__':
    parser = set_parser()
    args = parser.parse_args()
    main(args)
