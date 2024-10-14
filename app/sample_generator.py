import pickle
import numpy as np
from study_gmm.sampler import generate_sample_parameter, generate_samples
from study_gmm.sampler import sample_lengths, sampling_from_hmm, sample_multiple_markov_process
from study_gmm.hmm import HMM

import logging

_logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S',
                    handlers=[
                        logging.FileHandler("sample_generator.log"),
                        logging.StreamHandler()
                    ],
                    level=logging.INFO)


def main_gmm(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    _logger.info(f'{args=}')
    kmeans_param = generate_sample_parameter(args.cluster, args.dimension)
    _logger.info(f'{kmeans_param=}')

    X = generate_samples(args.N, kmeans_param)
    _logger.info(f'generated sample size: {X.shape=}')

    with open(args.out_file, 'wb') as f:
        pickle.dump({'model_param': kmeans_param,
                    'sample': X,
                    'model_type': 'KmeansClustering'}, f)
        _logger.info(f'output: {args.out_file}')


def main_hmm(args):
    _logger.info(f'{args=}')

    M = 2
    D = 5
    hmm_param = HMM(2, 5)
    hmm_param.init_state = np.array([0.1, 0.9])
    hmm_param.state_tran = np.array([[0.9, 0.1],
                           [0.5, 0.5]])
    hmm_param.obs_prob = np.array([\
        [0.50, 0.20, 0.20, 0.10, 0.00],\
        [0.00, 0.10, 0.40, 0.40, 0.10]\
    ])
    x_lengths = sample_lengths(args.avelen, args.N)
    st, x = sampling_from_hmm(x_lengths, hmm_param)

    with open(args.out_file, 'wb') as f:
        pickle.dump({'model_param': hmm_param,
                    'sample': x,
                    'latent': st,
                    'model_type': 'HMM'}, f)
        print(args.out_file)


def main_mm(args):
    _logger.info(f'{args=}')
    np.random.seed(1)

    M = 2
    D = 5
    init_state = np.array([0.1, 0.9])
    state_tran = np.array([[0.9, 0.1],
                           [0.5, 0.5]])

    print("main_mm")
    X = sample_multiple_markov_process(args.N, init_state, state_tran)
    print(X)

    #with open(args.out_file, 'wb') as f:
    #    pickle.dump({'model_param': kmeans_param,
    #                'sample': X,
    #                'model_type': 'KmeansClustering'}, f)
    #    print(args.out_file)

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='',
                    description='What the program does',
                    epilog='Text at the bottom of help')

    parser.add_argument('N', type=int, help='number of sample')
    parser.add_argument('out_file', type=str, \
        help='output file name(out.pickle)', default='out.pickle')

    subparsers = parser.add_subparsers(title='model',
                                   description='probabilistic models(gmm,mm,hmm)',
                                   help='select one from GMM,HMM,MM',
                                   required=True)
    parser_mm = subparsers.add_parser('MM', help='Markov models')
    #parser_mm.add_argument('bar', type=int, help='bar help')
    parser_mm.set_defaults(func=main_mm)

    parser_gmm = subparsers.add_parser('GMM', help='Gaussian Mixture models')
    parser_gmm.add_argument('--cluster', type=int, help='number of cluster', default=4)
    parser_gmm.add_argument('--dimension', type=int, help='vector dimension', default=2)
    #parser.add_argument('filename')
    #parser.add_argument('-v', '--verbose', action='store_true')
    parser_gmm.set_defaults(func=main_gmm)

    parser_hmm = subparsers.add_parser('HMM', help='Hidden markov models')
    parser_hmm .add_argument('--avelen', type=int, help='average sample lengths', default=10)
    parser_hmm.set_defaults(func=main_hmm)

    args = parser.parse_args()
    args.func(args)

