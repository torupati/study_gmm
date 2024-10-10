import pickle
import argparse

import numpy as np

import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

from study_gmm.kmeans import kmeans_clustering
from study_gmm.kmeans_plot import plot_distortion_history


def train_kmeans(args):
    _logger.info("input: %s", args.input_file)
    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)
        #_logger.debug(data.keys())
        X = data['sample']
        _logger.info('model type: %s', data.get('model_type'))
        param = data['model_param']
        mu_init = param.Mu

    if args.num_cluster > 0:
        Dim = X.shape[1]
        np.random.seed(args.random_seed)
        mu_init = np.random.randn(args.num_cluster, Dim)
        _logger.info("initial points generated from randn (%d %d)", args.num_cluster, Dim)

    #X = np.abs(X + 1.0E-6)
    #mu_init = np.abs(mu_init + 1.0E-6)
    kmeansparam, cost_history = kmeans_clustering(X, mu_init,
                                                  dist_mode=args.dist_mode,
                                                  max_it = 100)
    #print('Mu:', kmeansparam.Mu)
    #print('Sigma:', kmeansparam.Sigma)

    out_pngfile = "distortion.png"
    fig = plot_distortion_history(cost_history)
    fig.savefig(out_pngfile)
    _logger.info('out: %s', out_pngfile)

    R = kmeansparam.get_alignment(X)
    out_file = 'out_kmeans.pickle'
    with open(out_file, 'wb') as f:
        pickle.dump({'model': kmeansparam,
                     'history': cost_history,
                     'iteration': len(cost_history),
                     'alignment': R},
                    f)
    #print(sum(R == 1))
    _logger.info('out: %s', out_file)


def set_parser():
    parser = argparse.ArgumentParser(
                    prog='K-means clustering',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('input_file', type=str, help='sample data in pickle')
    parser.add_argument('--random_seed', type=int,
                        help='random seed', default=0)
    parser.add_argument('--dist_mode', type=str,
                        help='distance mode. linear, log, kldiv',
                        default="linear")
    parser.add_argument('-k', '--num_cluster', type=int,
                        help='number of cluster.',
                        default=0)
    return parser

def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S',
                    handlers=[
                        logging.FileHandler("kmeans.log"),
                        logging.StreamHandler()
                    ],
                    level=logging.INFO)
    parser = set_parser()
    args = parser.parse_args()
    train_kmeans(args)
    _logger.info("log: kmeans.log")

if __name__ == '__main__':
    main()
