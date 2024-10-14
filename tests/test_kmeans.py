import numpy as np
import pickle
from study_gmm.sampler import generate_sample_parameter, generate_samples
from study_gmm.kmeans import kmeans_clustering, KmeansCluster

def test_kmeans():
    # generate samples.
    kmeans_param = generate_sample_parameter(10, 16)
    X = generate_samples(1000, kmeans_param)
    with open("_tmp_kmeans_training.pkl", 'wb') as f:
        pickle.dump({'model_param': kmeans_param,
                    'sample': X,
                    'model_type': 'KmeansClustering'}, f)
 
    # run
    with open("_tmp_kmeans_training.pkl", 'rb') as f:
        data = pickle.load(f)
        #_logger.debug(data.keys())
        X = data['sample']
        param = data['model_param']
        mu_init = param.Mu

        Dim = X.shape[1]
        np.random.seed(0)
        mu_init = np.random.randn(8, Dim)

    #X = np.abs(X + 1.0E-6)
    #mu_init = np.abs(mu_init + 1.0E-6)
    kmeansparam, cost_history = kmeans_clustering(X, mu_init,
                                                  dist_mode="linear")
    assert len(cost_history) > 0

