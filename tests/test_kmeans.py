import numpy as np
import pickle
from study_gmm.sampler import generate_sample_parameter, generate_samples
from study_gmm.kmeans import kmeans_clustering

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


def kmeans_clustering_orig(X:np.ndarray, mu_init:np.ndarray, max_it:int = 20, save_ckpt:bool = False):
    """Run k-means clustering.

    Args:
        X (np.ndarray): vector samples (N, D)
        mu_init (np.ndarray): initial mean vectors(K, D)
        max_it (int, optional): Iteration steps. Defaults to 20.

    Returns:
        _type_: _description_
    """
    K = mu_init.shape[0]
    Dim = X.shape[1]
    kmeansparam = kmeans.KmeansCluster(K, Dim)
    kmeansparam.Mu = mu_init
    R = None # alignmnet
    cost_history = []
    fig0, axes0 = plt.subplots(1, 4, figsize=(16, 4))
    for it in range(0, max_it):
        R_new = kmeansparam.get_alignment(X) # alignment to all sample to all clusters
        #R_new = kmeansparam.get_soft_alignment(X) # not implemented yet
        #if it > 2:
        #    num_updated = (R_new - R) == np.zeros([N,K])
        #    print('num_updated = ', num_updated)
            #if num_updated == 0:
            #    break
        #print(R_new)
        R = R_new
        loss = kmeansparam.distortion_measure(X, R)
        cost_history.append(loss)
#        if it < (2*2):
#            print(int(floor(it/2)), it%2)
#            ax =axes0[it]
            #ax =axes0[floor(it/2), it%2]
            #show_prm(ax, X, R, mu, X_col, KmeansParam)
#            show_prm(ax, X, R, kmeansparam.Mu)
#            ax.set_title("iteration {0:d}".format(it + 1))
        print(f'iteration={it} distortion={loss}')
        kmeansparam.Mu = kmeansparam.get_centroid(X, R)
        loss = kmeansparam.distortion_measure(X, R)
        cost_history.append(loss)
        if save_ckpt:
            ckpt_file = f"kmeans_itr{it}.ckpt"
            with open(ckpt_file, 'wb') as f:
                pickle.dump({'model': kmeansparam,
                    'loss': loss,
                    'iteration': it,
                    'alignment': R}, f)
            print(ckpt_file)
        # convergence check must be done
    fig0.savefig("iteration.png")

    print("centroid:", np.round(kmeansparam.Mu, 5))
    print(kmeansparam.distortion_measure(X, kmeansparam.get_alignment(X)))
    return kmeansparam, cost_history

