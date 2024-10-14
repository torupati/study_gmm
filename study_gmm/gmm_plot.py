import matplotlib.pyplot as plt


def plot_loglikelihood_history(loglikelihood_history):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(range(0, len(loglikelihood_history)), loglikelihood_history, color='k', linestyle='-', marker='o')
    ax.set_xlim([0,len(loglikelihood_history)])
    ax.set_ylabel("log liklihood")
    ax.set_xlabel("iteration step")
    #plt.ylim([40, 80])
    ax.grid(True)
    return fig