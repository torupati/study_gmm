import matplotlib.pyplot as plt

def plot_gamma(ax, _gamma, state_labels: list = []):
    
    ax.imshow(_gamma.transpose(), cmap='Reds', vmin=0, vmax=1)
    ax.set_xlabel('time index')

    T, M = _gamma.shape
    if len(state_labels) == M:
        ax.set_yticks([0, 1, 2], labels=state_labels)
    else:
        ax.set_ylabel('state index')
    ax.invert_yaxis()  # labels read top-to-bottom
    #ax.set_ylabel('state index')
    return ax


def plot_categorical(ax, values: list[int], names: list[str] = None):
    D = len(values)
    t = np.arange(0, D)
    ax.bar(names, values, 
            bins=D, range=(0, D), density=True, align='left', rwidth=0.75, label="Sample", alpha=0.75)
    ax.xticks(t)
    
#plt.savefig('bino.png')
#plt.show()


