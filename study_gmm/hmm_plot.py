import matplotlib.pyplot as plt

def plot_gamma(ax, _gamma):
    
    ax.imshow(_gamma.transpose(), cmap='Reds', vmin=0, vmax=1)
    ax.set_xlabel('time index')
    ax.set_ylabel('state index')
    return ax

    
    