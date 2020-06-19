import sys
sys.path.insert(0,'../examples/1D_examples')
sys.path.insert(0,'../src/')
from clvs import *
from cusp import *
from numpy import *
from scipy.interpolate import *
from matplotlib import *
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection 
if __name__=="__main__":
#def plot_distribution():
    n_plot = 500 
    x = reshape(linspace(0.,1.,n_plot),\
            [1,n_plot])
    h_arr = [0.8,1.0,1.1]
    gamma_arr = [0.1,0.2,0.3,0.5,0.9]
    h, gamma = meshgrid(h_arr, gamma_arr)
    n_g, n_h = h.shape
    fig, ax = subplots(n_g, n_h)
    n_trj = 10000
    n_spinup = 100
    
    for i in range(n_g):
        for j in range(n_h):
            s = [h[i,j], gamma[i,j]]
            x1 = step(x, s, n_trj)[n_spinup:,0,:]
            x2 = x1.flatten()
            ax[i,j].hist(x2, bins=100, density=True)
            ax[i,j].set_title("h = %2.1f, g = %2.1f " \
                    %(s[0], s[1]), fontsize=30)
            ax[i,j].xaxis.set_tick_params(labelsize=30)
            ax[i,j].yaxis.set_tick_params(labelsize=30)
            ax[i,j].set_xlabel("x", fontsize=30)
            ax[i,j].set_ylabel(r"$\varphi(x)$", \
                    fontsize=30)
            ax[i,j].grid(True)

    fig.tight_layout()
    

#if __name__=="__main__":
def plot_map():
    n_plot = 500 
    x = reshape(linspace(0.,1.,n_plot),\
            [1,n_plot])
    h_arr = [0.1,0.5,1.0]
    gamma_arr = [0.,0.1,0.5,0.9]
    h, gamma = meshgrid(h_arr, gamma_arr)
    n_g, n_h = h.shape
    fig, ax = subplots(n_g, n_h)

    for i in range(n_g):
        for j in range(n_h):
            s = [h[i,j], gamma[i,j]]
            x1 = step(x, s, 1)[-1]
            ax[i,j].plot(x, x1, marker='.', \
                    ms=2.0, color='k')
            ax[i,j].set_title("h = %2.1f, g = %2.1f " \
                    %(s[0], s[1]), fontsize=30)
            ax[i,j].xaxis.set_tick_params(labelsize=30)
            ax[i,j].yaxis.set_tick_params(labelsize=30)
            ax[i,j].set_xlabel("x", fontsize=30)
            ax[i,j].set_ylabel(r"$\varphi(x)$", \
                    fontsize=30)
            ax[i,j].grid(True)

    fig.tight_layout()


