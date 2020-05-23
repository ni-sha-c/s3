import sys
sys.path.insert(0,'../examples/')
sys.path.insert(0,'../src/')
from clvs import *
from henon import *
from numpy import *
from scipy.interpolate import *
from matplotlib import *
from matplotlib.animation import FuncAnimation
#if __name__=="__main__":
def plot_iterations():
    """
    This function plots the evolution of 
    uniformly distributed points on a square, 
    under the Henon map. Compare with the animation 
    on the Wikipedia page of the Henon map, where each 
    iteration has been decomposed into the three substeps.
    """
    fig, ax = subplots(1,1)
    ax.axis("equal")
    x_min, x_max = (-2., 2.)
    y_min, y_max = (-2., 2.)
    ax.set_xlim([x_min,x_max]) 
    ax.set_ylim([y_min,y_max])
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    line, = ax.plot([],[],"ko",ms=3)

    
    s = [1.4,0.3] 
    n = 20
    n_atr = zeros(n,dtype=int)
    
    n_samples = 1000000
    u_last = zeros((2,n_samples))

    def init():
        global u_last 
        u0 = -2.0 + 4.0*rand(2, n_samples)
        u_last = u0
        line.set_label(n_samples)
        n_atr[0] = n_samples
        fig.legend(fontsize=30)
        line.set_data(u0[0], u0[1])
        return line,
    def animate(i):


        u_plot = u_last[:,:n_atr[i]]
        u_plot = step(u_plot, s, 1)[-1]
        
        x = u_plot[0]
        y = u_plot[1]
        atr_pts = (x < x_max) & (x > x_min) & \
                (y > y_min) & (y < y_max)
        u_plot = u_plot[:, atr_pts]
        n_atr[(i+1)%n] = u_plot.shape[1]
        u_last[:] = 0.
        u_last[:,:n_atr[(i+1)%n]] = u_plot
        line.set_data(u_plot[0], u_plot[1])
        line.set_label(n_atr[i])
        fig.legend(fontsize=30)
        return line,
   
    anim = FuncAnimation(fig, animate, \
            init_func=init, frames=n, interval=500)
    anim.save('henon_iterates.mp4')
if __name__=="__main__":
#def plot_clvs():
    s = [1.4,0.3]
    u = fixed_point(s)
    n = 10000
    u_trj = step(u,s,n)[0]
    d, n = u_trj.shape
    d_u = 1
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, d_u)
    


