import sys
sys.path.insert(0,'../examples/')
sys.path.insert(0,'../src/')
from clvs import *
from henon import *
from numpy import *
from scipy.interpolate import *
from matplotlib import *
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection 
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
#if __name__=="__main__":
def plot_clvs():
    s = [1.4,0.3]
    u = fixed_point(s)
    n = 40000
    u_trj = step(u,s,n)[0]
    d, n = u_trj.shape
    d_u = 2
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, d_u)
    fig, ax = subplots(1,1)
    n_spinup = 100
    v1 = clv_trj[n_spinup:-n_spinup,:,0].T
    v2 = clv_trj[n_spinup:-n_spinup,:,1].T
    x = u_trj[0,n_spinup:-n_spinup]
    y = u_trj[1,n_spinup:-n_spinup]
    eps = 1.e-2
    ax.plot([x - eps*v1[0], x + eps*v1[0]],\
            [y - eps*v1[1], y + eps*v1[1]],"r")
    #ax.plot([x - eps*v2[0], x + eps*v2[0]],\
     #       [y - eps*v2[1], y + eps*v2[1]],"b")

    ax.xaxis.set_tick_params(labelsize=40) 
    ax.yaxis.set_tick_params(labelsize=40) 
    ax.axis("scaled")
    ax.grid(True)
    ax.set_xlabel("$x_1$",fontsize=40)
    ax.set_ylabel("$x_2$",fontsize=40)

#def plot_W1():
if __name__=="__main__":
    s = [1.4,0.3]
    u = fixed_point(s)
    n = 100000
    u_trj = step(u,s,n)[0]
    d, n = u_trj.shape
    d_u = 1
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, d_u)

    ddu_trj = d2step(u_trj,s)
    W1 = dclv_clv(clv_trj, du_trj, ddu_trj)

    n_spinup = 100
    v1 = clv_trj[n_spinup:-n_spinup,:,0].T
    eps = 1.e-2
    u = u_trj[:,n_spinup:-n_spinup]
    W1 = W1[n_spinup:-n_spinup,:,0].T
    


    eps=array([-1E-2, 1E-2]).reshape([1,2,1])
    segments = u.T.reshape([-1,1,2]) + eps * v1.T.reshape([-1,1,2])
    cross_prod = abs(W1[0]*v1[1] - W1[1]*v1[0])
    segments_1 = segments[(abs(cross_prod) > 0.1)]
    cross_prod_1 = cross_prod[(abs(cross_prod) > 0.1)]
    segments_2 = segments[(abs(cross_prod) > 1.0)]
    cross_prod_2 = cross_prod[(abs(cross_prod) > 1.0)]
    segments_3 = segments[(abs(cross_prod) > 10.0)]
    cross_prod_3 = cross_prod[(abs(cross_prod) > 10.0)]
    segments_4 = segments[(abs(cross_prod) > 100.0)]
    cross_prod_4 = cross_prod[(abs(cross_prod) > 100.0)]

#cross_prod = log(cross_prod) 

    lc = LineCollection(segments_1, cmap=plt.get_cmap('coolwarm'), \
            norm=colors.LogNorm(min(cross_prod_1), max(cross_prod_1)))
    #lc.set_array(ones(u.shape[1]))
    lc.set_array(cross_prod_1)
    #lc.set_array(norm(W1,axis=0))
    lc.set_linewidth(1)

    fig1, ax1 = subplots(1,1)
    ax1.add_collection(lc)
    cbar = fig1.colorbar(cm.ScalarMappable(norm=colors.LogNorm(min(cross_prod_1),max(cross_prod_1)), cmap=plt.get_cmap('coolwarm')), ax=ax1, orientation="horizontal",shrink=0.4,pad=0.1)
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.xaxis.get_offset_text().set_fontsize(30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    ax1.axis('scaled')
    ax1.grid(True)
    ax1.set_xlabel("$x_1$",fontsize=30)
    ax1.set_ylabel("$x_2$",fontsize=30)
    ax1.set_facecolor("black")

    lc = LineCollection(segments_2, cmap=plt.get_cmap('coolwarm'), \
            norm=colors.LogNorm(min(cross_prod_2), max(cross_prod_2)))
    #lc.set_array(ones(u.shape[1]))
    lc.set_array(cross_prod_2)
    #lc.set_array(norm(W1,axis=0))
    lc.set_linewidth(1)

    fig1, ax1 = subplots(1,1)
    ax1.add_collection(lc)
    cbar = fig1.colorbar(cm.ScalarMappable(norm=colors.LogNorm(min(cross_prod_2),max(cross_prod_2)), cmap=plt.get_cmap('coolwarm')), ax=ax1, orientation="horizontal",shrink=0.4,pad=0.1)
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.xaxis.get_offset_text().set_fontsize(30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    ax1.axis('scaled')
    ax1.grid(True)
    ax1.set_xlabel("$x_1$",fontsize=30)
    ax1.set_ylabel("$x_2$",fontsize=30)
    ax1.set_facecolor("black")

    lc = LineCollection(segments_3, cmap=plt.get_cmap('coolwarm'), \
            norm=colors.LogNorm(min(cross_prod_3), max(cross_prod_3)))
    #lc.set_array(ones(u.shape[1]))
    lc.set_array(cross_prod_3)
    #lc.set_array(norm(W1,axis=0))
    lc.set_linewidth(1)

    fig1, ax1 = subplots(1,1)
    ax1.add_collection(lc)
    cbar = fig1.colorbar(cm.ScalarMappable(norm=colors.LogNorm(min(cross_prod_3),max(cross_prod_3)), cmap=plt.get_cmap('coolwarm')), ax=ax1, orientation="horizontal",shrink=0.4,pad=0.1)
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.xaxis.get_offset_text().set_fontsize(30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    ax1.axis('scaled')
    ax1.grid(True)
    ax1.set_xlabel("$x_1$",fontsize=30)
    ax1.set_ylabel("$x_2$",fontsize=30)
    ax1.set_facecolor("black")

    lc = LineCollection(segments_4, cmap=plt.get_cmap('coolwarm'), \
            norm=colors.LogNorm(min(cross_prod_4), max(cross_prod_4)))
    #lc.set_array(ones(u.shape[1]))
    lc.set_array(cross_prod_4)
    #lc.set_array(norm(W1,axis=0))
    lc.set_linewidth(1)

    fig1, ax1 = subplots(1,1)
    ax1.add_collection(lc)
    cbar = fig1.colorbar(cm.ScalarMappable(norm=colors.LogNorm(min(cross_prod_4),max(cross_prod_4)), cmap=plt.get_cmap('coolwarm')), ax=ax1, orientation="horizontal",shrink=0.4,pad=0.1)
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.xaxis.get_offset_text().set_fontsize(30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    ax1.axis('scaled')
    ax1.grid(True)
    ax1.set_xlabel("$x_1$",fontsize=30)
    ax1.set_ylabel("$x_2$",fontsize=30)
    ax1.set_facecolor("black")

