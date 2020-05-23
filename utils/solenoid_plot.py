import sys
sys.path.insert(0,'../examples/')
sys.path.insert(0,'../src/')
from solenoid import *
from clvs import *
from numpy import *
from scipy.interpolate import *
from matplotlib import *
from matplotlib.collections import LineCollection 
if __name__ == "__main__":
#def plot_dDV1cdotV1():
    u = rand(3).reshape(3,1)
    n = 10000
    s = [1.,10.]
    u_trj = step(u, s, n).T[0]
    d, n = shape(u_trj)
    d_u = 1
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, d_u)
    ddu_trj = d2step(u_trj,s)
    W1 = dclv_clv(clv_trj, du_trj, ddu_trj)
    
    
    n_spinup = 100
    v1 = clv_trj[n_spinup:,0:2,0].T
    eps = 1.e-2
    u = u_trj[:,n_spinup:]
    W1 = W1[n_spinup:,:,0].T
    eps=array([-1E-2, 1E-2]).reshape([1,2,1])
    segments = u[0:2].T.reshape([-1,1,2]) + eps * v1.T.reshape([-1,1,2])
    cross_prod = W1[0]*v1[1] - W1[1]*v1[0]
    lc = LineCollection(segments, cmap=plt.get_cmap('RdBu'), \
            norm=plt.Normalize(min(cross_prod), max(cross_prod)))
    #lc.set_array(ones(u.shape[1]))
    lc.set_array(cross_prod)
    #lc.set_array(norm(W1,axis=0))
    lc.set_linewidth(1)

    fig2, ax2 = subplots(1,1) 
    ax2.add_collection(lc)
    fig2.colorbar(cm.ScalarMappable(norm=plt.Normalize(min(cross_prod),max(cross_prod)), cmap=plt.get_cmap('RdBu')), ax=ax2)

    ax2.xaxis.set_tick_params(labelsize=30)
    ax2.yaxis.set_tick_params(labelsize=30)
    ax2.axis('scaled')
    ax2.grid(True)
#if __name__=="__main__":
def plot_clvs():
    u = rand(3,1)
    n = 1000
    s = [1.,1.e5]
    u_trj = step(u, s, n).T[0]
    d, n = shape(u_trj)
    d_u = 1
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, d_u)

    fig, ax = subplots(1,1)
    eps = 1.e-1
    
    n_spinup = 100
    v1 = clv_trj[n_spinup:,:,0].T 
    x,y,z=u_trj[:,n_spinup:]
    r = sqrt(x*x + y*y)
    t = reshape([-y/r, x/r],[2,-1])
    ax.plot([x - eps*v1[0], \
             x + eps*v1[0]], \
            [y - eps*v1[1], \
             y + eps*v1[1]], \
            "k")
    ax.plot([x - eps*t[0], \
             x + eps*t[0]], \
            [y - eps*t[1], \
             y + eps*t[1]], \
            "b")
    ax.grid(True)
    ax.set_xlabel('x',fontsize=30)
    ax.set_ylabel('y', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.axis("scaled")

def plot_attractor():
#if __name__=="__main__":
    u = rand(3,1)
    s = [1.0]
    n = 20000
    u_trj = step(u, s, n).T[0]
    fig, ax = subplots(1,2)
    ax[0].plot(u_trj[1],u_trj[0],"ko",ms=1.)
    ax[0].set_xlabel('y',fontsize=30)
    ax[0].set_ylabel('x', fontsize=30)
    ax[0].xaxis.set_tick_params(labelsize=30)
    ax[0].yaxis.set_tick_params(labelsize=30)
    ax[0].axis("scaled")
    ax[1].plot(u_trj[1],u_trj[2],"ko",ms=1.)
    ax[1].set_xlabel('y',fontsize=30)
    ax[1].set_ylabel('z', fontsize=30)
    ax[1].xaxis.set_tick_params(labelsize=30)
    ax[1].yaxis.set_tick_params(labelsize=30)
    ax[1].axis("scaled")
