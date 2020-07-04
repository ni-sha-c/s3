import sys
sys.path.insert(0,'../examples/')
sys.path.insert(0,'../src/')
from lorenz63 import *
from clvs import *
from numpy import *
from scipy.interpolate import *
from matplotlib import *
rcParams['font.size'] = 30
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection 
#if __name__ == "__main__":
def plot_W1_2D():
    u = rand(3).reshape(1,3)
    
    s = [10.,28.,8./3.] 
    u = step(u, s, 5000).T[0][:,-1]

    d_u = 1
    n_spinup = 100
    n = 100000

    u = reshape(u, [1, 3])
    u_trj = step(u, s, n).T[0]
    du_trj = dstep(u_trj.T, s)
    clv_trj = clvs(u_trj, du_trj, d_u)
    
    ddu_trj = d2step(u_trj.T,s)
    W = dclv_clv(clv_trj, du_trj, ddu_trj)
    W1 = W[n_spinup:-n_spinup,:,0].T
    V1 = clv_trj[n_spinup:-n_spinup,:,0].T
    U = u_trj[:,n_spinup:-n_spinup]

    wyz = W1[1:]
    vyz = V1[1:]
    uyz = U[1:]
    
    wxz = W1[:3:2]
    vxz = V1[:3:2]
    uxz = U[:3:2]

    eps=array([-5E-2, 5E-2]).reshape([1,2,1])

    fig, ax = subplots(1,2) 
    segments_xz = uxz.T.reshape([-1,1,2]) + \
                    eps * vxz.T.reshape([-1,1,2])
    cross_prod = norm(W1, axis=0) 
    lc_xz = LineCollection(segments_xz, cmap=\
                    plt.get_cmap('RdBu'), \
                    norm=colors.LogNorm(min(cross_prod),\
                    max(cross_prod)))
    lc_xz.set_array(cross_prod)
    lc_xz.set_linewidth(2)
    ax[0].add_collection(lc_xz)
    cbar = fig.colorbar(cm.ScalarMappable(norm=colors.LogNorm(min(cross_prod),max(cross_prod)), cmap=plt.get_cmap('RdBu')), ax=ax[0])
    cbar.ax.tick_params(labelsize=30)
    ax[0].xaxis.set_tick_params(labelsize=30)
    ax[0].yaxis.set_tick_params(labelsize=30)
    ax[0].axis('scaled')
    ax[0].set_xlabel("x",fontsize=30)
    ax[0].set_ylabel("z",fontsize=30)
    
    segments_yz = uyz.T.reshape([-1,1,2]) + \
                    eps * vyz.T.reshape([-1,1,2])
    lc_yz = LineCollection(segments_yz, cmap=\
                    plt.get_cmap('RdBu'), \
                    norm=colors.LogNorm(min(cross_prod),\
                    max(cross_prod)))
    lc_yz.set_array(cross_prod)
    lc_yz.set_linewidth(2)
    ax[1].add_collection(lc_yz)
    ax[1].xaxis.set_tick_params(labelsize=30)
    ax[1].yaxis.set_tick_params(labelsize=30)
    ax[1].axis('scaled')
    ax[1].set_xlabel("y", fontsize=30)
    ax[1].set_ylabel("z", fontsize=30)

#if __name__=="__main__":
def plot_W1_3D():
    u = rand(3).reshape(1,3)
    
    s = [10.,28.,8./3.] 
    u = step(u, s, 5000).T[0][:,-1]

    d_u = 1
    n_spinup = 100
    n = 100000

    u = reshape(u, [1, 3])
    u_trj = step(u, s, n).T[0]
    du_trj = dstep(u_trj.T, s)
    clv_trj = clvs(u_trj, du_trj, d_u)
    
    ddu_trj = d2step(u_trj.T,s)
    W = dclv_clv(clv_trj, du_trj, ddu_trj)
    W1 = W[n_spinup:-n_spinup,:,0].T
    V1 = clv_trj[n_spinup:-n_spinup,:,0].T
    U = u_trj[:,n_spinup:-n_spinup]
    d, n = U.shape
    eps=array([-5E-2, 5E-2]).reshape([1,2,1])

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')


    segments = U.T.reshape([-1,1,3]) + \
                    eps * V1.T.reshape([-1,1,3])
    cross_prod = norm(W1, axis=0) 
    '''
    lc = Line3DCollection(segments, cmap=\
                    plt.get_cmap('RdBu'), \
                    norm=colors.LogNorm(min(cross_prod),\
                    max(cross_prod)))
    lc.set_array(cross_prod)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    '''
    rdbu_cmap = plt.get_cmap('RdBu')
    colrs = colors.LogNorm(min(cross_prod), \
            max(cross_prod))
    for i in range(n):
        ax.plot(segments[i,:,0], segments[i,:,1], \
                segments[i,:,2], color=rdbu_cmap(colrs(\
                cross_prod[i])))

    cbar = fig.colorbar(cm.ScalarMappable(norm=colors.LogNorm(min(cross_prod),max(cross_prod)), cmap=plt.get_cmap('RdBu')), ax=ax)
    cbar.ax.tick_params(labelsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    #ax.axis('scaled')
    ax.set_xlabel("x",fontsize=30)
    ax.set_ylabel("z",fontsize=30)
    ax.set_zlabel("z",fontsize=30)

#def plot_unstable_manifold():
if __name__=="__main__":
    n_t = 100000
    t = linspace(-0.1,0.1,n_t)
    y0 = 0.
    z0 = 20   
    n_times = 4   
    n_interval = 20
    d = 3
    s = [10.,28.,8./3.] 

    rdbu_cmap = plt.get_cmap('RdBu')
    colrs = colors.Normalize(min(t), max(t))

    n_cols = 2
    n_rows = n_times//n_cols
    time_per_col = n_interval*(n_rows-1)
    
    fig = plt.figure()
    for i in range(n_rows):
        for j in range(n_cols):
            index = n_rows*j + i + 1
            ax = fig.add_subplot(n_rows,n_cols,index,\
                    projection='3d')
            locator_params(axis='x',nbins=3)
            locator_params(axis='y',nbins=3)
            locator_params(axis='z',nbins=3)
            '''
            if i==0 and j==0:
                cbar = fig.colorbar(cm.ScalarMappable(\
                        norm=colors.Normalize(min(t),\
                        max(t)), cmap=plt.get_cmap('RdBu'))\
                        , ax=ax)
                cbar.ax.tick_params(labelsize=30)
            '''
            timeij = (i + j*n_rows)*n_interval + 2000 
            print(index,timeij)
            u0 = reshape([t, y0*ones(n_t), z0*ones(n_t)],\
                [d, n_t]).T
            ut = step(u0, s, timeij)[-1]
            xt, yt, zt = ut
            ax.plot(xt, yt, zt, '.', ms=2)
            #for k,tk in enumerate(t):
             #   ax.plot([xt[k]], [yt[k]], [zt[k]], '.',ms=2.0,\
              #  mfc=rdbu_cmap(colrs(tk)))

            #ax.xaxis.set_tick_params(labelsize=30)
            #ax.yaxis.set_tick_params(labelsize=30)
            #ax.zaxis.set_tick_params(labelsize=30)
            real_time_ij = dt*timeij
            ax.set_title("t = %1.1f" % real_time_ij,\
                    fontsize=30)
            ax.set_xlabel("x", fontsize=30, labelpad=30)
            ax.set_ylabel("y", fontsize=30, labelpad=30)
            ax.set_zlabel("z", fontsize=30, labelpad=10)
            ax.ticklabel_format(useOffset=False)


    #for ti in times:
     #          



def plot_clvs():
    fig, ax = subplots(1,2)
    s = [0.9,0.4]
    eps = 1.e-2
    d = 2
    u0 = random.rand(d,1)
    n = 10000
    u = step(u0,s,n) #shape: (n+1)xdx1
    u = u[1:].T[0] # shape:dxn
    du = dstep(u,s) #shape:nxdxd
    P = clvs(u,du,d).T #shape:nxdxd
    v1 = P[0]
    v2 = P[1]
    ax[0].plot([u[0] - eps*v1[0], u[0] + eps*v1[0]],\
            [u[1] - eps*v1[1], u[1] + eps*v1[1]],\
            lw=2.0, color='red')
    ax[1].plot([u[0] - eps*v2[0], u[0] + eps*v2[0]],\
            [u[1] - eps*v2[1], u[1] + eps*v2[1]],\
            lw=2.0, color='black')

    ax[0].set_title('$V^1$',fontsize=24)
    
    ax[1].set_title('$V^2$',fontsize=24)
    for j in range(2):
            ax[j].xaxis.set_tick_params(labelsize=24)
            ax[j].yaxis.set_tick_params(labelsize=24)


    return fig,ax


