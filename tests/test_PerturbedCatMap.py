import sys
sys.path.insert(0,'../examples/')
from PerturbedCatMap import *
from numpy import *
from scipy.interpolate import *
def test_daugmented_step():
    d = 4
    m = 10
    u = rand(d,m)
    s = [0.7, 0.3]
    du = daugmented_step(u, s)
    n_eps = 6
    eps = logspace(-10,-2,n_eps)
    err = empty(n_eps)
    u_plus = copy(u)
    u_minus = copy(u)
    du_fd = empty((m,d,d))
    for n,ep in enumerate(eps):
        for i in range(d):
            u_plus[i] += ep
            u_minus[i] -= ep

            Tu_plus = augmented_step(u_plus,\
                s)[-1]
            Tu_minus = augmented_step(u_minus,\
                s)[-1]

            u_plus = copy(u)
            u_minus = copy(u)
            
            du_fd[:,:,i] = (Tu_plus - Tu_minus).T/(2*ep)
        err[n] = linalg.norm(abs(du_fd - du))
    fig, ax = subplots(1,1)
    ax.loglog(eps,err,'o-')
    return err


def plot_clvs():
    fig, ax = subplots(1,1)
    s = [0.7,0.3]
    eps = 5.e-2
    d = 2
    u0 = random.rand(d,1)
    n = 1000
    u = step(u0,s,n) #shape: (n+1)xdx1
    u = u[1:].T[0] # shape:dxn
    du = dstep(u,s) #shape:nxdxd
    P = clvs(u,du,d).T #shape:nxdxd
    v1 = P[0]
    v2 = P[1]
    ax.plot(u[0], u[1], 'k.', ms=1)
    ax.plot([u[0] - eps*v1[0], u[0] + eps*v1[0]],\
            [u[1] - eps*v1[1], u[1] + eps*v1[1]],\
            lw=2.0, color='red')
    ax.plot([u[0] - eps*v2[0], u[0] + eps*v2[0]],\
            [u[1] - eps*v2[1], u[1] + eps*v2[1]],\
            lw=2.0, color='black')
    return fig,ax

#def compare_clvs():
if __name__=="__main__":
    fig, ax = subplots(2,2)
    s = [0.7,0.3]
    #s = zeros(2)
    eps = 4.e-2
    d = 2
    u0 = random.rand(d,1)
    n = 20000
    u = step(u0,s,n) #shape: (n+1)xdx1
    u = u[1:].T[0] # shape:dxn
    du = dstep(u,s) #shape:nxdxd
    P = clvs(u,du,2).T #shape:nxdxd
    v1 = P[0]
    v2 = P[1]
    '''
    ax.plot(u[0], u[1], 'k.', ms=1)
    ax.plot([u[0] - eps*v1[0], u[0] + eps*v1[0]],\
            [u[1] - eps*v1[1], u[1] + eps*v1[1]],\
            lw=2.0, color='red')
    ax.plot([u[0] - eps*v2[0], u[0] + eps*v2[0]],\
            [u[1] - eps*v2[1], u[1] + eps*v2[1]],\
            lw=2.0, color='black')
    
    n_grid = 50
    x_x = linspace(0.,1.,n_grid)
    x_grid, y_grid = meshgrid(x_x, x_x)
    f = interp2d(u[0,10:], u[1,10:], v2[0,10:])
    vx = f(x_x, x_x).reshape(\
            n_grid,n_grid)
    c = ax.contourf(x_grid, x_grid, vx,20,vmin=-1.0,vmax=1.0)
    '''
    ax[0,0].tricontour(u[0,10:], u[1,10:], \
            v1[0,10:], \
            linewidths=0.5,\
            colors='k')
    ax[0,0].set_title('$V^1_1$',fontsize=24)
    cntr00 = ax[0,0].tricontourf(u[0,10:],\
            u[1,10:], v1[0,10:],\
            levels=linspace(min(v1[0,10:]),\
            max(v1[0,10:]), 50),\
            cmap="RdBu_r")
    
    ax[0,1].tricontour(u[0,10:], u[1,10:], \
            v1[1,10:], \
            linewidths=0.5,\
            colors='k')
    cntr01 = ax[0,1].tricontourf(u[0,10:],\
            u[1,10:], v1[1,10:],\
            levels=linspace(min(v1[1,10:]),\
            max(v1[1,10:]), 30),\
            cmap="RdBu_r")
    ax[0,1].set_title('$V^1_2$',fontsize=24)
    
    ax[1,0].tricontour(u[0,10:], u[1,10:], \
            v2[0,10:], \
            linewidths=0.5,\
            colors='k')
    cntr10 = ax[1,0].tricontourf(u[0,10:],\
            u[1,10:], v2[0,10:],\
            levels=linspace(min(v2[0,10:]),\
            max(v2[0,10:]), 30),\
            cmap="RdBu_r")
    ax[1,0].set_title('$V^2_1$',fontsize=24)

    ax[1,1].tricontour(u[0,10:], u[1,10:], \
            v2[1,10:], \
            linewidths=0.5,\
            colors='k')
    cntr11 = ax[1,1].tricontourf(u[0,10:],\
            u[1,10:], v2[1,10:],\
            levels=linspace(min(v2[1,10:]),\
            max(v2[1,10:]), 30),\
            cmap="RdBu_r")
    ax[1,1].set_title('$V^2_2$',fontsize=24)

    fig.colorbar(cntr00, ax=ax[0,0])
    ax[0,0].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr01, ax=ax[0,1])
    ax[0,1].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr10, ax=ax[1,0])
    ax[1,0].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr11, ax=ax[1,1])
    ax[1,1].set(xlim=(0, 1), ylim=(0, 1))

    for i in range(2):
        for j in range(2):
            ax[i,j].xaxis.set_tick_params(labelsize=24)
            ax[i,j].yaxis.set_tick_params(labelsize=24)


    #return fig,ax
