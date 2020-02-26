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
    fig, ax = subplots(1,2)
    s = [0.7,0.3]
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

def plot_clv_components():
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


    return fig,ax

def plot_clv_gradients():
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
    n_grid = 500
    x_x = linspace(0.,1.,n_grid)
    x_grid, y_grid = meshgrid(x_x, x_x)
    f = LinearNDInterpolator(u.T[10:],\
            v2[1,10:],)
    v2x = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    dx = x_x[1]
    #calculate x partial derivatives
    dv2x_dx = (hstack([v2x[:,1:],\
            v2x[:,0].reshape(-1,1)]) - \
            hstack([v2x[:,-1].reshape(-1,1),\
            v2x[:,0:-1]]))/(2*dx)
    f = LinearNDInterpolator(u.T[10:],\
            v1[1,10:],)
    v1y = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    dv1y_dx = (hstack([v1y[:,1:],\
            v1y[:,0].reshape(-1,1)]) - \
            hstack([v1y[:,-1].reshape(-1,1),\
            v1y[:,0:-1]]))/(2*dx)
    #calculate y partial derivatives
    dv2x_dy = (vstack([v2x[1:,:],\
            v2x[0,:].reshape(1,-1)]) - \
            vstack([v2x[-1,:].reshape(1,-1),\
            v2x[0:-1,:]]))/(2*dx)
    dv1y_dy = (vstack([v1y[1:,:],\
                    v1y[0,:].reshape(1,-1)]) - \
                    vstack([v1y[-1,:].reshape(1,-1),\
                    v1y[0:-1,:]]))/(2*dx)

    fig, ax = subplots(2,2)
    cntr00 = ax[0,0].contourf(x_grid,\
            y_grid, dv1y_dx, levels=linspace(nanmin(dv1y_dx),\
            nanmax(dv1y_dx)/1.5,30))
    cntr01 = ax[0,1].contourf(x_grid,\
            y_grid, dv2x_dx,levels=linspace(nanmin(dv2x_dx)/5,\
            nanmax(dv2x_dx)/8,30))
    cntr10 = ax[1,0].contourf(x_grid,\
            y_grid, dv1y_dy,levels=linspace(nanmin(dv1y_dy),nanmax(dv1y_dy),30))
    cntr11 = ax[1,1].contourf(x_grid,\
            y_grid, dv2x_dy,levels=linspace(nanmin(dv2x_dy)/5,nanmax(dv2x_dy)/5,30))


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


    ax[0,0].set_title('$\partial_1 V^1_2$',fontsize=24)

    ax[0,1].set_title('$\partial_1 V^2_1$',fontsize=24)

    ax[1,0].set_title('$\partial_2 V^1_2$',fontsize=24)

    ax[1,1].set_title('$\partial_2 V^2_1$',fontsize=24)
    return fig, ax

def plot_clv_directional_derivatives():
    s = [0.7,0.3]
    #s = zeros(2)
    eps = 4.e-2
    d = 2
    u0 = random.rand(d,1)
    n = 30000
    u = step(u0,s,n) #shape: (n+1)xdx1
    u = u[1:].T[0] # shape:dxn
    du = dstep(u,s) #shape:nxdxd
    P = clvs(u,du,2).T #shape:nxdxd
    v1 = P[0]
    v2 = P[1]
    n_grid = 100
    x_x = linspace(0.,1.,n_grid)
    x_grid, y_grid = meshgrid(x_x, x_x)
    f = LinearNDInterpolator(u.T[10:],\
            v1[0,10:],)
    v1x = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v1[1,10:],)
    v1y = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v2[0,10:],)
    v2x = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v2[1,10:],)
    v2y = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)

    dx = x_x[1]
    #calculate x partial derivatives
    dv1y_dx = (hstack([v1x[:,1:],\
            v1x[:,0].reshape(-1,1)]) - \
            hstack([v1x[:,-1].reshape(-1,1),\
            v1x[:,0:-1]]))/(2*dx)
    dv2y_dx = (hstack([v2x[:,1:],\
            v2x[:,0].reshape(-1,1)]) - \
            hstack([v2x[:,-1].reshape(-1,1),\
            v2x[:,0:-1]]))/(2*dx)
    #calculate y partial derivatives
    dv1y_dy = (vstack([v1x[1:,:],\
            v1x[0,:].reshape(1,-1)]) - \
            vstack([v1x[-1,:].reshape(1,-1),\
            v1x[0:-1,:]]))/(2*dx)
    dv2y_dy = (vstack([v2x[1:,:],\
                    v2x[0,:].reshape(1,-1)]) - \
                    vstack([v2x[-1,:].reshape(1,-1),\
                    v2x[0:-1,:]]))/(2*dx)

    # calculate directional derivatives
    dv1y_dv1 = dv1y_dx*v1x + dv1y_dy*v1y
    dv1y_dv2 = dv1y_dx*v2x + dv1y_dy*v2y
    dv2y_dv1 = dv2y_dx*v1x + dv2y_dy*v1y
    dv2y_dv2 = dv2y_dx*v2x + dv2y_dy*v2y


    fig, ax = subplots(2,2)
    cntr00 = ax[0,0].contourf(x_grid,\
            y_grid, dv1y_dv1, levels=linspace(nanmin(dv1y_dv1),\
            nanmax(dv1y_dv1),50))
    cntr01 = ax[0,1].contourf(x_grid,\
            y_grid, dv1y_dv2,levels=linspace(nanmin(dv1y_dv2),\
            nanmax(dv1y_dv2),50))
    cntr10 = ax[1,0].contourf(x_grid,\
            y_grid, dv2y_dv1,levels=linspace(nanmin(dv2y_dv1),nanmax(dv2y_dv1),50))
    cntr11 = ax[1,1].contourf(x_grid,\
            y_grid, dv2y_dv2,levels=linspace(nanmin(dv2y_dv1),nanmax(dv2y_dv2),50))


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


    ax[0,0].set_title('$D V^1_2\cdot V^1$',fontsize=24)

    ax[0,1].set_title('$D V^1_2\cdot V^2$',fontsize=24)

    ax[1,0].set_title('$D V^2_2\cdot V^1$',fontsize=24)

    ax[1,1].set_title('$D V^2_2 \cdot V^2$',fontsize=24)
    return fig, ax

if __name__=="__main__":
#def plot_clv_directional_derivatives():
    s = [0.7,0.3]
    #s = zeros(2)
    eps = 4.e-2
    d = 2
    u0 = random.rand(d,1)
    n = 30000
    u = step(u0,s,n) #shape: (n+1)xdx1
    u = u[1:].T[0] # shape:dxn
    du = dstep(u,s) #shape:nxdxd
    P = clvs(u,du,2).T #shape:nxdxd
    v1 = P[0]
    v2 = P[1]
    n_grid = 100
    x_x = linspace(0.,1.,n_grid)
    x_grid, y_grid = meshgrid(x_x, x_x)
    f = LinearNDInterpolator(u.T[10:],\
            v1[0,10:],)
    v1x = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v1[1,10:],)
    v1y = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v2[0,10:],)
    v2x = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v2[1,10:],)
    v2y = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)

    dx = x_x[1]
    #calculate x partial derivatives
    dv1x_dx = (hstack([v1x[:,1:],\
            v1x[:,0].reshape(-1,1)]) - \
            hstack([v1x[:,-1].reshape(-1,1),\
            v1x[:,0:-1]]))/(2*dx)
    dv1y_dx = (hstack([v1y[:,1:],\
            v1y[:,0].reshape(-1,1)]) - \
            hstack([v1y[:,-1].reshape(-1,1),\
            v1y[:,0:-1]]))/(2*dx)
    dv2x_dx = (hstack([v2x[:,1:],\
            v2x[:,0].reshape(-1,1)]) - \
            hstack([v2x[:,-1].reshape(-1,1),\
            v2x[:,0:-1]]))/(2*dx)
    dv2y_dx = (hstack([v2y[:,1:],\
            v2y[:,0].reshape(-1,1)]) - \
            hstack([v2y[:,-1].reshape(-1,1),\
            v2y[:,0:-1]]))/(2*dx)

    #calculate y partial derivatives
    dv1x_dy = (vstack([v1x[1:,:],\
            v1x[0,:].reshape(1,-1)]) - \
            vstack([v1x[-1,:].reshape(1,-1),\
            v1x[0:-1,:]]))/(2*dx)
    dv2x_dy = (vstack([v2x[1:,:],\
            v2x[0,:].reshape(1,-1)]) - \
            vstack([v2x[-1,:].reshape(1,-1),\
            v2x[0:-1,:]]))/(2*dx)
    dv1y_dy = (vstack([v1y[1:,:],\
            v1y[0,:].reshape(1,-1)]) - \
            vstack([v1y[-1,:].reshape(1,-1),\
            v1y[0:-1,:]]))/(2*dx)
    dv2y_dy = (vstack([v2y[1:,:],\
                    v2y[0,:].reshape(1,-1)]) - \
                    vstack([v2y[-1,:].reshape(1,-1),\
                    v2y[0:-1,:]]))/(2*dx)

    # calculate directional derivatives
    dv1x_dv1 = dv1x_dx*v1x + dv1x_dy*v1y 
    dv1y_dv1 = dv1y_dx*v1x + dv1y_dy*v1y
    dv2x_dv2 = dv2x_dx*v2x + dv2x_dy*v2y 
    dv2y_dv2 = dv2y_dx*v2x + dv2y_dy*v2y

    #normalize directional derivatives (only getting 
    # direction of the directional derivatives)
    norm_dv1_dv1 = sqrt(dv1x_dv1*dv1x_dv1 + dv1y_dv1*dv1y_dv1)
    norm_dv2_dv2 = sqrt(dv2x_dv2*dv2x_dv2 + dv2y_dv2*dv2y_dv2)
    dv1x_dv1 /= norm_dv1_dv1
    dv1y_dv1 /= norm_dv1_dv1
    dv2x_dv2 /= norm_dv2_dv2
    dv2y_dv2 /= norm_dv2_dv2

    #flatten the grid
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    dv1x_dv1 = dv1x_dv1.flatten()
    dv1y_dv1 = dv1y_dv1.flatten()
    dv2x_dv2 = dv2x_dv2.flatten()
    dv2y_dv2 = dv2y_dv2.flatten()

    # plot
    eps = 2.e-2
    fig, ax = subplots(1,2)
    ax[0].plot([x_grid - eps*dv1x_dv1, x_grid + eps*dv1x_dv1],\
            [y_grid - eps*dv1y_dv1, y_grid + eps*dv1y_dv1],\
            lw=2.0, color='red')
    ax[1].plot([x_grid - eps*dv2x_dv2, x_grid + eps*dv2x_dv2],\
            [y_grid - eps*dv2y_dv2, y_grid + eps*dv2y_dv2],\
            lw=2.0, color='black')
    
    
    ax[0].set_title('$D V^1\cdot V^1$',fontsize=24)

    ax[1].set_title('$D V^1\cdot V^2$',fontsize=24)

    #return fig, ax

