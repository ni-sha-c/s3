import sys
sys.path.insert(0,'../examples/')
from PerturbedCatMap import *
from numpy import *
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

def compare_clvs():
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

