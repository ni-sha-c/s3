import sys
sys.path.insert(0,'../examples/')
from lorenz63 import *
from numpy import *
from scipy.interpolate import *
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

def test_dstep():
    n = 100
    u = rand(n,3)
    s = rand(3)
    du_ana = dstep(u, s).T
    eps = 1.e-7
    du_x = (step(u + eps*reshape([1.,0.,0.],[1,3]),s,1) - \
            step(u - eps*reshape([1.,0.,0.],[1,3]),s,1))/\
            (2*eps)
    du_y = (step(u + eps*reshape([0.,1.,0.],[1,3]),s,1) - \
            step(u - eps*reshape([0.,1.,0.],[1,3]),s,1))/\
                    (2*eps)

    du_z = (step(u + eps*reshape([0.,0.,1.],[1,3]),s,1) - \
            step(u - eps*reshape([0.,0.,1.],[1,3]),s,1))/\
            (2*eps)
    du_fd_x = du_x[-1]
    du_fd_y = du_y[-1]
    du_fd_z = du_z[-1]

    assert(allclose(du_fd_x, du_ana[0]))
    assert(allclose(du_fd_y, du_ana[1]))
    assert(allclose(du_fd_z, du_ana[2]))


def test_d2step():
    n = 100
    u = rand(n,3)
    
    s = rand(3)
    d2_ana = d2step(u, s)

    eps = 1.e-10
    d2_x = (dstep(u + eps*reshape([1.,0.,0.],[1,3]), s) -\
           dstep(u - eps*reshape([1.,0.,0.],[1,3]), s))/\
           (2*eps)
    d2_y = (dstep(u + eps*reshape([0.,1.,0.],[1,3]), s) -\
           dstep(u - eps*reshape([0.,1.,0.],[1,3]), s))/\
           (2*eps)
    d2_z = (dstep(u + eps*reshape([0.,0.,1.],[1,3]), s) -\
           dstep(u - eps*reshape([0.,0.,1.],[1,3]), s))/\
           (2*eps)
  
    assert(allclose(d2_x, d2_ana[:,0])) 
    assert(allclose(d2_y, d2_ana[:,1])) 
    assert(allclose(d2_z, d2_ana[:,2])) 
