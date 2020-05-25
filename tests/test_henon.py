import sys
sys.path.insert(0,'../examples/')
sys.path.insert(0,'../src/')
from henon import *
from clvs import *
from numpy import *
from scipy.interpolate import *
from mpl_toolkits.mplot3d import Axes3D
def test_dstep():
    m = 100
    d = 2
    u = rand(d, m)
    eps = 1.e-5
    s = [1.4,0.3]
    du_dx_fd = (step(u + eps*reshape([1.0, 0.],\
            [d,1]), s, 1) - \
               step(u - eps*reshape([1.0, 0.],\
               [d,1]), s, 1))[:,:,-1]/(2*eps)
    du_dy_fd = (step(u + eps*reshape([0., 1.],\
            [d,1]), s, 1) - \
               step(u - eps*reshape([0., 1.],\
               [d,1]), s, 1))[:,:,-1]/(2*eps) 
    du = dstep(u, s).T
    du_dx = vstack([du[0,0], du[0,1]])
    du_dy = vstack([du[1,0], du[1,1]])
    assert(allclose(du_dx, du_dx_fd.T))
    assert(allclose(du_dy, du_dy_fd.T))


def test_d2step():
    m = 100
    d = 2
    u = rand(d, m)
    eps = 1.e-5
    s = [1.4,0.3]
    dx = reshape([1.,0.],[d,1])
    dy = reshape([0.,1.],[d,1])
    ddu_x_fd = (dstep(u + eps*dx, s) - \
            dstep(u - eps*dx, s))/(2*eps)
    ddu_y_fd = (dstep(u + eps*dy, s) - \
            dstep(u - eps*dy, s))/(2*eps)
    ddu = d2step(u,s)
    assert(allclose(ddu[:,:,0,:].T,ddu_x_fd))
    assert(allclose(ddu[:,:,1,:].T,ddu_y_fd))
    
#def test_clv():
if __name__=="__main__":
    s = [1.4,0.3]
    u = fixed_point(s)
    n = 10000
    u_trj = step(u,s,n).T[0]
    d, n = u_trj.shape
    d_u = 1
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, d_u)
    n_spinup = 100
    '''
    v1 = clv_trj[n_spinup:-n_spinup,:,0].T
    v2 = clv_trj[n_spinup:-n_spinup,:,1].T
    v3 = clv_trj[n_spinup:-n_spinup,:,2].T
    u_trj = u_trj[:,n_spinup:-n_spinup]
    n = n - 2*n_spinup
    x, y, z = u_trj[0], u_trj[1], u_trj[2]
    abs_r = sqrt(x*x + y*y)
    theta = reshape([-y/abs_r, x/abs_r],[2, -1]).T
    z_dir = reshape([zeros(n), zeros(n), ones(n)],[3,-1]).T
    r_dir = reshape([x/abs_r, y/abs_r, z],[3,-1]).T
    assert(allclose(diag(dot(theta, v2[0:2])),\
            zeros(n)))
    assert(allclose(diag(dot(theta, v3[0:2])),\
            zeros(n)))
    #assert(allclose(diag(dot(z_dir, v2)),\
     #       zeros(n)))
    #assert(allclose(diag(dot(r_dir, v1)),\
     #       zeros(n)))
   
    #assert(allclose(v1[2],zeros(n)))
    #assert(allclose(v1[2],zeros(n)))
       
    #assert(allclose(v1[2],zeros(n)))
    '''
