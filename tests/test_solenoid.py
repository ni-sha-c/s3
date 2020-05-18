import sys
sys.path.insert(0,'../examples/')
from solenoid import *
from numpy import *
from scipy.interpolate import *
from mpl_toolkits.mplot3d import Axes3D
def plot_solenoid():
#if __name__ == "__main__":
    m = 1
    n = 2000
    u0 = rand(3,m)
    u_trj = step(u0, [0.], n)
    fig = figure(figsize=(36,36))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(u_trj[:,0,0], u_trj[:,1,0], u_trj[:,2,0], \
            "o",ms=2.0)
def test_dstep():
    m = 100
    u = rand(3, m)
    eps = 1.e-5
    s = [0.]
    du_dx_fd = (step(u + eps*reshape([1.0, 0.,0.],\
            [3,1]), s, 1) - \
               step(u - eps*reshape([1.0, 0.,0.],\
            [3,1]), s, 1))[-1]/(2*eps)
    du_dy_fd = (step(u + eps*reshape([0., 1.,0.],\
            [3,1]), s, 1) - \
               step(u - eps*reshape([0., 1.,0.],\
            [3,1]), s, 1))[-1]/(2*eps) 
    du_dz_fd = (step(u + eps*reshape([0., 0.,1.],\
            [3,1]), s, 1) - \
               step(u - eps*reshape([0., 0.,1.],\
            [3,1]), s, 1))[-1]/(2*eps)
    du = dstep(u, s).T
    du_dx = vstack([du[0,0], du[0,1], du[0,2]])
    du_dy = vstack([du[1,0], du[1,1], du[1,2]])
    du_dz = vstack([du[2,0], du[2,1], du[2,2]])
    
    assert(allclose(du_dx, du_dx_fd))
    assert(allclose(du_dy, du_dy_fd))
    assert(allclose(du_dz, du_dz_fd))
