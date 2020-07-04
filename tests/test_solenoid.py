import sys
sys.path.insert(0,'../examples/')
sys.path.insert(0,'../src/')
from solenoid import *
from clvs import *
from numpy import *
from scipy.interpolate import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
def plot_solenoid():
#if __name__ == "__main__":
    m = 1
    n = 2000
    u0 = rand(3,m)
    s = [1.,4.]
    u_trj = step(u0, s, n)[0]
    fig = figure(figsize=(36,36))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(u_trj[0], u_trj[1], u_trj[2], \
            "o",ms=2.0)
def test_dstep():
    m = 100
    u = rand(3, m)
    eps = 1.e-5
    s = [1.,4.]
    du_dx_fd = (step(u + eps*reshape([1.0, 0.,0.],\
            [3,1]), s, 1) - \
               step(u - eps*reshape([1.0, 0.,0.],\
               [3,1]), s, 1))[:,:,-1]/(2*eps)
    du_dy_fd = (step(u + eps*reshape([0., 1.,0.],\
            [3,1]), s, 1) - \
               step(u - eps*reshape([0., 1.,0.],\
               [3,1]), s, 1))[:,:,-1]/(2*eps) 
    du_dz_fd = (step(u + eps*reshape([0., 0.,1.],\
            [3,1]), s, 1) - \
               step(u - eps*reshape([0., 0.,1.],\
               [3,1]), s, 1))[:,:,-1]/(2*eps)
    du = dstep(u, s).T
    du_dx = vstack([du[0,0], du[0,1], du[0,2]])
    du_dy = vstack([du[1,0], du[1,1], du[1,2]])
    du_dz = vstack([du[2,0], du[2,1], du[2,2]])
    
    assert(allclose(du_dx, du_dx_fd.T))
    assert(allclose(du_dy, du_dy_fd.T))
    assert(allclose(du_dz, du_dz_fd.T))
def test_clv():
    s = [1.,1.e5]
    u = rand(3,1)
    n = 1000
    u_trj = step(u,s,n)[0]
    d, n = u_trj.shape
    d_u = 3
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, d_u)
    n_spinup = 100
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
def analytical_V1(u):
    x,y,z = u
    d, n = u.shape
    r,t = sqrt(x*x + y*y), arctan2(y,x)
    du = zeros((d,d,n))
    r_1 = 1 + cos(t)/2
    s2t = sin(2*t)
    c2t = cos(2*t)
    st = sin(t)
    ct = cos(t)
    dgamma1_dt = -2.0*r_1*s2t - 0.5*st*c2t
    dgamma2_dt = 2.0*r_1*c2t - 0.5*st*s2t
    dgamma3_dt = 0.5*ct
    tangent_vector_field = reshape([dgamma1_dt,\
            dgamma2_dt,\
            dgamma3_dt], [d,n])
    return tangent_vector_field/norm(tangent_vector_field,\
            axis=0)

def analytical_jacobian(u):
    x,y,z = u
    d, n = u.shape
    r,t = sqrt(x*x + y*y), arctan2(y,x)
    du = zeros((d,d,n))
    r_1 = 1 + cos(t)/2
    s2t = sin(2*t)
    c2t = cos(2*t)
    st = sin(t)
    ct = cos(t)
    dgamma1_dt = -2.0*r_1*s2t - 0.5*st*c2t
    dgamma2_dt = 2.0*r_1*c2t - 0.5*st*s2t
    dgamma3_dt = 0.5*ct

    dt_dx = -st/r
    dt_dy = ct/r
    dt_dz = zeros(n)

    dgamma1_dx = dgamma1_dt*dt_dx 
    dgamma2_dx = dgamma2_dt*dt_dx 
    dgamma3_dx = dgamma3_dt*dt_dx 

    dgamma1_dy = dgamma1_dt*dt_dy 
    dgamma2_dy = dgamma2_dt*dt_dy 
    dgamma3_dy = dgamma3_dt*dt_dy

    dgamma1_dz = dgamma1_dt*dt_dz
    dgamma2_dz = dgamma2_dt*dt_dz 
    dgamma3_dz = dgamma3_dt*dt_dz

    du[0,0] = dgamma1_dx 
    du[1,0] = dgamma1_dy
    du[2,0] = dgamma1_dz

    
    du[0,1] = dgamma2_dx 
    du[1,1] = dgamma2_dy
    du[2,1] = dgamma2_dz


    du[0,2] = dgamma3_dx 
    du[1,2] = dgamma3_dy
    du[2,2] = dgamma3_dz

    return du.T

def analytical_W1(u):
    x,y,z = u
    d, n = u.shape
    r,t = sqrt(x*x + y*y), arctan2(y,x)
    du = zeros((d,d,n))
    r_1 = 1 + cos(t)/2
    
    ct = cos(t)
    c2t = cos(2*t)
    c3t = cos(3*t)
    c4t = cos(4*t)
    c5t = cos(5*t)
    st = sin(t)
    s2t = sin(2*t)
    s3t = sin(3*t)
    s4t = sin(4*t)
    s5t = sin(5*t)

    den = (2*(16*ct + 2*c2t + 19)**1.5)

    dV1_dt = -(193*ct + 392*c2t + 267*c3t + 68*c4t + \
            6*c5t + 36)/den    
    dV2_dt = -(189*st + 392*s2t + 267*s3t + 68*s4t + \
            6*s5t)/den
    dV3_dt = -(19*st + 8*st*ct + 2*st*c2t - 2*s2t*ct)/\
            (den/2)
    
    V = analytical_V1(u)
    dt_dx = -st/r
    dt_dy = ct/r

    dV1_V = dV1_dt*dt_dx*V[0] + dV1_dt*dt_dy*V[1]
    dV2_V = dV2_dt*dt_dx*V[0] + dV2_dt*dt_dy*V[1]
    dV3_V = dV3_dt*dt_dx*V[0] + dV3_dt*dt_dy*V[1]

    return reshape([dV1_V, dV2_V, dV3_V],[d,n])

if __name__=="__main__":
#def test_V1():
    s = [1.,Inf]
    u = rand(3,1)
    n = 1000
    u0 = step(u,s,n)[0,:,-1]
    u0 = u0.reshape(3,1)
    u_trj = step(u0,s,n)[0]
    
    d, n = u_trj.shape
    t_trj = arctan2(u_trj[1], u_trj[0])
    t_dir_trj = reshape([-sin(t_trj), cos(t_trj), \
            zeros(n)], [d, n])
    du_trj = dstep(u_trj, s)
    v_trj = zeros((n,d))
    v_trj[-1] = [rand(), rand(), 0.]
    v_trj[-1] /= norm(v_trj[-1])
    for i in range(n):
        v_trj[i] = dot(du_trj[i-1],v_trj[i-1])
        v_trj[i] /= norm(v_trj[i])
    clv_trj = clvs(u_trj, du_trj, 1)
    ddu_trj = d2step(u_trj,s)
    W1 = dclv_clv(clv_trj, du_trj, ddu_trj)[:,:,0].T
    W1_ana = analytical_W1(u_trj) 
