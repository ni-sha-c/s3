import sys
sys.path.insert(0,'../examples/')
sys.path.insert(0,'../src/')
from solenoid import *
from clvs import *
from numpy import *
from scipy.interpolate import *
from matplotlib import *
from matplotlib.collections import LineCollection
from numpy.linalg import *
def analytical_V1(u):
    x,y,z = u
    d, n = u.shape
    r,t = sqrt(x*x + y*y), (arctan2(y,x) % (2*pi))
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
    tangent_vector_field = tangent_vector_field/\
            norm(tangent_vector_field,\
            axis=0)
    return tangent_vector_field

def analytical_jacobian(u):
    x,y,z = u
    d, n = u.shape
    r,t = sqrt(x*x + y*y), (arctan2(y,x) % (2*pi))
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
    r,t = sqrt(x*x + y*y), (arctan2(y,x) % (2*pi))
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
    dt_dx = -s2t/r
    dt_dy = c2t/r

    dV1_V = dV1_dt*dt_dx*V[0] + dV1_dt*dt_dy*V[1]
    dV2_V = dV2_dt*dt_dx*V[0] + dV2_dt*dt_dy*V[1]
    dV3_V = dV3_dt*dt_dx*V[0] + dV3_dt*dt_dy*V[1]

    return reshape([dV1_V, dV2_V, dV3_V],[d,n])


#if __name__=="__main__":
def test_clvs():
    d = 3
    u = rand(d,1)
    n = 1000
    s = [1.,Inf]
    u_trj = step(u, s, n)[0]
    x, y, z = u_trj
    r,t = cart_to_cyl(x,y)
    d, n = shape(u_trj)
    d_u = 1
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, d_u)
    st, ct = sin(t), cos(t)
    s2t, c2t = st[1:], ct[1:]
    st, ct = st[:-1], ct[:-1]
    v = rand(d)
    v[-1] = 0.
    v1 = zeros((n,d))
    v1[0] = v
    le = 0.
    for i in range(n-1):
        v = dot(du_trj[i],v)
        le += log(norm(v))/n
        v /= norm(v)
        v1[i+1] = v 

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
    s = [1.0, Inf]
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

#if __name__=="__main__":
def plot_V1():
    """
    When the contraction factor s[1] is Infty, the Solenoid 
    map can be parameterized using a single parameter corresponding to the theta coordinate. For this case, the analytical 
    curvature is computed by taking the attractor to be a 3D space curve parameterized by t.
    This function verifies that the unit tangent vector field 
    to this space curve is indeed the numerically computed 
    1st CLV. Note that 
        `step` returns an mxdxn primal trajectories of 
        length n starting at m different initial conditions
        `clvs` returns an nxdxd_u tensor made up of 
        a length n trajectory of d_u CLVs.
    """
    s = [1.,Inf]
    u = rand(3,1)
    n = 1000
    u_trj = step(u,s,n)[0]
    d, n = u_trj.shape
    d_u = 1
    du_trj = dstep(u_trj, s)
    u_trj = u_trj[:-1]
    du_trj = du_trj[:,:-1,:-1]
    clv_trj = clvs(u_trj, du_trj, d_u)
    n_spinup = 100
    v1 = clv_trj[n_spinup:-n_spinup,:,0].T
    # gamma is the attractor curve
    x, y = u_trj[0,n_spinup:-n_spinup],\
            u_trj[1,n_spinup:-n_spinup]
    r, t = cart_to_cyl(x,y) 
    st, ct = sin(t), cos(t)
    s2t, c2t = sin(2*t), cos(2*t)
    gamma = lambda t: vstack([(s[0] + cos(t)/2)*cos(2*t),\
            (s[0] + cos(t)/2)*sin(2*t),\
            sin(t)/2])
    dtdx = lambda x, y: -y/(x*x + y*y)
    dtdy = lambda x, y: x/(x*x + y*y)
    gamma_dot = vstack([-2*s2t* - s2t*ct - 0.5*st*c2t,\
            2*c2t + c2t*ct - 0.5*st*s2t,\
            0.5*ct])
    
    gamma_dot_x = gamma_dot[0]*dtdx(x,y)
    gamma_dot_y = gamma_dot[1]*dtdy(x,y)
    gamma_dot = vstack([gamma_dot_x, gamma_dot_y])
    ngamma_dot = norm(gamma_dot, axis=0)
    gamma_dot /= ngamma_dot
    gamma_dot = gamma_dot[:,:-1]

    x, y = x[1:], y[1:]
    st, ct = st[1:], ct[1:]
    v1 = v1[:,1:]
    r_dir = vstack([ct, st])
    theta_dir = vstack([-st, ct])
    '''
    fig, ax = subplots(1,1)
    eps = 5.e-2
    n = x.shape[0]
    ax.plot([x - eps*v1[0], x + eps*v1[0]],\
            [y - eps*v1[1], y + eps*v1[1]],'r.-',ms=10)
    
    ax.plot([x - eps*gamma_dot[0], x + eps*gamma_dot[0]],\
            [y - eps*gamma_dot[1], y + eps*gamma_dot[1]],\
            '+-',color='purple',ms=10)
    ax.plot([zeros(n), x],[zeros(n), y],color='gray',alpha=0.1) 
    # v1 is not aligned with t, nor is gamma_dot
    # v1 is more aligned with gamma_dot than either 
    # with t.
    # draw circle of radius rad
    rad1 = 1.5
    xx = linspace(-rad1,rad1,10000)
    yy = sqrt(rad1*rad1 - xx*xx)
    ax.plot(xx, yy,'-',color='gray',alpha=0.2)
    ax.plot(xx, -yy,'-',color='gray',alpha=0.2)
    rad2 = 0.5
    xx = linspace(-rad2,rad2,10000)
    yy = sqrt(rad2*rad2 - xx*xx)
    ax.plot(xx, yy,'-',color='gray',alpha=0.2)
    ax.plot(xx, -yy,'-',color='gray',alpha=0.2)
    ax.axis('scaled')
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.grid(True) 
    '''
#if __name__=="__main__":
def test_W1():
    """
    This function computes analytically the curvature 
    at various points on the attractor and compares against 
    the numerical values obtained from the algorithm in 
    dclv_clv, for the directional derivative of the 1st 
    CLV along itself.
    """
    s = [1.,1.e10]
    u = rand(3,1)
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
    # gamma is the attractor curve
    x, y = u_trj[0,n_spinup:-n_spinup],\
            u_trj[1,n_spinup:-n_spinup]
    r, t = cart_to_cyl(x,y) 
    st, ct = sin(t), cos(t)
    s2t, c2t = sin(2*t), cos(2*t)
    s0 = s[0]
    gamma_dot = vstack([(s0 + ct/2)*(-2.0*s2t) - st*c2t/2,\
            (s0 + ct/2)*(2*c2t) - st*s2t/2,\
            ct/2])
    ngamma_dot = norm(gamma_dot, axis=0)
    gamma_dot /= ngamma_dot
    ngamma_dot = ngamma_dot[:-1]
    gamma_dot = gamma_dot[:,:-1]
    gamma_ddot = vstack([-0.5*(8*s0 + 5*ct)*c2t + 2*st*s2t,\
            -2*c2t*st - (1/2)*(8*s0 + 5*ct)*s2t,\
            -st/2])
    ngamma_ddot = norm(gamma_ddot,axis=0)
    gamma_ddot = gamma_ddot[:,:-1]
    ngamma_ddot = ngamma_ddot[:-1] 
   
  
    x, y = x[1:], y[1:]
    st, ct = st[1:], ct[1:]
    s2t, c2t = s2t[1:], c2t[1:]
    v1 = v1[:-1,1:]
    n = x.shape[0]
    eps=array([-1E-2, 1E-2]).reshape([1,2,1])
    u = vstack([x,y])
    W1 = W1[n_spinup:-n_spinup,:,0].T
    W1 = W1[:,1:]
    segments = u.T.reshape([-1,1,2]) + eps * v1.T.reshape([-1,1,2])
    curvature = sqrt((ngamma_dot**2.0)*(ngamma_ddot**2.0) -\
            sum(gamma_dot*gamma_ddot,axis=0)**2.0)/ngamma_dot**3.0
    #curvature = ngamma_ddot/ngamma_dot**2.0
    

    curvature_num = norm(W1, axis=0) 
    assert(allclose(curvature, curvature_num,rtol=0.05))

    lc = LineCollection(segments, cmap=plt.get_cmap('RdBu'), \
            norm=plt.Normalize(min(curvature), max(curvature)))
    lc.set_array(curvature)
    lc.set_linewidth(1)
    lc1 = LineCollection(segments, cmap=plt.get_cmap('RdBu'), \
            norm=plt.Normalize(min(curvature_num), max(curvature_num)))
    lc1.set_array(curvature_num)
    lc1.set_linewidth(1)


    fig, ax = subplots(1,2)    
    ax[0].add_collection(lc) 
    ax[1].add_collection(lc1) 
    # v1 is not aligned with t, nor is gamma_dot
    # v1 is more aligned with gamma_dot than either 
    # with t.
    # draw circle of radius rad
    rad1 = 1.5
    xx = linspace(-rad1,rad1,10000)
    yy = sqrt(rad1*rad1 - xx*xx)
    ax[0].plot(xx, yy,'-',color='gray',alpha=0.2)
    ax[0].plot(xx, -yy,'-',color='gray',alpha=0.2)
    
    ax[1].plot(xx, yy,'-',color='gray',alpha=0.2)
    ax[1].plot(xx, -yy,'-',color='gray',alpha=0.2)
    rad2 = 0.5
    xx = linspace(-rad2,rad2,10000)
    yy = sqrt(rad2*rad2 - xx*xx)
    ax[0].plot(xx, yy,'-',color='gray',alpha=0.2)
    ax[0].plot(xx, -yy,'-',color='gray',alpha=0.2)
    
    ax[1].plot(xx, yy,'-',color='gray',alpha=0.2)
    ax[1].plot(xx, -yy,'-',color='gray',alpha=0.2)
    ax[0].axis('scaled')
    ax[1].axis('scaled')
   
    ax[0].set_title('Analytical curvature', fontsize=30)
    ax[1].set_title('Numerical curvature', fontsize=30)
    ax[0].xaxis.set_tick_params(labelsize=30)
    ax[0].yaxis.set_tick_params(labelsize=30)
    
    ax[1].xaxis.set_tick_params(labelsize=30)
    ax[1].yaxis.set_tick_params(labelsize=30)
    ax[0].grid(True) 
    ax[1].grid(True) 
    cbar = fig.colorbar(lc, cmap=get_cmap('RdBu'),norm=Normalize(min(curvature), max(curvature)),ax=ax[0])
    cbar.ax.tick_params(labelsize=30) 
    cbar1 = fig.colorbar(lc1, cmap=get_cmap('RdBu'),norm=Normalize(min(curvature), max(curvature)),ax=ax[1])
    cbar1.ax.tick_params(labelsize=30) 

if __name__=="__main__":
    s = [1.,Inf]
    u = rand(3,1)
    n = 10000
    u0 = step(u,s,n)[0,:,-1]
    u0 = u0.reshape(3,1)
    u_trj = step(u0,s,n)[0]

    d, n = u_trj.shape
    x, y, z = u_trj
    t_trj = arctan2(y,x)[1:]
    t_trj = t_trj % (2*pi)
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, 1)
    V1 = clv_trj[1:,:,0]
    V1_ana = analytical_V1(u_trj).T[:-1]
    fig, ax = subplots(1,1)
    ax.plot(t_trj, abs(V1[:,0]), ".", label="numerical", \
            ms=2.0)
    ax.plot(t_trj, abs(V1_ana[:,0]), ".", \
            label="analytical", ms=2.0)
    ax.set_xlabel(r"$\theta$", fontsize=30)
    ax.set_ylabel(r"$V^1_{x_1}$", fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.axis("scaled")
    ax.grid(True)
    fig.legend(loc='lower right', bbox_to_anchor=(0.75,0.4),\
            fontsize=30,markerscale=10.0)

    fig, ax = subplots(1,1)
    ax.plot(t_trj, abs(V1[:,1]), ".", label="numerical",\
            ms=2.0)
    ax.plot(t_trj, abs(V1_ana[:,1]), ".", \
            label="analytical", ms=2.0)
    ax.set_xlabel(r"$\theta$", fontsize=30)
    ax.set_ylabel(r"$V^1_{x_2}$", fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.axis("scaled")
    ax.grid(True)
    fig.legend(fontsize=30)
    
    fig, ax = subplots(1,1)
    ax.plot(t_trj, abs(V1[:,2]), ".", label="numerical",\
            ms=2.0)
    ax.plot(t_trj, abs(V1_ana[:,2]), ".", \
            label="analytical", ms=2.0)
    ax.set_xlabel(r"$\theta$", fontsize=30)
    ax.set_ylabel("$V^1_{x_3}$", fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.axis("scaled")
    ax.grid(True)
    fig.legend(fontsize=30)
