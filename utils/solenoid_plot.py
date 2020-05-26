import sys
sys.path.insert(0,'../examples/')
sys.path.insert(0,'../src/')
from solenoid import *
from clvs import *
from numpy import *
from scipy.interpolate import *
from matplotlib import *
from matplotlib.collections import LineCollection 
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

if __name__=="__main__":
#def plot_V1():
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
    s = [1.,1.e10]
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
    gamma_dot /= norm(gamma_dot,axis=0)
    gamma_dot = gamma_dot[:,:-1]
    x, y = x[1:], y[1:]
    st, ct = st[1:], ct[1:]
    v1 = v1[:,1:]
    r_dir = vstack([ct, st])
    theta_dir = vstack([-st, ct])
    v1_proj = v1 - diag(dot(vstack([ct,st]).T,v1))*r_dir
    v1_proj /= norm(v1_proj, axis=0)
    assert(allclose(v1_proj, theta_dir))
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
    n = 10000
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
    gamma_dot = vstack([-2*s2t* - s2t*ct - 0.5*st*c2t,\
            2*c2t + c2t*ct - 0.5*st*s2t,\
            0.5*ct])
    gamma_dot /= norm(gamma_dot,axis=0)
    gamma_dot = gamma_dot[:,:-1]
    x, y = x[1:], y[1:]
    st, ct = st[1:], ct[1:]
    s2t, c2t = s2t[1:], c2t[1:]
    v1 = v1[:,1:]
    gamma_ddot = vstack([-0.5*(8 + 5*ct)*c2t + 2*st*s2t,\
            -2*c2t*st - (1/2)*(8 + 5*ct)*s2t,\
            -st/2])
    n = x.shape[0]
    assert(allclose(gamma_ddot[0]*c2t + \
            gamma_ddot[1]*s2t + \
            2.5*ct, -4*ones(n)))
    eps=array([-1E-2, 1E-2]).reshape([1,2,1])
    u = vstack([x,y])
    v1 = v1[:-1]
    W1 = W1[n_spinup:-n_spinup,:,0].T
    W1 = W1[:,1:]
    segments = u.T.reshape([-1,1,2]) + eps * v1.T.reshape([-1,1,2])
    curvature = norm(gamma_ddot, axis=0)
    curvature_ana = sqrt(20.0*ct + c2t + 85/4)
    curvature_num = norm(W1, axis=0) 
    assert(allclose(curvature, curvature_ana))

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


