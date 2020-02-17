from numpy import *
import harminv
from scipy import interpolate
def step(u, s=[0.,0.], n=1, m=1):
    # n: number of timesteps
    # m: number of initial conditions
    # s[0] = abs(lambda), s[1] = alpha
    # output size: (n+1)xdxm
    theta = lambda phi: 2*pi*phi - s[1]
    Psi = lambda phi: (1/pi)*arctan(s[0]*sin(theta(phi))\
            /(1. - s[0]*cos(theta(phi))))
    u_trj = empty((n+1,2,m))
    u_trj[0,0] = u[0]
    u_trj[0,1] = u[1]
    for i in range(n):
        x = u_trj[i,0]
        y = u_trj[i,1]

        psix = Psi(x)
        u_trj[i+1,0] = (2*x + y + psix) % 1
        u_trj[i+1,1] = (x + y + psix) % 1
    return u_trj
def inverse_step(u, s=[0.,0.], n=1, m=1):
    """
    n: number of timesteps
    m: number of initial conditions
    s[0] = abs(lambda), s[1] = alpha
    output size: (n+1)xdxm
    """
    #Psi1 = lambda phi: (1/pi)*arctan(-s[0]*sin(2*pi*phi +s[1])\
     #       /(1. - s[0]*cos(2*pi*phi + s[1])))
    Psi2 = lambda phi: (1/pi)*arctan(s[0]*sin(2*pi*phi-s[1])\
            /(1. - s[0]*cos(2*pi*phi - s[1])))

    u_trj = empty((n+1,2,m))
    u_trj[0,0] = u[0]
    u_trj[0,1] = u[1]
    for i in range(n):
        x = u_trj[i,0]
        y = u_trj[i,1]

        #psi1 = Psi1(x-y)
        psi2 = Psi2(x-y)
        u_trj[i+1,0] = (x - y) % 1
        u_trj[i+1,1] = (-x + 2*y - psi2) % 1
    return u_trj


def dstep(u, s=[0.,0.], m=1):
    """
    Input info:
    m: number of initial conditions
    u.shape = (d, m)
    
    Output:
    Jacobian matrices at each u
    shape: mxdxd
    """

    d = u.shape[0]
    theta = lambda phi: 2*pi*phi - s[1]
    dtheta = 2*pi
    num_t = lambda phi:  s[0]*sin(theta(phi))
    den_t = lambda phi: 1. - s[0]*cos(theta(phi))
    t = lambda phi: num_t(phi)/den_t(phi)
    Psi = lambda t: (1/pi)*arctan(t)
    dnum_t = lambda phi: s[0]*cos(theta(phi))*dtheta
    dden_t = lambda phi: s[0]*sin(theta(phi))*dtheta
    dt = lambda phi: (den_t(phi)*dnum_t(phi) - \
            num_t(phi)*dden_t(phi))/\
            (den_t(phi)**2.0) 
    dPsi_dt = lambda t: 1.0/pi/(1 + t*t)
    dPsi = lambda phi: dPsi_dt(t(phi))*dt(phi)
   
    dPsix = dPsi(u[0])
    du1_1 = 2.0 + dPsix
    du2_1 = 1.0 + dPsix
    dTu_u = ravel(vstack([du1_1, ones(m), du2_1, \
            ones(m)]), order='F')
    dTu_u = dTu_u.reshape([-1,2,2])
    return dTu_u

def clvs(u,du):
    """
    Inputs:
    u: primal trajectory, shape:dxn
    du: Jacobian trajectory, shape:nxdxd

    Outputs are: 
    P: CLV basis along u, shape:nxdxd
    the ith column at the nth location, 
    P[n,:,i] contains the ith CLV at u_n

    """

    d = u.shape[0]
    n = u.shape[1]

    v = rand(d,d)
    v /= linalg.norm(v, axis=0)
     
    P = empty((n,d,d))
    v2_trj = empty((n,2))
    r_trj = empty((n,2,2))
    l = zeros(2)
    for i in range(n):
        v = dot(dstep_trj[i],v)
        v,r = linalg.qr(v)
        l += log(abs(diag(r)))/n
        v1_trj[i] = v[:,0]
        v2_trj[i] = v[:,1]
        r_trj[i] = r

    c = array([0.,1.])
    for i in range(n-1,-1,-1):
        v2_trj[i] = c[0]*v1_trj[i] + c[1]*v2_trj[i]
        v2_trj[i] /= norm(v2_trj[i])
        c /= norm(c)
        c = linalg.solve(r_trj[i], c)

    print('Lyapunov exponents: ', l)
    return u_trj, v1_trj.T, v2_trj.T

def plot_clvs():
    fig, ax = subplots(1,1)
    s = [0.7, 0.3]
    eps = 5.e-2
    u, v1, v2 = clvs(1000,s)
    ax.plot(u[0], u[1], 'k.', ms=1)
    ax.plot([u[0] - eps*v1[0], u[0] + eps*v1[0]],\
            [u[1] - eps*v1[1], u[1] + eps*v1[1]],\
            lw=2.0, color='red')
    ax.plot([u[0] - eps*v2[0], u[0] + eps*v2[0]],\
            [u[1] - eps*v2[1], u[1] + eps*v2[1]],\
            lw=2.0, color='black')
    return fig,ax

def augmented_step(u, s=[0.,0.], n=1):
    '''
      Augmented cat map
      F(x,y,[d/dx],[d/dy]) = (step(x,y)[0], step(x,y)[1],
      Dstep(x,y)[:,0], Dstep(x,y)[:,1])
      
      The Cat map but on the tangent bundle of 
      the 2-torus.

      n: number of timesteps
      m: number of initial conditions
      s[0] = abs(lambda), s[1] = alpha
      output size: (n+1)xdxm
    '''
    theta = lambda phi: 2*pi*phi - s[1]
    Psi = lambda phi: (1/pi)*arctan(s[0]*sin(theta(phi))\
            /(1. - s[0]*cos(theta(phi))))
    d = 2
    d_aug = 2*d
    m = u.shape[1]
    u_trj = empty((n+1,d_aug,m))
    u_trj[0] = u
    for i in range(n):
        x = u_trj[i,0]
        y = u_trj[i,1] 
        psix = Psi(x)
        u_trj[i+1,0] = (2*x + y + psix) % 1
        u_trj[i+1,1] = (x + y + psix) % 1
        v = u_trj[i,d:].T
        v = v.reshape(m,d,1)
        v = matmul(dstep(vstack([x,y]),\
                s,m), v)
        u_trj[i+1,d:] = v.reshape((m,d)).T
       
    return u_trj

def daugmented_step(u, s=[0.,0.]):
    """
    Input info:
    m: number of initial conditions
    u.shape = (d_aug, m)
    d = 2*2
    Output:
    Jacobian matrices at each u
    shape: d_augxd_augxm
    """
    d = u.shape[0]
    m = u.shape[1]
    theta = lambda phi: 2*pi*phi - s[1]
    dtheta = 2*pi
    num_t = lambda phi:  s[0]*sin(theta(phi))
    den_t = lambda phi: 1. - s[0]*cos(theta(phi))
    t = lambda phi: num_t(phi)/den_t(phi)
    Psi = lambda t: (1/pi)*arctan(t)
    dnum_t = lambda phi: s[0]*cos(theta(phi))*dtheta
    dden_t = lambda phi: s[0]*sin(theta(phi))*dtheta
    dt = lambda phi: (den_t(phi)*dnum_t(phi) - \
            num_t(phi)*dden_t(phi))/\
            (den_t(phi)**2.0) 
    dPsi_dt = lambda t: 1.0/pi/(1 + t*t)
    dPsi = lambda phi: dPsi_dt(t(phi))*dt(phi)
   
    d2num_t = lambda phi: -s[0]*sin(theta(phi))*dtheta*dtheta
    d2den_t = lambda phi: s[0]*cos(theta(phi))*dtheta*dtheta
    dden_dnum = lambda phi: dnum_t(phi)*dden_t(phi)
    d2t = lambda phi: -2.0*dden_dnum(phi)/(den_t(phi))**2.0 + \
            + d2num_t(phi)/den_t(phi) + \
            2*num_t(phi)*(dden_t(phi)**2.0)/(den_t(phi)**3.0) - \
            num_t(phi)/(den_t(phi)**2.0)*d2den_t(phi)
    d2Psi_dx2 = lambda phi: 1/pi*d2t(phi)/(1 + t(phi)**2.0) - \
            1/pi*(dt(phi)**2.0)*(2*t(phi))/(1 + t(phi)**2.0)**2.0


    dPsix = dPsi(u[0])
    du1_1 = 2.0 + dPsix
    du2_1 = 1.0 + dPsix
      
    dv1_u1 = d2Psi_dx2(u[0])*u[2]
    
    d_u1 = hstack([du1_1, du2_1, dv1_u1, dv1_u1])
    d_u2 = hstack([ones(m), ones(m), zeros(m), zeros(m)])


    d_v1 = hstack([zeros(m), zeros(m), du1_1, du2_1]) 
    d_v2 = hstack([zeros(m), zeros(m), ones(m), ones(m)])

    daug_step = vstack([d_u1, d_u2, d_v1, d_v2]).reshape(d,d,m).T
    return daug_step


