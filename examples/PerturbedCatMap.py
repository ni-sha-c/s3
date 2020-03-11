from numpy import *
from scipy import interpolate
def step(u, s=[0.,0.], n=1):
    '''
    Inputs:
        u: array of initial conditions, shape:dxm
        s: parameter array, shape:2
        n: number of timesteps
        m: number of initial conditions
        s[0] = abs(lambda), s[1] = alpha
    Output:
        primal trajectory, shape: (n+1)xdxm
    '''
    d = u.shape[0]
    m = u.shape[1]
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
def inverse_step(u, s=[0.,0.], n=1):
    '''
    Inputs:
        u: array of initial conditions, shape:dxm
        s: parameter array, shape:2
        n: number of timesteps
        m: number of initial conditions
        s[0] = abs(lambda), s[1] = alpha
    Output:
        primal trajectory, shape: (n+1)xdxm
    '''
    d = u.shape[0]
    m = u.shape[1]
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


def dstep(u, s=[0.,0.]):
    """
    Input info:
    s: parameters, shape:2
    m: number of initial conditions
    u.shape = (d, m)
    
    Output:
    Jacobian matrices at each u
    shape: mxdxd
    """
    m = u.shape[1]
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
def lyapunov_exponents(u,Du,dim):
    """
    Inputs:
    u: primal trajectory, shape:dxn
    Du: Jacobian trajectory, shape:nxdxd
    dim: number of LEs to calculate

    Outputs: 
    L: Lyapunov exponents, shape:(dim,)

    """

    d = u.shape[0]
    n = u.shape[1]
    
    P = empty((n,d,dim))
    #P[0] = vstack([random.rand(2,dim),\
    #        zeros((2,dim))])
    P[0] = random.rand(d,dim)
    P[0] /= linalg.norm(P[0], axis=0)

    R = empty((n,dim,dim))
    l = zeros(dim)
    for i in range(n-1):
        P[i+1] = dot(Du[i],P[i])
        P[i+1],R[i+1] = linalg.qr(P[i+1])
        l += log(abs(diag(R[i+1])))/(n-1)

    return l

def clvs(u,Du,dim):
    """
    Inputs:
    u: primal trajectory, shape:dxn
    Du: Jacobian trajectory, shape:nxdxd
    dim: number of CLVs to calculate

    Outputs are: 
    P: CLV basis of a dim-dimensional subspace 
    along u, shape:nxdxdim
    the ith column at the nth location, 
    P[n,:,i] contains the ith CLV at u_n

    """

    d = u.shape[0]
    n = u.shape[1]
    
    P = empty((n,d,dim))
    #P[0] = vstack([random.rand(2,dim),\
    #        zeros((2,dim))])
    P[0] = random.rand(d,dim)
    P[0] /= linalg.norm(P[0], axis=0)

    R = empty((n,dim,dim))
    l = zeros(dim)
    for i in range(n-1):
        P[i+1] = dot(Du[i],P[i])
        P[i+1],R[i+1] = linalg.qr(P[i+1])
        l += log(abs(diag(R[i+1])))/(n-1)
    c = eye(dim)
    for i in range(n-1,0,-1):
        P[i] = dot(P[i],c)
        P[i] /= linalg.norm(P[i],axis=0)
        c /= linalg.norm(c,axis=0)
        c = linalg.solve(R[i], c)

    #stop
    print('Lyapunov exponents: ', l)
    return P

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
                s), v)
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
    shape: mxd_augxd_aug
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


