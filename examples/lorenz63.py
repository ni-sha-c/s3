from numpy import *
import sys
dt = 0.005
def step(u, s=[10.,8./3,28.], n=1):
    '''
    Inputs:
        u: array of initial conditions, shape:mxd
        s: parameter array, shape:4
        n: number of timesteps
        m: number of initial conditions
    Output:
        primal trajectory, shape: (n+1)xdxm
    '''
    m = u.shape[0]
    d = u.shape[1]
    u_trj = empty((n+1,d,m))
    u_trj[0] = u
    sigma, rho, beta = s
    for i in range(n):
        x = u_trj[i,0]
        y = u_trj[i,1]
        z = u_trj[i,2]

        dxdt = sigma*(y - x)
        dydt = x*(rho - z) - y
        dzdt = x*y - beta*z

        u_trj[i+1,0] = x + dt*dxdt
        u_trj[i+1,1] = y + dt*dydt
        u_trj[i+1,2] = z + dt*dzdt

    return u_trj
def dstep(u, s=[10.,8./3,28.]):
    """
    Input info:
    s: parameters, shape:4
    m: number of initial conditions
    u.shape = (m, d)
    
    Output:
    Jacobian matrices at each u
    shape: mxdxd
    """
    m, d = u.shape
    x, y, z = u.T

    sigma, rho, beta = s
    dTx_dx = 1.0 - dt*sigma  
    dTx_dy = dt*sigma
    dTx_dz = zeros(m)
    dTy_dx = dt*(rho - z)
    dTy_dy = 1.0 - dt
    dTy_dz = dt*(-x)
    dTz_dx = dt*y
    dTz_dy = dt*x
    dTz_dz = 1.0 - beta

    dTu_u = vstack([dTx_dx, dTx_dy, dTx_dz, \
                    dTy_dx, dTy_dy, dTy_dz, \
                    dTz_dx, dTz_dy, dTz_dz])
    dTu_u = dTu_u.T.reshape([-1,d,d])
    return dTu_u

def d2step(u, s):
    """
    This function computes D^2 varphi
    at the points u
    """
    n, d = u.shape
    ddu = zeros((d,d,d,n))
    ddu[1, 0, 2] = -dt
    ddu[1, 2, 0] = -dt
    ddu[2, 0, 1] = dt
    ddu[2, 1, 0] = dt
    return ddu

