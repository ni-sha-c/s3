from numpy import *
import sys
def step(u, s=zeros(4), n=1):
    '''
    Inputs:
        u: array of initial conditions, shape:dxm
        s: parameter array, shape:4
        n: number of timesteps
        m: number of initial conditions
    Output:
        primal trajectory, shape: (n+1)xdxm
    '''
    d = u.shape[0]
    m = u.shape[1]
    u_trj = empty((n+1,2,m))
    u_trj[0,0] = u[0]
    u_trj[0,1] = u[1]
    vx, vy = 8, 5
    mag = 2*pi*(vx**2 + vy**2)
    for i in range(n):
        x = u_trj[i,0]
        y = u_trj[i,1]
        
        psi_u = sin(2*pi*(vx*x + vy*y))/mag
        psi_s = sin(2*pi*(vy*x - vx*y))/mag

        u_trj[i+1,0] = (2*x + y + s[0]*psi_s*vy + \
                s[1]*psi_s*vx + s[2]*psi_u*vy + \
                s[3]*psi_u*vx) % 1
        u_trj[i+1,1] = (x + y - s[0]*psi_s*vx + \
                s[1]*psi_s*vy - s[2]*psi_u*vx + \
                s[3]*psi_u*vy) % 1
    return u_trj.T
def dstep(u, s=[0.,0.]):
    """
    Input info:
    s: parameters, shape:4
    m: number of initial conditions
    u.shape = (d, m)
    
    Output:
    Jacobian matrices at each u
    shape: mxdxd
    """
    x, y = u
    vx, vy = 8, 5
    mag = 2*pi*(vx**2 + vy**2)
    dpsi_u = cos(2*pi*(vx*x + vy*y))/mag
    dpsi_u_x = dpsi_u*2*pi*vx 
    dpsi_u_y = dpsi_u*2*pi*vy
    dpsi_s = cos(2*pi*(vy*x - vx*y))/mag
    dpsi_s_x = dpsi_s*2*pi*vy 
    dpsi_s_y = -dpsi_s*2*pi*vx

    du1_x = 2.0 + (s[0]*vy + s[1]*vx)*dpsi_s_x + \
            (s[2]*vy + s[3]*vx)*dpsi_u_x
    du1_y = (s[0]*vy + s[1]*vx)*dpsi_s_y + \
            (s[2]*vy + s[3]*vx)*dpsi_u_y
    du2_x = (-s[0]*vx + s[1]*vy)*dpsi_s_x + \
            (-s[2]*vx + s[3]*vy)*dpsi_u_x
    du2_y = 1.0 + (-s[0]*vx + s[1]*vy)*dpsi_s_y + \
            (-s[2]*vx + s[3]*vy)*dpsi_u_y

    dTu_u = vstack([du1_x, du1_y, du2_x, \
            du2_y])
    dTu_u = dTu_u.T.reshape([-1,2,2])
    return dTu_u

def d2step(u_trj, s):
    """
    This function computes D^2 varphi
    along a trajectory using finite 
    difference
    """
    d, n = u_trj.shape
    eps = 1.e-4
    u_trj_x_p = u_trj + \
            reshape([eps*ones(n), zeros(n)], \
            [2,n])
    u_trj_x_m = u_trj - \
            reshape([eps*ones(n), zeros(n)], \
            [2,n])

    u_trj_y_p = u_trj + \
            reshape([zeros(n), eps*ones(n)], \
            [2,n])
    u_trj_y_m = u_trj - \
            reshape([zeros(n), eps*ones(n)], \
            [2,n])


    ddu_dx_trj =  (dstep(u_trj_x_p, s) - \
            dstep(u_trj_x_m, s))/(2.0*eps)

    ddu_dy_trj =  (dstep(u_trj_y_p, s) - \
            dstep(u_trj_y_m, s))/(2.0*eps)

    return reshape(hstack([ddu_dx_trj, ddu_dy_trj]),\
            [n,d,d,d])

