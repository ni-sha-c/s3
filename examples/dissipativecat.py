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
    st_x = s[0]*vy + s[1]*vx
    st_y = s[0]*(-vx) + s[1]*vy
    us_x = s[2]*vy + s[3]*vx
    us_y = s[2]*(-vx) + s[3]*vy
    for i in range(n):
        x = u_trj[i,0]
        y = u_trj[i,1]
        
        psi_st = sin(2*pi*(vy*x - vx*y))/mag
        psi_us = sin(2*pi*(vx*x + vy*y))/mag


        u_trj[i+1,0] = (2*x + y + st_x*psi_st + \
                        us_x*psi_us) % 1
        u_trj[i+1,1] = (x + y + st_y*psi_st + \
                        us_y*psi_us) % 1
    return u_trj.T
def dstep(u, s=zeros(4)):
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

    dpsi_st = cos(2*pi*(vy*x - vx*y))/mag
    dpsi_st_x = dpsi_st*2*pi*vy 
    dpsi_st_y = -dpsi_st*2*pi*vx


    dpsi_us = cos(2*pi*(vx*x + vy*y))/mag
    dpsi_us_x = dpsi_us*2*pi*vx 
    dpsi_us_y = dpsi_us*2*pi*vy
    
    st_x = s[0]*vy + s[1]*vx
    st_y = s[0]*(-vx) + s[1]*vy
    us_x = s[2]*vy + s[3]*vx
    us_y = s[2]*(-vx) + s[3]*vy

    du1_x = 2.0 + st_x*dpsi_st_x + us_x*dpsi_us_x
    du1_y = st_x*dpsi_st_y + us_x*dpsi_us_y
    du2_x = st_y*dpsi_st_x + us_y*dpsi_us_x
    du2_y = 1.0 + st_y*dpsi_st_y + us_y*dpsi_us_y

    dTu_u = vstack([du1_x, du1_y, du2_x, \
            du2_y])
    dTu_u = dTu_u.T.reshape([-1,2,2])
    return dTu_u

def d2step(u, s):
    """
    This function computes D^2 varphi
    at the points u
    """
    d, n = u.shape
    eps = 1.e-6
    ddu = zeros((d,d,d,n))
    
    vx, vy = 8, 5
    st_x = s[0]*vy + s[1]*vx
    st_y = -s[0]*vx + s[1]*vy
    us_x = s[2]*vy + s[3]*vx
    us_y = -s[2]*vx + s[3]*vy
    
    mag = 2*pi*(vx**2 + vy**2)
    fpi2 = 4*(pi**2)
    x, y = u
    d2psi_st_dx2 = -fpi2*vy*vy*sin(2*pi*(vy*x - vx*y))/mag
    d2psi_st_dxdy = fpi2*vy*vx*sin(2*pi*(vy*x - vx*y))/mag
    d2psi_st_dy2 = -fpi2*vx*vx*sin(2*pi*(vy*x - vx*y))/mag
    
    d2psi_us_dx2 = -fpi2*vx*vx*sin(2*pi*(vx*x + vy*y))/mag
    d2psi_us_dxdy = -fpi2*vy*vx*sin(2*pi*(vx*x + vy*y))/mag
    d2psi_us_dy2 = -fpi2*vy*vy*sin(2*pi*(vx*x + vy*y))/mag

    ddu11 = vstack([st_x*d2psi_st_dx2 + us_x*d2psi_us_dx2,\
                    st_x*d2psi_st_dxdy + us_x*d2psi_us_dxdy\
                    ])
    ddu21 = vstack([st_y*d2psi_st_dx2 + us_y*d2psi_us_dx2,\
                    st_y*d2psi_st_dxdy + us_y*d2psi_us_dxdy\
                    ])
    ddu12 = vstack([st_x*d2psi_st_dxdy + us_x*d2psi_us_dxdy\
            , st_x*d2psi_st_dy2 + us_x*d2psi_us_dy2])

    ddu22 = vstack([st_y*d2psi_st_dxdy + us_y*d2psi_us_dxdy\
            , st_y*d2psi_st_dy2 + us_y*d2psi_us_dy2])
                
    
    ddu = reshape([ddu11, ddu21, ddu12, ddu22],[2,2,2,-1])
    return ddu.T

