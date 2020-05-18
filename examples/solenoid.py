from numpy import *
def step(u, s=[0.], n=1):
    d, m = u.shape
    u_trj = empty((n+1,d, m))
    u_trj[0] = u
    s0 = s[0]
    for i in range(1,n+1):
        r, t = cart_to_cyl(u_trj[i-1,0], u_trj[i-1,1])
        z = u_trj[i-1,2]

        r_next = s0 + (r - s0)/4 + cos(t)/2
        t_next = 2*t
        z_next = z/4 + sin(t)/2

        u_trj[i,0], u_trj[i,1] = cyl_to_cart(r_next,\
                t_next)
        u_trj[i,2] = z_next
    return u_trj

def dstep(u, s=[0]):
    d, m = u.shape
    du_trj = empty((d,d,m))
    s0 = s[0]
    
    x,y,z = u[0], u[1], u[2]
    r, t = cart_to_cyl(x,y)
    ct = cos(t)
    st = sin(t)
    
    r1 = s0 + (r - s0)/4 + ct/2
    t1 = 2*t
    z1 = z/4 + st/2

    dx1, dy1 = dcyl_to_cart(r1, t1)
    dr, dt = dcart_to_cyl(x,y)

    dr1_r = (1/4)*ones(m) 
    dr1_t = -st/2
    dt1_t = 2*ones(m)
    dz1_z = (1/4)*ones(m)
    dz1_t = ct/2

    dr1_x = dr1_r*dr[0] + dr1_t*dt[0]
    dt1_x = dt1_t*dt[0]
    dz1_x = dz1_t*dt[0]

    dr1_y = dr1_r*dr[1] + dr1_t*dt[1]
    dt1_y = dt1_t*dt[1]
    dz1_y = dz1_t*dt[1]

    du = zeros((d,d,m))
    du[0,0] = dx1[0]*dr1_x + dx1[1]*dt1_x
    du[0,1] = dy1[0]*dr1_x + dy1[1]*dt1_x
    du[0,2] = dz1_x

    du[1,0] = dx1[0]*dr1_y + dx1[1]*dt1_y
    du[1,1] = dy1[0]*dr1_y + dy1[1]*dt1_y
    du[1,2] = dz1_y

    du[2,2] = 1/4*ones(m)

    return du.T

def d2step(u, s=[0.]):
    d, m = u.shape
    ddu_trj = zeros((m,d,d,d))
    eps = 1.e-5
    ddu_trj[:,0] = (dstep(u + eps*reshape([1.,0.,0.],\
            [3,1]), s) - dstep(u - eps*reshape([1.,0.,0.],\
            [3,1]),s))/(2*eps)
    ddu_trj[:,1] = (dstep(u + eps*reshape([0.,1.,0.],\
            [3,1]), s) - dstep(u - eps*reshape([0.,1.,0.],\
            [3,1]),s))/(2*eps)
    ddu_trj[:,2] = (dstep(u + eps*reshape([0.,0.,1.],\
            [3,1]), s) - dstep(u - eps*reshape([0.,0.,1.],\
            [3,1]),s))/(2*eps)
    return ddu_trj

def cart_to_cyl(x,y):
    return [sqrt(x*x + y*y), arctan2(y, x) % (2*pi)]
def cyl_to_cart(r,t):
    return [r*cos(t), r*sin(t)]
def dcyl_to_cart(r,t):
    return vstack([cos(t), -r*sin(t)]), \
           vstack([sin(t), r*cos(t)])
def dcart_to_cyl(x,y):
    r2 = x*x + y*y
    r = sqrt(r2)
    return vstack([x/r, y/r]),\
            vstack([-y/r2, x/r2])
            

