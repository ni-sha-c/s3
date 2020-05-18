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
def cart_to_cyl(x,y):
    return [sqrt(x*x + y*y), arctan2(y, x) % (2*pi)]
def cyl_to_cart(r,t):
    return [r*cos(t), r*sin(t)]
'''
def dstep(u, s=[1.4,0.3]):
    d, m = u.shape
    du_trj = empty((d,d,m))
    a, b = s
    du_trj[0,0] = -a*2*u[0]
    du_trj[1,0] = 1.0
    du_trj[0,1] = b
    du_trj[1,1] = 0.0
    return du_trj.T
def d2step(u, s=[1.4,0.3]):
    d, m = u.shape
    ddu_trj = zeros((d,d,d,m))
    a, b = s
    ddu_trj[0,0,0] = -2*a
    return ddu_trj
'''
