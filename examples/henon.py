from numpy import *
def step(u, s=[1.4,0.3], n=1):
    d, m = u.shape
    u_trj = empty((n+1,d, m))
    u_trj[0] = u
    a, b = s
    for i in range(1,n+1):
        u_trj[i,0] = 1.0 - a*u_trj[i-1,0]*u_trj[i-1,0] + \
                u_trj[i-1,1]
        u_trj[i,1] = b*u_trj[i-1,0]
    return u_trj.T
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
    return ddu_trj.T
def fixed_point(s):
    a,b = s
    x = 1/(2*a)*(-(1-b) + \
            sqrt((1-b)**2.0 + 4*a))
    y = b*x
    return reshape([x,y],[2,1])

