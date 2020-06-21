from numpy import *
def step(x, s=[1.,0.], n=1):
    h, gamma = s
    d, m = x.shape
    assert(d==1)
    x_trj = empty((n+1, d, m)) 
    x_trj[0] = x
    for i in range(n):
        x_trj[i+1] = (h - \
                abs(0.5 - x_trj[i]) - \
                (h - 0.5)*(abs(1. - 2.*x_trj[i]))**gamma) % 1
    return x_trj



