from numpy import *
def step(x, s=[1.,0.], n):
    h, gamma = s
    d, m = x.shape
    assert(d==1)
    x_trj = empty((n+1, d, m)) 
    x_trj[0] = x
    for i in range(n):
        x_trj[i+1] = h*(1. - \
                abs(0.5 - x_trj[i]) - \
                (0.25 - x_trj[i]/2)**gamma) % 1
    return x_trj



