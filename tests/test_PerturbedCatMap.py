import sys
sys.path.insert(0,'../examples/')
from PerturbedCatMap import *
def test_daugmented_step():
    d = 4
    m = 10
    u = rand(d,m)
    s = [0.7, 0.3]
    du = daugmented_step(u, s)
    n_eps = 6
    eps = logspace(-10,-2,n_eps)
    err = empty(n_eps)
    u_plus = copy(u)
    u_minus = copy(u)
    du_fd = empty((m,d,d))
    for n,ep in enumerate(eps):
        for i in range(d):
            u_plus[i] += ep
            u_minus[i] -= ep

            Tu_plus = augmented_step(u_plus,\
                s)[-1]
            Tu_minus = augmented_step(u_minus,\
                s)[-1]

            u_plus = copy(u)
            u_minus = copy(u)
            
            du_fd[:,:,i] = (Tu_plus - Tu_minus).T/(2*ep)
        err[n] = linalg.norm(abs(du_fd - du))
    fig, ax = subplots(1,1)
    ax.loglog(eps,err,'o-')
    return err



