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

