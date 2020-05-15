from numpy import *
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

def dclv_clv(clv_trj, du_trj, ddu_trj):
    """
    This function computes the derivative of each 
    clv along itself. Here n is the size of the trajectory 
    over which data is input, d_u is the number of CLVs to be 
    differentiated, d is the dimension of the system. 

    The inputs are:
    clv_trj: shape:nxdxd_u, the d_u unstable CLVs computed 
    along a size n trajectory
    du_trj: shape: nxdxd, the Jacobian matrix along a size 
    n trajectory
    ddu_trj: shape: nxdxdxd, the second derivative of the primal,
    ddu_trj[i,j,k,l] = D_j D_l (k-th component of varphi)(u_i)

    The output is:
    W: shape:nxdxd_u
    W[i,j,k] = D_{V_k} (j-th component of V_k)(u_i)
    """
    n, d, d_u = clv_trj.shape
    z_trj = empty((n,d_u))
    W1 = empty((n,d,d_u))
    W1[0] = random.rand(d,d_u)
    for i in range(n-1):
        clvsi = clv_trj[i]
        W1i = W1[i]
        z_trj[i] = linalg.norm(dot(du_trj[i],\
                clvsi), axis=0)
        z2 = z_trj[i]*z_trj[i]
        ddui = ddu_trj[i].T
        ddu_dpx = dot(dot(ddui, clvsi).T[0], clvsi)/z2
        dclv_dpx = dot(du_trj[i], W1i)/z2
        W1i_part = ddu_dpx + dclv_dpx
        clv_ip1 = clv_trj[i+1]
        dot_clv_ip1 = linalg.norm(clv_ip1,axis=0)**2.0
        dz_dx = diag(dot(W1i_part.T, clv_ip1))*z_trj[i]/dot_clv_ip1
        W1[i+1] = W1i_part - dz_dx/z_trj[i]*clv_ip1
    return W1

