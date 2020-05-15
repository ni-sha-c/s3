import sys
sys.path.insert(0,'../examples/')
sys.path.insert(0,'../src/')
from PerturbedCatMap import *
from clvs import *
from numpy import *
from scipy.interpolate import *
from matplotlib import *
#if __name__=="__main__":
def plot_density():
    fig, ax = subplots(1,2)
    s = [0.7,0.3]
    d = 2
    u0 = random.rand(d,1)
    n = 10000
    n_maps = 100
    ubar = empty((n_maps,d))
    uvar = empty((n_maps,d))
    l = linspace(0.2,1.,n_maps)
    for i in range(n_maps):
        s[1] = l[i]
        u = step(u0,s,n) #shape: (n+1)xdx1
        u = u[1:].T[0] # shape:dxn
        ubar[i] = mean(u,axis=1) 
        uvar[i] = mean((u-ubar[i].reshape(d,1))**2.0, \
                axis=1)
    ax[0].plot(l, ubar, 'o-', ms=2, lw=2.0)
    ax[1].plot(l, uvar, 'o-', ms=2, lw=2.0)
    ax[0].set_xlabel(r'$\lambda$', fontsize=24)
    ax[1].set_xlabel(r'$\lambda$',fontsize=24)
    ax[0].set_ylabel(r'$<u>$', fontsize=24)
    ax[1].set_ylabel(r'var $u$', fontsize=24)
    ax[0].xaxis.set_tick_params(labelsize=24)
    ax[0].yaxis.set_tick_params(labelsize=24)
    ax[1].xaxis.set_tick_params(labelsize=24)
    ax[1].yaxis.set_tick_params(labelsize=24)
    return fig,ax
#if __name__=="__main__":
def sensitivity_of_les():
    # Vary lambda
    d = 2
    s = [0.7,0.3]
    u0 = random.rand(d,1)
    n = 1000
    n_maps = 30
    lamb = linspace(0.,1.,n_maps)
    alph = linspace(0.,1.,n_maps)
    les = empty((n_maps,n_maps,d))
    for i in range(n_maps):
        s[0] = lamb[i]
        print(i, s[0])
        for j in range(n_maps):
            s[1] = alph[j]
            u = step(u0,s,n) #shape: (n+1)xdx1
            u = u[1:].T[0] # shape:dxn
            du = dstep(u,s) #shape:nxdxd
            les[i,j] = lyapunov_exponents(u,du,d) #shape:d
    les = les.T[0]
    #Plot LEs at various parameter values
    fig, ax = subplots(1,2)
    n_plots = 5
    colors = cm.get_cmap('coolwarm', n_plots)
    ind = range(0,n_maps,n_plots)
    n_lamb_plots = n_maps//n_plots 
    for i, lamb_ind in enumerate(ind):
        color = colors(i/n_lamb_plots)
        text_loc = n_maps_alph//2
        ax[0].plot(alph, les[:,lamb_ind], 'o-', ms=5, lw=2.0, color=color)
        
        ax[1].plot(lamb, les[:,lamb_ind], 'o-', ms=5, lw=2.0, color=color)
        if i == 1:
            ax[0].text(alph[text_loc] - 0.01, les[:,lamb_ind][text_loc] - 0.005, "s = {0:.2f}".format(lamb[lamb_ind]), size=24, \
                color=color)
            ax[1].text(alph[text_loc] - 0.01, les[:,lamb_ind][text_loc] - 0.005, "s = {0:.2f}".format(lamb[lamb_ind]), size=24, \
                color=color)

        elif i==0:
            ax[0].text(alph[text_loc] - 0.01, les[:,lamb_ind][text_loc] + 0.005, "s = {0:.2f}".format(lamb[lamb_ind]), size=24, \
                color=color)

        else:
            ax[0].text(alph[text_loc] - 0.01, les[:,lamb_ind][text_loc] - 0.015, "s = {0:.2f}".format(lamb[lamb_ind]), size=24, \
                color=color)
    ax[0].set_xlabel('s angle', fontsize=24)
    ax[1].set_xlabel('s magnitude',fontsize=24)
    ax[0].set_ylabel(r'$\lambda^1$', fontsize=24)
    ax[1].set_ylabel(r'$\lambda^1$', fontsize=24)
    ax[0].xaxis.set_tick_params(labelsize=24)
    ax[0].yaxis.set_tick_params(labelsize=24)
    ax[1].xaxis.set_tick_params(labelsize=24)
    ax[1].yaxis.set_tick_params(labelsize=24)
    return fig,ax

def fixed_point():
    """
    Fixed point
    """

    d = 2
    s = [0.7,0.3]
    s_c = s[0]*exp(1j*s[1])
    s_cc = s_c.conjugate()
    z1_fixed = (1.0 + s_c)/\
            (1 + s_cc)
    z2_fixed = 1.0
    u0_fixed = 1/(2*pi)*arctan(\
            z1_fixed.imag/\
            z1_fixed.real) % 1
    u1_fixed = 1/(2*pi)*arctan(\
            z2_fixed.imag/\
            z2_fixed.real) % 1
    u_fixed = array([u0_fixed,\
            u1_fixed]).reshape(2,1)
    du_fixed = dstep(u_fixed,s)[0]
    l_fixed, clvs_fixed = eig(du_fixed)
    v0_fixed = clvs_fixed[:,0].reshape(2,1)
    v1_fixed = clvs_fixed[:,1]
    upert = u_fixed + 1.e-2*v0_fixed
    u = inverse_step(upert,s,10)

def test_dG():
    """
    G(v)(u) := Dvarphi(u) v/||Dvarphi(u) v||
    This function computes DG(V^i(u))(u), 
    the derivative of G at the CLVs, analytically.
    """
    u = rand(2).reshape(2,1)
    n = 10000
    #s = [0.75, 0.99]
    s = zeros(2)
    u_trj = step(u, s, n).T[0] 
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, 2) 
    u_trj = u_trj.T

    n = n+1
    n_eps = 4
    dG_0_ana = empty((n,2,2))
    dG_1_ana = empty((n,2,2))
    v0 = rand(2)
    v1 = rand(2)
    dG_0_fd = empty((n,2,2))
    dG_1_fd = empty((n,2,2))
    eps = 1.e-4
   
    for i in range(n):
     
        v0 = clv_trj[i,0]
        v1 = clv_trj[i,1]
        x_pert = zeros(2)
        y_pert = zeros(2)
        x_pert[0] += eps
        y_pert[1] += eps 

        A = du_trj[i] 
        G_0 = dot(A, v0).reshape(2,1)
        z_0 = norm(G_0)
        G_0 /= z_0
        
        G_1 = dot(A, v1).reshape(2,1)
        z_1 = norm(G_1)
        G_1 /= z_1

        
        dG_0_ana[i] = A/z_0 - dot(G_0/z_0, dot(G_0.T, A))
        dG_1_ana[i] = A/z_1 - dot(G_1/z_1, dot(G_1.T, A))

           
        G_plus_0_x = dot(A, v0 + x_pert).reshape(2,1)
        G_plus_0_x /= norm(G_plus_0_x)

        G_plus_0_y = dot(A, v0 + y_pert).reshape(2,1)
        G_plus_0_y /= norm(G_plus_0_y)

        G_minus_0_x = dot(A, v0 - x_pert).reshape(2,1)
        G_minus_0_x /= norm(G_minus_0_x)

        G_minus_0_y = dot(A, v0 - y_pert).reshape(2,1)
        G_minus_0_y /= norm(G_minus_0_y)

        G_plus_1_x = dot(A, v1 + x_pert).reshape(2,1)
        G_plus_1_x /= norm(G_plus_1_x)

        G_plus_1_y = dot(A, v1 + y_pert).reshape(2,1)
        G_plus_1_y /= norm(G_plus_1_y)

        G_minus_1_x = dot(A, v1 - x_pert).reshape(2,1)
        G_minus_1_x /= norm(G_minus_1_x)

        G_minus_1_y = dot(A, v1 - y_pert).reshape(2,1)
        G_minus_1_y /= norm(G_minus_1_y)

        dG_0_fd[i] = hstack([G_plus_0_x - G_minus_0_x, \
                    G_plus_0_y - G_minus_0_y])/(2.0*eps)

        dG_1_fd[i] = hstack([G_plus_1_x - G_minus_1_x, \
                    G_plus_1_y - G_minus_1_y])/(2.0*eps)
      
        assert(allclose(dG_0_fd[i], dG_0_ana[i]))
        assert(allclose(dG_1_fd[i], dG_1_ana[i]))
       
def plot_dG():       
#if __name__=="__main__":
    """
    G(v)(u) := Dvarphi(u) v/||Dvarphi(u) v||
    This function plots DG(V(u))(u).V^i(u).
    """
    u = rand(2).reshape(2,1)
    n = 10000
    s = [0.75, 0.99]
    #s = zeros(2)
    u_trj = step(u, s, n).T[0] 
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, 2) 
    u_trj = u_trj.T

    n = n+1
    dG = empty((n,2,2))
    dGv = empty((n,2))
    v0 = rand(2)
    v1 = rand(2)
    c = rand()
    for i in range(n):
     
        v0 = clv_trj[i,0]
        v1 = clv_trj[i,1]
        v = v0 + c*(v1-v0)
        A = du_trj[i] 
        G = dot(A, v).reshape(2,1)
        z = norm(G)
        G /= z
               
        dG[i] = A/z - dot(G/z, dot(G.T, A))
        dGv[i] = dot(dG[i], v0)

    fig, ax = subplots(1,1)
    
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    
    eps_plot = 1.e-2
    u_trj1 = u_trj.T[:,::10]
    dGv1 = (dGv.T/norm(dGv,axis=1))[:,::10]
    ax.plot([u_trj1[0] - eps_plot*dGv1[0],\
                u_trj1[0] + eps_plot*dGv1[1]],\
                [u_trj1[1] - eps_plot*dGv1[1],\
                u_trj1[1] + eps_plot*dGv1[1]],\
                lw=2.5,color='r')
    ax.set_title(r'$\partial G/\partial V(v)\cdot V^1$',fontsize=30)
    
    fig1, ax1 = subplots(1,1)
    
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    n_grid = 50
    x_x = linspace(0.,1.,n_grid)
    x_grid, y_grid = meshgrid(x_x, x_x)
    x1_trj = u_trj[:,0]
    x2_trj = u_trj[:,1]
    ndGv = norm(dGv, axis=1)
    f1 = interp2d(x1_trj, x2_trj, ndGv)

    a1 = f1(x_x, x_x).reshape(\
            n_grid,n_grid)

    plot1 = ax1.contourf(x_grid, x_grid, a1,20,vmin=-1.0,vmax=1.0)

#if __name__=="__main__":
def test_dz():
    """
    z(v)(u) := ||Dvarphi(u) v||
    This function computes dG(V^i(u))(u), 
    the derivative of G at the CLVs as 
    dG(V)(u) = G/z - (1/z*z) G (grad(z)(V)*V)
    """
    u = rand(2).reshape(2,1)
    n = 1000
    #s = [0.75, 0.99]
    s = zeros(2)
    u_trj = step(u, s, n).T[0] 
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, 2) 
    u_trj = u_trj.T

    n = n+1
    n_eps = 4
    dz_0 = empty((n_eps,n,2))
    dz_1 = empty((n_eps,n,2))
    eps = 1.e-2
    v0 = rand(2)
    v1 = rand(2)
    for k in range(n_eps):
        eps *= 1.e-1
        x_pert = zeros(2)
        y_pert = zeros(2)
        x_pert[0] += eps
        y_pert[1] += eps 

        for i in range(n):
     
            v0 = clv_trj[i,0]
            v1 = clv_trj[i,1]

            A = du_trj[i] 
            
            G_plus_0_x = dot(A, v0 + x_pert)
            z_plus_0_x = norm(G_plus_0_x)

            G_plus_0_y = dot(A, v0 + y_pert)
            z_plus_0_y = norm(G_plus_0_y)

            G_minus_0_x = dot(A, v0 - x_pert)
            z_minus_0_x = norm(G_minus_0_x)

            G_minus_0_y = dot(A, v0 - y_pert)
            z_minus_0_y = norm(G_minus_0_y)

            G_0 = dot(A, v0).reshape(2,1)
            z_0 = norm(G_0)
            G_0 /= z_0

            G_plus_1_x = dot(A, v1 + x_pert)
            z_plus_1_x = norm(G_plus_1_x)

            G_plus_1_y = dot(A, v1 + y_pert)
            z_plus_1_y = norm(G_plus_1_y)

            G_minus_1_x = dot(A, v1 - x_pert)
            z_minus_1_x = norm(G_minus_1_x)

            G_minus_1_y = dot(A, v1 - y_pert)
            z_minus_1_y = norm(G_minus_1_y)
            
            G_1 = dot(A, v1).reshape(2,1)
            z_1 = norm(G_1)
            G_1 /= z_1

            dz_0[k,i] = hstack([z_plus_0_x - z_minus_0_x, \
                    z_plus_0_y - z_minus_0_y])/(2.0*eps)
            dz_1[k,i] = hstack([z_plus_1_x - z_minus_1_x, \
                    z_plus_1_y - z_minus_1_y])/(2.0*eps) 

            if(k==3):
                assert(allclose(dz_0[k,i], dot(G_0.T, A)))
                assert(allclose(dz_1[k,i], dot(G_1.T, A)))
    '''
    fig, ax = subplots(1,2)
    dG_0 = dG_0.T
    dG_1 = dG_1.T
    ax[0].plot(dG_0[0,200], dG_0[1,200],'.',ms=10) 
    ax[0].plot(dG_0[0,2000], dG_0[1,2000],'.',ms=10)  
    
    ax[1].plot(dG_1[0,1000], dG_1[1,1000],'.',ms=10) 
    ax[1].plot(dG_1[0,200], dG_1[1,200],'.',ms=10)  
    
    ax[0].xaxis.set_tick_params(labelsize=30)
    ax[1].xaxis.set_tick_params(labelsize=30)
    ax[0].yaxis.set_tick_params(labelsize=30)
    ax[1].yaxis.set_tick_params(labelsize=30)
    '''

if __name__ == "__main__":
#def plot_dDV1cdotV1():
    u = rand(2).reshape(2,1)
    n = 10000
    s = [0.75, 0.2]
    #s = zeros(2)
    u_trj = step(u, s, n).T[0]
    d, n = shape(u_trj)
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, 1)
    du = 1
    ddu_trj = d2step(u_trj,s)
    W1 = dclv_clv(clv_trj[:,:,:du], du_trj, ddu_trj)
    
    fig, ax = subplots(1,1)
    n_spinup = 100
    v1 = clv_trj[n_spinup:,:,0].T
    eps = 5.e-3
    u = u_trj[:,n_spinup:]
    W1 = W1[n_spinup:,:,0].T
    ax.plot([u[0] - eps*v1[0], u[0] + eps*v1[0]],\
            [u[1] - eps*v1[1], u[1] + eps*v1[1]],\
            lw=1.0,label=r"$V^1$",color="red")
    ax.plot([u[0] - eps*W1[0], u[0] + eps*W1[0]],\
           [u[1] - eps*W1[1], u[1] + eps*W1[1]],\
            lw=1.0,color="blue")

    ax.axis("scaled")
    fig1, ax1 = subplots(1,1)
    angles = arctan(v1[1]/v1[0])
    nor = colors.Normalize(vmin = min(angles), \
            vmax = max(angles))
    colormap = cm.get_cmap('binary')
    cols = colormap(nor(angles))
    for i in range(n-n_spinup):
        ax1.plot(u[0,i], u[1,i],"o",\
                fillstyle="full",\
                ms=5.0,color=cols[i])
    ax1.plot([u[0] - eps*W1[0], u[0] + eps*W1[0]],\
            [u[1] - eps*W1[1], u[1] + eps*W1[1]],\
            lw=1.0,color="blue")
    ax1.axis("equal")
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)

    
def clv_along_clv():
    """
    Differentiating along CLV
    """
    u = rand(2).reshape(2,1)
    n = 100000
    s = [0.75, 0.99]
    #s = zeros(2)
    u_trj = step(u, s, n).T[0] 
    du_trj = dstep(u_trj, s)
    clv_trj = clvs(u_trj, du_trj, 2).T 
    n_origin = 50
    u_trj = u_trj.T

    origin = u_trj[n_origin]
    dist_origin = linalg.norm(u_trj - origin, axis=1)
    

    clv1_trj = clv_trj[0].T
    clv2_trj = clv_trj[1].T


    x1_axis = clv1_trj[n_origin]
    x2_axis = clv2_trj[n_origin]
    

   
    x1_perp = rand(2)
    x1_perp /= linalg.norm(x1_perp)
    x1_perp -= dot(x1_perp, x1_axis)*x1_axis
    c = dot(x2_axis, x1_perp)
    clv1_x2 = dot(clv1_trj, x1_perp)/c
    clv2_x2 = dot(clv2_trj, x1_perp)/c
    

    v_trj = u_trj - origin
    v_trj /= norm(v_trj)
    v_x2 = dot(v_trj, x1_perp)/c
    
    x2_perp = rand(2)
    x2_perp /= linalg.norm(x2_perp)
    x2_perp -= dot(x2_perp, x2_axis)*x2_axis
    c = dot(x1_axis, x2_perp)
    clv1_x1 = dot(clv1_trj, x2_perp)/c
    clv2_x1 = dot(clv2_trj, x2_perp)/c
    
    v_x1 = dot(v_trj, x2_perp)/c
    assert(allclose(v_x1.reshape(-1,1)*x1_axis + \
            v_x2.reshape(-1,1)*x2_axis, \
            v_trj))
    assert(allclose(clv1_x1.reshape(-1,1)*x1_axis + \
            clv1_x2.reshape(-1,1)*x2_axis,\
            clv1_trj))
    assert(allclose(clv2_x1.reshape(-1,1)*x1_axis + \
            clv2_x2.reshape(-1,1)*x2_axis, \
            clv2_trj))

    #Plots
    fig, ax = subplots(1,2)
    ax[0].xaxis.set_tick_params(labelsize=30)
    ax[1].xaxis.set_tick_params(labelsize=30)
    ax[0].yaxis.set_tick_params(labelsize=30)
    ax[1].yaxis.set_tick_params(labelsize=30)
    grid(True)    
   
    delta = 0.1
    w_x1 = dot(v_trj, x1_perp)
    w_x2 = dot(v_trj, x2_axis)
    condlist = (dist_origin < delta) &(abs(w_x1) < 1.e-6) 
    neighbours = u_trj[condlist]
    v_x1 = v_x1[condlist]
    clv1_trj_x = clv1_trj[condlist][:,0]
    clv1_x1 = clv1_x1[condlist]
    dist = dist_origin[condlist]
    ax[0].plot(v_x1, clv1_trj_x, 'k.', ms=20) 
    ax[1].plot(v_x1, clv1_x1, 'r.', ms=20)

    '''
    n_grid = 50
    x_x = linspace(0.,1.,n_grid)
    x_grid, y_grid = meshgrid(x_x, x_x)
    x1_trj = u_trj[:,0] - origin[0]
    x2_trj = u_trj[:,1] - origin[1]
    f1 = interp2d(x1_trj, x2_trj, angle_1)
    f2 = interp2d(x1_trj, x2_trj, angle_2)

    a1 = f1(x_x, x_x).reshape(\
            n_grid,n_grid)

    a2 = f2(x_x, x_x).reshape(\
            n_grid,n_grid)
    plot1 = ax[0].contourf(x_grid, x_grid, a1,20,vmin=-1.0,vmax=1.0)
    plot2 = ax[1].contourf(x_grid, x_grid, a2,20,vmin=-1.0,vmax=1.0)
    '''

    #for i in range(n_origin,n):
     #   ax.plot(clv1_x1[i], clv1_x2[i], 'r.', label='1st CLV', ms=10.0)
        #ax.plot(clv2_x1[i], clv2_x2[i], 'b.', label='2nd CLV', ms=10.0)
        #pause(0.001)
#if __name__=="__main__":
def plot_DVW():
    """
    Plot V^i.W^i
    """
    u = rand(2).reshape(2,1)
    n = 10000
    s = [0.75, 0.99]
    #s = zeros(2)
    u_trj = step(u, s, n).T[0] 
    du_trj = dstep(u_trj, s)
    u_trj_rev = fliplr(u_trj)
    duT_trj_rev = flipud(transpose(du_trj,[0,2,1]))

     
    clv_trj = clvs(u_trj, du_trj, 2) 
    adj_clv_trj = clvs(u_trj_rev, duT_trj_rev, 2)
    adj_clv_trj = flipud(adj_clv_trj)
    u_trj = u_trj.T
    c_trj = matmul(clv_trj, transpose(adj_clv_trj, \
            [0,2,1]))
    c0 = c_trj[:,0,0]
    c1 = c_trj[:,1,1]
    '''
    n_grid = 50
    x_x = linspace(0.,1.,n_grid)
    x_grid, y_grid = meshgrid(x_x, x_x)
    u_trj = u_trj.T
    f = interp2d(u_trj[0,10:-10], u_trj[1,10:-10], c0[10:-10])
    cx = f(x_x, x_x).reshape(\
            n_grid,n_grid)
    '''
    fig, ax = subplots(1,2)
    ax[0].xaxis.set_tick_params(labelsize=30)
    ax[0].yaxis.set_tick_params(labelsize=30)
    
    ax[1].xaxis.set_tick_params(labelsize=30)
    ax[1].yaxis.set_tick_params(labelsize=30)
    ax[0].set_title(r'$W^1\cdot V^1$',fontsize=30)
    ax[0].set_xlabel(r'$y$',fontsize=30)

    ax[1].set_title(r'$W^2\cdot V^2$',fontsize=30)
    ax[1].set_xlabel(r'$y$',fontsize=30)
    #cplot = ax.contourf(x_grid, x_grid, cx,20)
    ax[0].plot(u_trj[20:-20,1], c0[20:-20],'r.',ms=1)
    ax[1].plot(u_trj[20:-20,1], c1[20:-20],'b.',ms=1)

def plot_clvs():
    fig, ax = subplots(1,2)
    s = [0.9,0.4]
    eps = 1.e-2
    d = 2
    u0 = random.rand(d,1)
    n = 10000
    u = step(u0,s,n) #shape: (n+1)xdx1
    u = u[1:].T[0] # shape:dxn
    du = dstep(u,s) #shape:nxdxd
    P = clvs(u,du,d).T #shape:nxdxd
    v1 = P[0]
    v2 = P[1]
    ax[0].plot([u[0] - eps*v1[0], u[0] + eps*v1[0]],\
            [u[1] - eps*v1[1], u[1] + eps*v1[1]],\
            lw=2.0, color='red')
    ax[1].plot([u[0] - eps*v2[0], u[0] + eps*v2[0]],\
            [u[1] - eps*v2[1], u[1] + eps*v2[1]],\
            lw=2.0, color='black')

    ax[0].set_title('$V^1$',fontsize=24)
    
    ax[1].set_title('$V^2$',fontsize=24)
    for j in range(2):
            ax[j].xaxis.set_tick_params(labelsize=24)
            ax[j].yaxis.set_tick_params(labelsize=24)


    return fig,ax

def plot_clv_components():
    fig, ax = subplots(2,2)
    s = [0.7,0.3]
    #s = zeros(2)
    eps = 4.e-2
    d = 2
    u0 = random.rand(d,1)
    n = 20000
    u = step(u0,s,n) #shape: (n+1)xdx1
    u = u[1:].T[0] # shape:dxn
    du = dstep(u,s) #shape:nxdxd
    P = clvs(u,du,2).T #shape:nxdxd
    v1 = P[0]
    v2 = P[1]
    '''
    ax.plot(u[0], u[1], 'k.', ms=1)
    ax.plot([u[0] - eps*v1[0], u[0] + eps*v1[0]],\
            [u[1] - eps*v1[1], u[1] + eps*v1[1]],\
            lw=2.0, color='red')
    ax.plot([u[0] - eps*v2[0], u[0] + eps*v2[0]],\
            [u[1] - eps*v2[1], u[1] + eps*v2[1]],\
            lw=2.0, color='black')
    
    n_grid = 50
    x_x = linspace(0.,1.,n_grid)
    x_grid, y_grid = meshgrid(x_x, x_x)
    f = interp2d(u[0,10:], u[1,10:], v2[0,10:])
    vx = f(x_x, x_x).reshape(\
            n_grid,n_grid)
    c = ax.contourf(x_grid, x_grid, vx,20,vmin=-1.0,vmax=1.0)
    '''
    ax[0,0].tricontour(u[0,10:], u[1,10:], \
            v1[0,10:], \
            linewidths=0.5,\
            colors='k')
    ax[0,0].set_title('$V^1_1$',fontsize=24)
    cntr00 = ax[0,0].tricontourf(u[0,10:],\
            u[1,10:], v1[0,10:],\
            levels=linspace(min(v1[0,10:]),\
            max(v1[0,10:]), 50),\
            cmap="RdBu_r")
    
    ax[0,1].tricontour(u[0,10:], u[1,10:], \
            v1[1,10:], \
            linewidths=0.5,\
            colors='k')
    cntr01 = ax[0,1].tricontourf(u[0,10:],\
            u[1,10:], v1[1,10:],\
            levels=linspace(min(v1[1,10:]),\
            max(v1[1,10:]), 30),\
            cmap="RdBu_r")
    ax[0,1].set_title('$V^1_2$',fontsize=24)
    
    ax[1,0].tricontour(u[0,10:], u[1,10:], \
            v2[0,10:], \
            linewidths=0.5,\
            colors='k')
    cntr10 = ax[1,0].tricontourf(u[0,10:],\
            u[1,10:], v2[0,10:],\
            levels=linspace(min(v2[0,10:]),\
            max(v2[0,10:]), 30),\
            cmap="RdBu_r")
    ax[1,0].set_title('$V^2_1$',fontsize=24)

    ax[1,1].tricontour(u[0,10:], u[1,10:], \
            v2[1,10:], \
            linewidths=0.5,\
            colors='k')
    cntr11 = ax[1,1].tricontourf(u[0,10:],\
            u[1,10:], v2[1,10:],\
            levels=linspace(min(v2[1,10:]),\
            max(v2[1,10:]), 30),\
            cmap="RdBu_r")
    ax[1,1].set_title('$V^2_2$',fontsize=24)

    fig.colorbar(cntr00, ax=ax[0,0])
    ax[0,0].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr01, ax=ax[0,1])
    ax[0,1].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr10, ax=ax[1,0])
    ax[1,0].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr11, ax=ax[1,1])
    ax[1,1].set(xlim=(0, 1), ylim=(0, 1))

    for i in range(2):
        for j in range(2):
            ax[i,j].xaxis.set_tick_params(labelsize=24)
            ax[i,j].yaxis.set_tick_params(labelsize=24)


    return fig,ax

def plot_clv_gradients():
    s = [0.7,0.3]
    #s = zeros(2)
    eps = 4.e-2
    d = 2
    u0 = random.rand(d,1)
    n = 20000
    u = step(u0,s,n) #shape: (n+1)xdx1
    u = u[1:].T[0] # shape:dxn
    du = dstep(u,s) #shape:nxdxd
    P = clvs(u,du,2).T #shape:nxdxd
    v1 = P[0]
    v2 = P[1]
    n_grid = 500
    x_x = linspace(0.,1.,n_grid)
    x_grid, y_grid = meshgrid(x_x, x_x)
    f = LinearNDInterpolator(u.T[10:],\
            v2[1,10:],)
    v2x = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    dx = x_x[1]
    #calculate x partial derivatives
    dv2x_dx = (hstack([v2x[:,1:],\
            v2x[:,0].reshape(-1,1)]) - \
            hstack([v2x[:,-1].reshape(-1,1),\
            v2x[:,0:-1]]))/(2*dx)
    f = LinearNDInterpolator(u.T[10:],\
            v1[1,10:],)
    v1y = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    dv1y_dx = (hstack([v1y[:,1:],\
            v1y[:,0].reshape(-1,1)]) - \
            hstack([v1y[:,-1].reshape(-1,1),\
            v1y[:,0:-1]]))/(2*dx)
    #calculate y partial derivatives
    dv2x_dy = (vstack([v2x[1:,:],\
            v2x[0,:].reshape(1,-1)]) - \
            vstack([v2x[-1,:].reshape(1,-1),\
            v2x[0:-1,:]]))/(2*dx)
    dv1y_dy = (vstack([v1y[1:,:],\
                    v1y[0,:].reshape(1,-1)]) - \
                    vstack([v1y[-1,:].reshape(1,-1),\
                    v1y[0:-1,:]]))/(2*dx)

    fig, ax = subplots(2,2)
    cntr00 = ax[0,0].contourf(x_grid,\
            y_grid, dv1y_dx, levels=linspace(nanmin(dv1y_dx),\
            nanmax(dv1y_dx)/1.5,30))
    cntr01 = ax[0,1].contourf(x_grid,\
            y_grid, dv2x_dx,levels=linspace(nanmin(dv2x_dx)/5,\
            nanmax(dv2x_dx)/8,30))
    cntr10 = ax[1,0].contourf(x_grid,\
            y_grid, dv1y_dy,levels=linspace(nanmin(dv1y_dy),nanmax(dv1y_dy),30))
    cntr11 = ax[1,1].contourf(x_grid,\
            y_grid, dv2x_dy,levels=linspace(nanmin(dv2x_dy)/5,nanmax(dv2x_dy)/5,30))


    fig.colorbar(cntr00, ax=ax[0,0])
    ax[0,0].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr01, ax=ax[0,1])
    ax[0,1].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr10, ax=ax[1,0])
    ax[1,0].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr11, ax=ax[1,1])
    ax[1,1].set(xlim=(0, 1), ylim=(0, 1))

    for i in range(2):
        for j in range(2):
            ax[i,j].xaxis.set_tick_params(labelsize=24)
            ax[i,j].yaxis.set_tick_params(labelsize=24)


    ax[0,0].set_title('$\partial_1 V^1_2$',fontsize=24)

    ax[0,1].set_title('$\partial_1 V^2_1$',fontsize=24)

    ax[1,0].set_title('$\partial_2 V^1_2$',fontsize=24)

    ax[1,1].set_title('$\partial_2 V^2_1$',fontsize=24)
    return fig, ax

def plot_clv_directional_derivatives():
    s = [0.7,0.3]
    #s = zeros(2)
    eps = 4.e-2
    d = 2
    u0 = random.rand(d,1)
    n = 30000
    u = step(u0,s,n) #shape: (n+1)xdx1
    u = u[1:].T[0] # shape:dxn
    du = dstep(u,s) #shape:nxdxd
    P = clvs(u,du,2).T #shape:nxdxd
    v1 = P[0]
    v2 = P[1]
    n_grid = 100
    x_x = linspace(0.,1.,n_grid)
    x_grid, y_grid = meshgrid(x_x, x_x)
    f = LinearNDInterpolator(u.T[10:],\
            v1[0,10:],)
    v1x = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v1[1,10:],)
    v1y = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v2[0,10:],)
    v2x = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v2[1,10:],)
    v2y = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)

    dx = x_x[1]
    #calculate x partial derivatives
    dv1y_dx = (hstack([v1x[:,1:],\
            v1x[:,0].reshape(-1,1)]) - \
            hstack([v1x[:,-1].reshape(-1,1),\
            v1x[:,0:-1]]))/(2*dx)
    dv2y_dx = (hstack([v2x[:,1:],\
            v2x[:,0].reshape(-1,1)]) - \
            hstack([v2x[:,-1].reshape(-1,1),\
            v2x[:,0:-1]]))/(2*dx)
    #calculate y partial derivatives
    dv1y_dy = (vstack([v1x[1:,:],\
            v1x[0,:].reshape(1,-1)]) - \
            vstack([v1x[-1,:].reshape(1,-1),\
            v1x[0:-1,:]]))/(2*dx)
    dv2y_dy = (vstack([v2x[1:,:],\
                    v2x[0,:].reshape(1,-1)]) - \
                    vstack([v2x[-1,:].reshape(1,-1),\
                    v2x[0:-1,:]]))/(2*dx)

    # calculate directional derivatives
    dv1y_dv1 = dv1y_dx*v1x + dv1y_dy*v1y
    dv1y_dv2 = dv1y_dx*v2x + dv1y_dy*v2y
    dv2y_dv1 = dv2y_dx*v1x + dv2y_dy*v1y
    dv2y_dv2 = dv2y_dx*v2x + dv2y_dy*v2y


    fig, ax = subplots(2,2)
    cntr00 = ax[0,0].contourf(x_grid,\
            y_grid, dv1y_dv1, levels=linspace(nanmin(dv1y_dv1),\
            nanmax(dv1y_dv1),50))
    cntr01 = ax[0,1].contourf(x_grid,\
            y_grid, dv1y_dv2,levels=linspace(nanmin(dv1y_dv2),\
            nanmax(dv1y_dv2),50))
    cntr10 = ax[1,0].contourf(x_grid,\
            y_grid, dv2y_dv1,levels=linspace(nanmin(dv2y_dv1),nanmax(dv2y_dv1),50))
    cntr11 = ax[1,1].contourf(x_grid,\
            y_grid, dv2y_dv2,levels=linspace(nanmin(dv2y_dv1),nanmax(dv2y_dv2),50))


    fig.colorbar(cntr00, ax=ax[0,0])
    ax[0,0].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr01, ax=ax[0,1])
    ax[0,1].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr10, ax=ax[1,0])
    ax[1,0].set(xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(cntr11, ax=ax[1,1])
    ax[1,1].set(xlim=(0, 1), ylim=(0, 1))

    for i in range(2):
        for j in range(2):
            ax[i,j].xaxis.set_tick_params(labelsize=24)
            ax[i,j].yaxis.set_tick_params(labelsize=24)


    ax[0,0].set_title('$D V^1_2\cdot V^1$',fontsize=24)

    ax[0,1].set_title('$D V^1_2\cdot V^2$',fontsize=24)

    ax[1,0].set_title('$D V^2_2\cdot V^1$',fontsize=24)

    ax[1,1].set_title('$D V^2_2 \cdot V^2$',fontsize=24)
    return fig, ax

def plot_clv_directional_derivatives():
    s = [0.7,0.3]
    #s = zeros(2)
    eps = 4.e-2
    d = 2
    u0 = random.rand(d,1)
    n = 30000
    u = step(u0,s,n) #shape: (n+1)xdx1
    u = u[1:].T[0] # shape:dxn
    du = dstep(u,s) #shape:nxdxd
    P = clvs(u,du,2).T #shape:nxdxd
    v1 = P[0]
    v2 = P[1]
    n_grid = 100
    x_x = linspace(0.,1.,n_grid)
    x_grid, y_grid = meshgrid(x_x, x_x)
    f = LinearNDInterpolator(u.T[10:],\
            v1[0,10:],)
    v1x = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v1[1,10:],)
    v1y = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v2[0,10:],)
    v2x = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)
    f = LinearNDInterpolator(u.T[10:],\
            v2[1,10:],)
    v2y = f(x_grid, y_grid).reshape(\
            n_grid,n_grid)

    dx = x_x[1]
    #calculate x partial derivatives
    dv1x_dx = (hstack([v1x[:,1:],\
            v1x[:,0].reshape(-1,1)]) - \
            hstack([v1x[:,-1].reshape(-1,1),\
            v1x[:,0:-1]]))/(2*dx)
    dv1y_dx = (hstack([v1y[:,1:],\
            v1y[:,0].reshape(-1,1)]) - \
            hstack([v1y[:,-1].reshape(-1,1),\
            v1y[:,0:-1]]))/(2*dx)
    dv2x_dx = (hstack([v2x[:,1:],\
            v2x[:,0].reshape(-1,1)]) - \
            hstack([v2x[:,-1].reshape(-1,1),\
            v2x[:,0:-1]]))/(2*dx)
    dv2y_dx = (hstack([v2y[:,1:],\
            v2y[:,0].reshape(-1,1)]) - \
            hstack([v2y[:,-1].reshape(-1,1),\
            v2y[:,0:-1]]))/(2*dx)

    #calculate y partial derivatives
    dv1x_dy = (vstack([v1x[1:,:],\
            v1x[0,:].reshape(1,-1)]) - \
            vstack([v1x[-1,:].reshape(1,-1),\
            v1x[0:-1,:]]))/(2*dx)
    dv2x_dy = (vstack([v2x[1:,:],\
            v2x[0,:].reshape(1,-1)]) - \
            vstack([v2x[-1,:].reshape(1,-1),\
            v2x[0:-1,:]]))/(2*dx)
    dv1y_dy = (vstack([v1y[1:,:],\
            v1y[0,:].reshape(1,-1)]) - \
            vstack([v1y[-1,:].reshape(1,-1),\
            v1y[0:-1,:]]))/(2*dx)
    dv2y_dy = (vstack([v2y[1:,:],\
                    v2y[0,:].reshape(1,-1)]) - \
                    vstack([v2y[-1,:].reshape(1,-1),\
                    v2y[0:-1,:]]))/(2*dx)

    # calculate directional derivatives
    dv1x_dv1 = dv1x_dx*v1x + dv1x_dy*v1y 
    dv1y_dv1 = dv1y_dx*v1x + dv1y_dy*v1y
    dv2x_dv2 = dv2x_dx*v2x + dv2x_dy*v2y 
    dv2y_dv2 = dv2y_dx*v2x + dv2y_dy*v2y

    #normalize directional derivatives (only getting 
    # direction of the directional derivatives)
    norm_dv1_dv1 = sqrt(dv1x_dv1*dv1x_dv1 + dv1y_dv1*dv1y_dv1)
    norm_dv2_dv2 = sqrt(dv2x_dv2*dv2x_dv2 + dv2y_dv2*dv2y_dv2)
    dv1x_dv1 /= norm_dv1_dv1
    dv1y_dv1 /= norm_dv1_dv1
    dv2x_dv2 /= norm_dv2_dv2
    dv2y_dv2 /= norm_dv2_dv2

    #flatten the grid
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    dv1x_dv1 = dv1x_dv1.flatten()
    dv1y_dv1 = dv1y_dv1.flatten()
    dv2x_dv2 = dv2x_dv2.flatten()
    dv2y_dv2 = dv2y_dv2.flatten()

    # plot
    eps = 2.e-2
    fig, ax = subplots(1,2)
    ax[0].plot([x_grid - eps*dv1x_dv1, x_grid + eps*dv1x_dv1],\
            [y_grid - eps*dv1y_dv1, y_grid + eps*dv1y_dv1],\
            lw=2.0, color='red')
    ax[1].plot([x_grid - eps*dv2x_dv2, x_grid + eps*dv2x_dv2],\
            [y_grid - eps*dv2y_dv2, y_grid + eps*dv2y_dv2],\
            lw=2.0, color='black')
    
    
    ax[0].set_title('$D V^1\cdot V^1$',fontsize=24)

    ax[1].set_title('$D V^1\cdot V^2$',fontsize=24)

    return fig, ax

