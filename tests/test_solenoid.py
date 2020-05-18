import sys
sys.path.insert(0,'../examples/')
from solenoid import *
from numpy import *
from scipy.interpolate import *
from mpl_toolkits.mplot3d import Axes3D
def plot_solenoid():
#if __name__ == "__main__":
    m = 1
    n = 2000
    u0 = rand(3,m)
    u_trj = step(u0, [0.], n)
    fig = figure(figsize=(36,36))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(u_trj[:,0,0], u_trj[:,1,0], u_trj[:,2,0], \
            "o",ms=2.0)


