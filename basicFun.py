import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pcl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
from itertools import product
from scipy.signal import butter, freqz
import zipfile
import os

def timeshift1d(s,t0):
    r'''time shift (t0) for 1-d signal'''
    x = np.array(s)
    X = np.fft.fft(x)
    w = np.fft.fftfreq(len(x))*2*np.pi
    F = np.exp(1j*w*t0)
    xs = np.fft.ifft(X*F).real
    
    return xs

def deri1d(s,order=1):
    r'''calculate nth-order derivative
        order-derivative order (scalar, int)'''
    x = np.array(s)
    X = np.fft.fft(x)
    w = np.fft.fftfreq(len(x))*2*np.pi
    for i in range(order):
        X *= 1j*w
    x = np.fft.ifft(X).real
    return x

def cubeidx(idx):
    r'''find a cube index with given idx as the cube left-top corner'''
    idx_cube = np.tile(idx,(8,1))
    c = 0
    for i,j,k in product([0,1],[0,1],[0,1]):
        idx_cube[c,:] += np.array([i,j,k])
        c += 1
    
    idx_cube = tuple(idx_cube.transpose())
    
    return idx_cube

def expand(md, N):
    r'''expand array with N+1 'edge' condition'''
    mdp = np.pad(md, N+1, 'edge')
    return mdp   
    
#readin basic modeling parameter
def readparm(path = './resources',disp=True):
    r'''readin basic modelling parameters, including:
        dt,nt,f0,fs,Na,M; they are explained in the parm.txt file.
            path-path of parm.txt where the parameters are written'''
    
    with open('/'.join((path,'parm.txt'))) as f:
        lines = f.readlines()
    par = []
    for idx, line in enumerate(lines):
        if idx % 2:
            par.append(float(line))
        if disp:
            print(line)
    dt = par[0]
    nt = int(par[1])
    f0 = par[2]
    ln = int(par[3])
    lm = int(par[4])
    return dt, nt, f0, ln, lm

def show3D(md, xyz, xyzi=(0,0,0), ea=(30,-45), clip=1, rcstride=(10,10), clim=None):
    r'''plot 3D cube image:
        md-3-D data volume (3darray, float, (n1,n2,n3))
        xyz-3-D axes coordinates (list, 1darray, (3,))
        xyzi-position of three slicing image indices (tuple, int, (3,))
        ea-viewing angle (tuple, float, (2,))
        clip-image clipping (scalar, float, <1)
        rcstride-2-D plotting stride (tuple, int, (2,))
        clim-colorbar range (None or tuple, int, (2,)): if it is not None, clip is overwritten'''

    # slice zero index image along each dimension
    mx = md[xyzi[0],:,:].transpose()
    my = md[:,xyzi[1],:].transpose()
    mz = md[:,:,xyzi[2]].transpose()
    MIN = min([np.amin(mx),np.amin(my),np.amin(mz)])
    MAX = max([np.amax(mx),np.amax(my),np.amax(mz)])
    if clim is None:
        cN = pcl.Normalize(vmin=MIN*clip, vmax=MAX*clip)
        rg = [MIN*clip,(MAX-MIN)*clip]
    else:
        cN = pcl.Normalize(vmin=clim[0], vmax=clim[1])
        rg = [clim[0],clim[1]-clim[0]]
    # plot the model
    fig = plt.figure(figsize = (8,5))
    ax = fig.gca(projection='3d')
    
    # plot the indicator line
    xi = xyz[0][xyzi[0]]
    yi = xyz[1][xyzi[1]]
    zi = xyz[2][xyzi[2]]
    ax.plot([xi,xi],[xyz[1][0],xyz[1][0]],[xyz[2][0],xyz[2][-1]],'r-',linewidth=2,zorder=10)
    ax.plot([xi,xi],[xyz[1][0],xyz[1][-1]],[xyz[2][0],xyz[2][0]],'r-',linewidth=2,zorder=10)
    ax.plot([xyz[0][0],xyz[0][0]],[yi,yi],[xyz[2][0],xyz[2][-1]],'r-',linewidth=2,zorder=10)
    ax.plot([xyz[0][0],xyz[0][-1]],[yi,yi],[xyz[2][0],xyz[2][0]],'r-',linewidth=2,zorder=10)
    ax.plot([xyz[0][0],xyz[0][-1]],[xyz[2][0],xyz[2][0]],[zi,zi],'r-',linewidth=2,zorder=10)
    ax.plot([xyz[0][0],xyz[0][0]],[xyz[1][0],xyz[1][-1]],[zi,zi],'r-',linewidth=2,zorder=10)
    
    # plot the three surfaces
    ax = slice_show(ax, mz, xyz, 0, rg=rg, rcstride=rcstride)
    ax = slice_show(ax, mx, xyz, 1, rg=rg, rcstride=rcstride)
    ax = slice_show(ax, my, xyz, 2, rg=rg, rcstride=rcstride)
    
    # set the axes
    ax.set_xticks(np.linspace(xyz[0][0],xyz[0][-1],5))
    ax.set_yticks(np.linspace(xyz[1][0],xyz[1][-1],5))
    ax.set_zticks(np.linspace(xyz[2][0],xyz[2][-1],5))
    ax.invert_zaxis()
    ax.invert_xaxis()
    ax.set_xlabel('x (m)',fontsize=12)
    ax.set_ylabel('y (m)',fontsize=12)
    ax.set_zlabel('z (m)',fontsize=12)
    ax.view_init(elev=ea[0],azim=ea[1])
    fig.colorbar(cm.ScalarMappable(norm=cN, cmap='gray'))
    plt.show()

    return fig, ax

def slice_show(ax, ms, xyz, od, rg=None, offset=0, rcstride=(10,10)):
    r'''show specific slice of model'''
    
    if rg is None:
        shift = np.amin(ms)
        normalizer = np.amax(ms)-shift
    else:
        shift = rg[0]
        normalizer = rg[1]
    if normalizer == 0:
        msN = np.zeros_like(ms)+0.5
    else:
        msN = (ms-shift)/normalizer
    colors = plt.cm.gray(msN)
    if od == 0:
        [X,Y] = np.meshgrid(xyz[0],xyz[1])
        Z = np.zeros_like(X)+xyz[2][0]+offset
    if od == 1:
        [Y,Z] = np.meshgrid(xyz[1],xyz[2])
        X = np.zeros_like(Y)+xyz[0][0]+offset
    if od == 2:
        [X,Z] = np.meshgrid(xyz[0],xyz[2])
        Y = np.zeros_like(X)+xyz[1][0]+offset
    surf = ax.plot_surface(X, Y, Z, 
                           facecolors=colors, rstride=rcstride[0], cstride=rcstride[1], zorder=1)
    
    return ax

def un_zip(file_name,dd):
    """unzip the zip file "file_name" into "dd" directory"""
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(dd):
        pass
    else:
        os.mkdir(dd)
    for names in zip_file.namelist():
        zip_file.extract(names,dd)
    zip_file.close()
    
    
    
    
    
    
    
    
    
    
    
    
    