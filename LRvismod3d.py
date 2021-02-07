from math import pi, isnan
import numpy as np
import cupy as cp
import cupy.fft as fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
from itertools import product, combinations
import scipy.linalg as la
import progressbar
from basicFun import expand, show3D, cubeidx, deri1d

import time
import GPUtil

from gvopt import *

# create source wavelet class
class source:
    r'''define a Ricker wavelet or readin a source wavelet function'''
    def __init__(self, fs, dt, t0, sl, order=0, tol=0.01, path = './resources', fn = None, disp=True):
        self.disp = disp # display boolean (boolean scalar)
        self.order = order # derivative order for Ricker (int scalar)
        self.fs = fs # source dominant frequency (foat scalar, Hz)
        self.dt = dt # source function interval (float scalar, s)
        self.t0 = t0 # Ricker shift (float scalar, s) or None for readin general source wavelet
        self.sl = sl # length of the Ricker source (int scalar)
        self.path = path # path of the source function file (str)
        self.fn = fn # source function file name, if not given, generate the Ricker wavelet and save to this file (str)
        self.ss = self.s() # wavelet time series (1darray, float, (sl,))
        
        self.tol = tol # amplitude threshold when selecting valid frequency samples for the source wavelet (float scalar, <1)
        self.ws, fd = self.wscal() # frequency samples with "big enough" (according to the self.tol) amplitude (1darray, float, (nw,))
        if fn is not None:
            self.fs = fd
        
    def s(self):
        r'''generate source wavelet time seris, or readin source wavelet according to path and fn'''
        if self.fn is None:
            t = np.arange(0,self.sl*self.dt,self.dt,dtype=np.float32)+self.t0
            y = (1-2*(pi*self.fs*t)**2)*np.exp(-(pi*self.fs*t)**2)
            if self.order!=0:
                y = deri1d(y,self.order)
        else:
            y = np.fromfile('/'.join((path,fn)))
            self.t0 = None
            self.sl = len(y)

        # display time domain signal
        if self.disp:
            fig,ax = plt.subplots(1,1)
            ax.plot(t-self.t0,y)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
        
        return y
    
    def wscal(self):
        r'''calculate the frequency samples and the dominant frequency of the source amplitude spectrum'''
        f = np.fft.fftfreq(self.sl, self.dt)
        Sa = np.abs(np.fft.fft(self.ss))
        San = Sa[:(self.sl+1)//2]
        Sam = np.amax(San)
        Sab = Sam*self.tol
        idx_ab = np.where(San>=Sab)
        idx_m = np.where(San==Sam)
        ws = f[idx_ab]*2*pi
        fd = f[idx_m]
    
        # display
        if self.disp:
            fig,ax = plt.subplots(1,1)
            ax.plot(f[:(self.sl+1)//2],San)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
        
        return ws, fd
        
#create 3-D velocity or Q model class
class model:
    r'''3D velocity or Q model class, including the model array as well as its
        grid information. It also involves model display method.
        The models information files include:
            1. head file
            2. bindary file'''
    def __init__(self, head, path = './resources', f0=None):
        self.path = path # head and biniray files path (str)
        inp = self._readhead(head)
        self.fn = inp[0] # bininary file name (str)
        self.n = inp[1] # model dimension (list, int, (3,))
        self.o = inp[2] # model origin (list, float, (3,))
        self.d = inp[3] # model cell size (list, float, (3,))
        self.x, self.y, self.z = self.axcoordinate() # model grid coordinates (1darray, float, (n[012],))
        self.md = self.mdarray() # model value array (3darray, float, n)
        if f0 is not None:
            self.w0 = f0*2*np.pi # reference angular frequency for velocity model (float scalar)
        else:
            self.w0 = None # no reference angular frequency for other models
        
    def axcoordinate(self):
        r'''create model grid coordinates'''
        x,y,z = deque(np.arange(self.o[i],self.o[i]+self.d[i]*self.n[i],self.d[i]) for i in range(3))
        return x, y, z
    
    def _readhead(self, head):
        r'''readin parameters in head files, which includes：
                in-model binary file path (str)
                n1,n2,n3-dimension of the model (int scalar)
                o1,o2,o3-origin of the model (float scalar)
                d1,d2,d3-spatial intervals (float scalar)
                ***esize and format is fixed as:
                    4 (float32) and big endian 
                    (i.e., dtype = '>f4')'''    
        fh = '/'.join((self.path,head))
        with open(fh) as f:
            lines = f.readlines()
        n = []
        o = []
        d = []
        for idx,line in enumerate(lines):
            if idx == 0:
                fl = line.split('"')
                fn = fl[1]
            else:
                if idx == 4:
                    fl = line.split('=')
                    esize = int(fl[1])
                else:
                    fl = line.split(' ')
                    for i in range(3):
                        x = fl[i].split('=')
                        if idx == 1:
                            n.append(int(x[1]))
                        if idx == 2:
                            o.append(float(x[1]))
                        if idx == 3:
                            d.append(float(x[1]))
        if esize != 4:
            raise ImportError('The head folder indicates wrong esize!')
        fn = '/'.join((self.path,fn))
        return fn, n, o, d
    
    def mdarray(self):
        r'''readin 3-D model value array:
            the dimension order of md is (x,y,z)'''
        md = np.fromfile(self.fn, dtype = '>f4')
        md = np.array(np.reshape(md,tuple(self.n),'F'), dtype = np.float32)
        return md 

# create slicing indices and params for hybrid absorbing boundary condition
class hABC:
    def __init__(self, n, N, M=0, linw=False):
        r'''define hybrid absorbing boundary nodes related to
            hybrid OWWE absorbing boundary conditions
            n-model deminsion (model.n)
            N-number of non-zero absorbing layers (int scalar)
            M-pure owwe absorbing layers among the N layers (int scalar)
            linw-wether to use linear weight (boolean)'''
        self.N = N
        self.M = M
        self.n = n
        self.linw = linw
        self.ax = np.array([1,2,3],dtype=np.int16)
        self.w2 = self._w2cal()
        # generate outer directions for plane, edge and corner boundaries
        pod = [1,2,3,-1,-2,-3]
        eod = np.tile(np.array(deque(product([1,-1],[1,-1])),dtype=np.int16),(3,1))*\
              np.tile(np.array(deque(combinations([1,2,3],2)),dtype=np.int16),(4,1))
        cod = np.tile([1,2,3],(8,1))*\
              np.array(deque(product([1,-1],[1,-1],[1,-1])),dtype=np.int16)
        
        # generate all 6 plane-boundary nodes and cat them together
        for i in range(len(pod)):
            print(f'Plane No: {i+1}')
            w, Bi, Ii, Bin, Iin = self.planeidx(pod[i])
            if i == 0:
                wp, Bp, Ip, Bnp, Inp = w, Bi, Ii, Bin, Iin
            else:
                wp = np.concatenate((wp,w),axis=0)
                Bp = np.concatenate((Bp,Bi),axis=0)
                Ip = np.concatenate((Ip,Ii),axis=0)
                Bnp = np.concatenate((Bnp,Bin),axis=1)
                Inp = np.concatenate((Inp,Iin),axis=1)
        self.wp, self.Bp, self.Ip, self.Bnp, self.Inp = \
        wp, tuple(Bp.T), tuple(Ip.T),\
        [tuple(Bnp[i].T) for i in range(4)],\
        [tuple(Inp[i].T) for i in range(4)]
        
        # generate all 12 edge-boundary nodes and cat them together
        for i in range(len(eod)):
            print(f'Edge No: {i+1}')
            w, Bi, Bin = self.edgeidx(eod[i])
            if i == 0:
                we, Be, Bne = w, Bi, Bin
            else:
                we = np.concatenate((we,w),axis=0)
                Be = np.concatenate((Be,Bi),axis=0)
                Bne = np.concatenate((Bne,Bin),axis=1)
        self.we, self.Be = we, tuple(Be.T)
        self.Bne = [tuple(Bne[i].T) for i in range(2)]
        
        # generate all 8 corner-boundary nodes and cat them together
        for i in range(len(cod)):
            print(f'Corner No: {i+1}')
            w, Bi, Bin, Binn, Bind = self.corneridx(cod[i])
            if i == 0:
                wc, Bc, Bnc, Bnnc, Bndc = \
                w, Bi, Bin, Binn, Bind
            else:
                wc = np.concatenate((wc,w),axis=0)
                Bc = np.concatenate((Bc,Bi),axis=0)
                Bnc = np.concatenate((Bnc,Bin),axis=1)
                Bnnc = np.concatenate((Bnnc,Binn),axis=1)
                Bndc = np.concatenate((Bndc,Bind),axis=1)
        self.wc, self.Bc = wc, tuple(Bc.T)
        self.Bnc, self.Bnnc, self.Bndc = [tuple(Bnc[i].T) for i in range(3)]\
                                        ,[tuple(Bnnc[i].T) for i in range(3)]\
                                        ,[tuple(Bndc[i].T) for i in range(3)]
    
    def _w2cal(self):
        r'''calculate the transferring weight for TWWE:
            output:
                w2-TWWE weight vector (1D float array, (N,))'''
        w1 = np.zeros(self.N,dtype=np.float32)
        if self.linw:
            a = 1
        else:
            a = 1.5+0.07*(self.N-self.M)
        for i in range(self.N):
            if i <= self.M:
                w1[i] = 1
            elif i < self.N:
                w1[i] = ((self.N-i)/(self.N-self.M))**a
        w2 = 1-np.flip(w1)
        
        return w2
        
    def planeidx(self, od):
        r'''generate plane Boundary nodes indices according to outer direction:
            od-standing x, y or z axes representing by 1, 2, 3 
                (sign of od representing directions)'''
        
        # identify tangent directions td1, td2
        s = np.sign(od)
        mask = ~(self.ax == abs(od))
        td1, td2 = self.ax[mask]
        # set the outter direction starting index
        sidx = self.N-1
        if od > 0:
            sidx += self.n[od-1]+1
        # find the basic number of nodes along tangent directions
        n1 = self.n[td1-1]
        n2 = self.n[td2-1]
        Ni = [(n1+2*i)*(n2+2*i) for i in range(self.N)]
        Ni.insert(0,0)
        Nic = np.cumsum(Ni)
        N = Nic[-1]
        # loop through all Bi and save idx_od, idx_td1, idx_td2 and w
        Bi = np.zeros((N,3),dtype=np.int16)
        w = np.zeros(N,dtype=np.float32)
        for i in range(self.N):
            I = sidx+s*i
            Bi[Nic[i]:Nic[i+1],0] = I
            Bi[Nic[i]:Nic[i+1],1:] = np.array(deque(product(range(n1+2*i),range(n2+2*i))),dtype=np.int16)
            Bi[Nic[i]:Nic[i+1],1:] += self.N-i
            w[Nic[i]:Nic[i+1]] = self.w2[i]
        # generate Ii
        Ii = np.array(Bi)
        Ii[:,0] -= s
        # generate Bin and Iin
        Bin = np.tile(Bi,(4,1,1))
        Iin = np.tile(Ii,(4,1,1))
        Bin[0][:,1] += 1
        Bin[1][:,1] -= 1
        Bin[2][:,2] += 1
        Bin[3][:,2] -= 1
        Iin[0][:,1] += 1
        Iin[1][:,1] -= 1
        Iin[2][:,2] += 1
        Iin[3][:,2] -= 1

        # set the permutation order
        odr = [abs(od),td1,td2]
        pm = np.zeros(3,dtype=np.int16)
        for i in range(3):
            pm[odr[i]-1] = i
        
        # permutated idx
        pBi = Bi[:,pm]
        pIi = Ii[:,pm]
        pBin = Bin[:,:,pm]
        pIin = Iin[:,:,pm]
        
        return w, pBi, pIi, pBin, pIin
        
        
    def edgeidx(self, od):
        r'''generate edge Boundary nodes indices according to outer direction:
            od-outer direction tuple, e.g., (-1,2) indcates (-x,y) direction'''
        
        # identify tangent direction
        od1 = abs(od[0])
        s1 = np.sign(od[0])
        od2 = abs(od[1])
        s2 = np.sign(od[1])
        mask = (self.ax!=od1) * (self.ax!=od2)
        td = self.ax[mask][0]
        # set the outter direction starting index
        sidx1 = self.N-1
        sidx2 = self.N-1
        if s1 > 0:
            sidx1 += self.n[od1-1]+1
        if s2 > 0:
            sidx2 += self.n[od2-1]+1
        # find the basic number of nodes along tangent direction
        n = self.n[td-1]
        Ni = [n+2*i for i in range(self.N)]
        Ni.insert(0,0)
        Nic = np.cumsum(Ni)
        N = Nic[-1]
        # loop through all Bi
        Bi = np.zeros((N,3),dtype=np.int16)
        w = np.zeros(N,dtype=np.float32)
        for i in range(self.N):
            I1 = sidx1+s1*i
            I2 = sidx2+s2*i
            Bi[Nic[i]:Nic[i+1],:] = np.array([[I1,I2,j+self.N-i] for j in range(n+2*i)],dtype=np.int16)
            w[Nic[i]:Nic[i+1]] = self.w2[i]

        # generate Bin
        Bin = np.tile(Bi,(2,1,1))
        Bin[0][:,0] -= s1
        Bin[1][:,1] -= s2

        # set the permutation order
        odr = [od1,od2,td]
        pm = np.zeros(3,dtype=np.int16)
        for i in range(3):
            pm[odr[i]-1] = i
            
        # permutated idx
        pBi = Bi[:,pm]
        pBin = Bin[:,:,pm]
        
        return w, pBi, pBin
        
    def corneridx(self, od):
        r'''generate corner Boundary nodes indices according to outer direction:
            od-outer direction tuple, e.g., (-1,2,-3) indcates (-x,y,-z) direction'''
        
        # identify outter direction and its sign
        s = np.sign(od)
        # set the outter direction starting index
        sidx = [self.N-1,self.N-1,self.N-1]
        for i in range(3):
            if s[i] > 0:
                sidx[i] += self.n[i]+1
        # loop through all Bi and w
        Bi = np.array([[sidx[j]+s[j]*i for j in range(3)] for i in range(self.N)],dtype=np.int16)
        w = self.w2
        
        # Generate Bin:(Bx, By, Bz), Binn:(Bxx, Byy, Bzz), Bind:(Bxy, Byz, Bzx)
        Bin = np.tile(Bi,(3,1,1))
        for i,j in product(range(3),range(self.N)):
            Bin[i][j][i] -= s[i]
        Binn = np.copy(Bin)
        for i,j in product(range(3),range(self.N)):
            Binn[i][j][i] -= s[i]
        Bind = np.copy(Bin)
        for j in range(self.N):
            Bind[0][j][1] -= s[1]
            Bind[1][j][2] -= s[2]
            Bind[2][j][0] -= s[0]
        
        return w, Bi, Bin, Binn, Bind

# expand the original v and Q model properly for hybrid absorbing boundary condition
class expand_model_hABC:
    r'''expand the velocity and Q model according to given absorbing parameters:
        the obtained class is a new model class includes both expanded velocity and gamma model information'''
    def __init__(self, vm, Qm, Na):
        r'''vm-velocity model (model class)
            Qm-Q model (model class)
            Na-total absorbing layer number (int scalar)'''
        
        # test the consistency between vm and Qm
        if not(vm.d == Qm.d) or not(vm.n == Qm.n) or not(vm.o == Qm.o):
            raise ImportError('vm and Qm are not in consistent space!')
        self.d = vm.d
        self.Nmax = Na
        self.ne = np.array(vm.n)
        self.oe = np.array(vm.o)
        for i in range(3):
            self.ne[i] += 2*(self.Nmax) # expanded model dimension (list, int, (3,))
            self.oe[i] -= self.Nmax*self.d[i] # expanded model origin (list, float, (3,))
        # expanded model grid coordinates (1darray, float, (ne[012],))
        self.xe, self.ye, self.ze = deque(np.arange(self.oe[i],self.oe[i]+self.d[i]*self.ne[i],self.d[i]) for i in range(3))
        self.vme = np.array(expand(vm.md, self.Nmax)) # velocity array (3darray, float, ne)
        self.gme = np.array(expand(np.arctan(1/Qm.md)/pi, self.Nmax)) # gamma array (3darray, float, ne)
        self.w0 = vm.w0

# create acquisation geometry information and its corresponding sorting methods
class acqgeo:
    def __init__(self, ids, idr, osxy=None):
        r'''define source and receiver indices, it requires
            ids and idr as inputs:
                ids-source index along x, y and z directions (2D int array, (ns,3))
                idr-receiver index along x, y and z directions (2D int array, (nr,3))
            the offset range for limit the receiver number for each shot (default is no limitation):
                osxy-(osx,osy) (osx and osy are scalar)'''
        
        self.ids = np.array(ids)
        self.idr_total = np.array(idr)
        self.ns = self.ids.shape[0]
        self.nr_total = self.idr_total.shape[0]
        # create idr for each shot
        if osxy is None:
            self.idr_shot = np.tile(self.idr_total,(self.ns,1,1))
            self.nr_shot = np.zeros(self.ns,dtype=np.int16)+self.nr_total
        else:
            self.idr_shot = []
            self.nr_shot = []
            osx, osy = osxy
            for i in range(self.ns):
                xs, ys, _ = self.ids[i]
                mask = (np.abs(self.idr_total[:,0]-xs)<=osx) * \
                        (np.abs(self.idr_total[:,1]-ys)<=osy)
                self.idr_shot.append(self.idr_total[mask])
                self.nr_shot.append(len(self.idr_shot[-1]))
    
    def geometry_show(self, x, y, z, sps='middle', ea=(-90,90)):
        r'''display the acquisation geometry'''
        fig = plt.figure(figsize = (10,5))
        ax = fig.gca(projection='3d')
        # plot all sources and receivers
        xyzs = np.array([[x[idx],y[idy],z[idz]] for idx, idy, idz in self.ids])
        ax.scatter(xyzs[:,0],xyzs[:,1],xyzs[:,2],c='r',marker='o',label='sources')
        xyzr = np.array([[x[idx],y[idy],z[idz]] for idx, idy, idz in self.idr_total])
        ax.scatter(xyzr[:,0],xyzr[:,1],xyzr[:,2],c='b',marker='v',label='receivers')
        # plot the sample source and its corrsponding receivers
        if sps == 'middle':
            sNo = self.ns//2
        else:
            sNo = sps
        ax.scatter(x[self.ids[sNo][0]],y[self.ids[sNo][1]],\
                   z[self.ids[sNo][2]],c='k',marker='o',label=f'source No. {sNo}')
        xyzrs = np.array([[x[idx],y[idy],z[idz]] for idx, idy, idz in self.idr_shot[sNo]])
        ax.scatter(xyzrs[:,0],xyzrs[:,1],xyzrs[:,2],c='g',marker='v',label=f'receivers of source No. {sNo}')
        # general setting
        ax.legend()
        ax.set_xticks(np.linspace(0,x[-1],5))
        ax.set_yticks(np.linspace(0,y[-1],5))
        ax.set_zticks(np.linspace(0,z[5],2))
        ax.invert_zaxis()
        ax.invert_xaxis()
        ax.set_xlabel('x (m)',fontsize=12)
        ax.set_ylabel('y (m)',fontsize=12)
        ax.set_zlabel('z (m)',fontsize=12)
        ax.view_init(elev=ea[0],azim=ea[1])
        plt.show()
        
        return ax
        
    def sort_CSG_line(self, inds, lidr, lm='x'):
        r'''Given shot index, find the shot No.;
            Given x or y index, find the corresponding receiver No.
                inds-source index tuple (idx,idy,idz)
                lidr-x or y index for the receiver line (int scalar or int tuple)
                lm-line mode:
                    'x'-lidr is x index
                    'y'-lidr is y index
                    'xy'-lidr is tuple with x and y index'''
        
        # find shot No.
        sn = np.arange(self.ns)
        mask = (self.ids[:,0]==inds[0])*(self.ids[:,1]==inds[1])\
                *(self.ids[:,2]==inds[2])
        sNo = sn[mask][0]
        # find the receiver No.
        rn = np.arange(self.nr_shot[sNo])
        idr = self.idr_shot[sNo]
        if lm == 'x':
            mask = (idr[:,0]==lidr)
        if lm == 'y':
            mask = (idr[:,1]==lidr)
        if lm == 'xy':
            mask = (idr[:,0]==lidr[0])*(idr[:,1]==lidr[1])
        rNo = rn[mask]
        
        return sNo, rNo
    
    def sort_COG_line(self, lids, os, lm='x'):
        r'''Given source x or y index, find the shots' No.;
            Given offset (unit: grid size), find the corresponding receiver
            No. for each shot;
            lids-source x or y index (int scalar)
            os-offset along x or y direction in terms of grid size (int scalar)
            lm-line mode:
                'x'-lids is x index
                'y'-lids is y index'''
        
        # find shots' No.
        sn = range(self.ns)
        if lm == 'x':
            mask = (self.ids[:,0]==lids)
        if lm == 'y':
            mask = (self.ids[:,1]==lids)
        sNo = sn[mask]
        # find the receiver No. for each shot
        rNo = []
        for j in sNo:
            rn = range(self.nr_shot[j])
            idr = self.idr_shot[j]
            ids = self.ids[j]
            if lm == 'x':
                mask = (ids[0]-idr[:,0]==os)
            if lm == 'y':
                mask = (ids[1]-idr[:,1]==os)
            rNo.append(rn[mask])
        
        return sNo, rNo
    
    def sort_CSG_sperec(self, inds, indr):
        r'''find source and receiver indices for given inds and indr:
            inds-source index, (tuple, int, (3,))
            indr-receiver index, (tuple, int or None, (3,))
                free dimension index is given by None, e.g., indr=(None, 10, 10)'''
        # find shot No
        sn = np.arange(self.ns)
        mask = (self.ids[:,0]==inds[0])*(self.ids[:,1]==inds[1])\
                *(self.ids[:,2]==inds[2])
        sNo = sn[mask][0]
        # find the receiver No.
        rn = np.arange(self.nr_shot[sNo])
        idr = self.idr_shot[sNo]
        mask = np.zeros((3,self.nr_shot[sNo]),dtype=bool)
        for i in range(3):
            if indr[i] is None:
                mask[i] = True
            else:
                mask[i] = (idr[:,i]==indr[i])
        mask = mask[0]*mask[1]*mask[2]
        rNo = rn[mask]
        
        return sNo, rNo
            

# create Low-rank approximation for mixed-domain operators
class LRdecomp:
    r'''perform low-rank decomposition for viscoacoustic modeling updating matrix'''
    
    def __init__(self, typ_tsc, n, m, dt, vgme, dvg=(1,1e-5), mp=cp.get_default_memory_pool()):
        r'''typ_tsc-temporal compensation type (0-no compensator; 1-compensators A; 2-compensators B)
            typ_abc-model type (0-habc; 1-naabc)
            n,m-ranks for decomposing along wavenumber and space direction (int scalar)
            dt-modeling time interval (float scalar)
            vgme-expanded velocity and gamma model class (expand_vgme_habc or expand_vgme_naabc class)
            dvg-rounding off errors for finding unique v ang gamma (tuple, float, (2,))
            mp-memory pool for gpu'''
        
        if typ_tsc not in (0,1,2):
            raise ImportError(f'typ_tsc={typ_stc}, which is not among (0,1,2)!')
        self.typ_tsc = typ_tsc
        self.dt = dt
        self.n = n
        self.m = m
        self.dvg = dvg
        
        self.ms = vgme.ne # 3-D model dimension
        self.d = vgme.d # grid cell size
        self.w0 = vgme.w0 # velocity model reference angular frequency

        self.mp = mp
        
        # calculate k
        k = deque(cp.fft.fftfreq(self.ms[i],self.d[i]) for i in range(3))
        [Kx, Ky, Kz] = cp.meshgrid(k[0],k[1],k[2],indexing='ij')
        self.K = cp.sqrt(Kx**2+Ky**2+Kz**2).flatten()*2*pi
        
        del k[2], k[1], k[0], Kx, Ky, Kz
        self.mp.free_all_blocks()

        # define the spacial parms
        v = cp.expand_dims(cp.array(vgme.vme).flatten(),axis=1)
        g = cp.expand_dims(cp.array(vgme.gme).flatten(),axis=1)
        self.Px = cp.concatenate((v,g),axis=1)
        # define unique K and Px according to round off accuracy for Px
        Pxrd = cp.concatenate((cp.around(v/dvg[0])*dvg[0],cp.around(g/dvg[1])*dvg[1]),axis=1)
        Pxrdc = Pxrd[:,0].get()+1j*Pxrd[:,1].get()
        Pxu = np.unique(Pxrdc)
        self.Pxu = cp.stack((cp.array(Pxu.real),cp.array(Pxu.imag)),axis=0).transpose()
        self.Ku = cp.array(np.sort(np.unique(self.K.get())))

        # expand Pxu if its length smaller than 4*m
        Nxu = len(self.Pxu[:,0])
        if Nxu < 4*m:
            self.Pxu = cp.pad(self.Pxu,((0,4*m-Nxu),(0,0)),'edge')
        # reorder Pxu according to ascending velocity 
        idx = np.argsort(self.Pxu[:,0].get())
        self.Pxu = self.Pxu[idx,:]
        
        del v,g,Pxrd
        self.mp.free_all_blocks()
    
    def free_gpublocks(self):
        r'''free all GPU blocks after finishing decomposition'''
        del self.K, self.Px, self.Pxu, self.Ku
        self.mp.free_all_blocks()
     
    def _Vdts(self,k,px):
        r'''dispersion-related compensated mixed-domain operator calculation'''
        v = cp.expand_dims(px[:,0],axis=0)
        g = cp.expand_dims(px[:,1],axis=0)
        K = cp.expand_dims(k,axis=1)
        if self.typ_tsc == 0:
            x = K**(2+2*g)
        else:
            if self.typ_tsc == 1:
                k0g = (self.w0/v)**(-g)
                Kg = K**(1+g)
                gc = cp.cos(pi*g)
                gh = cp.cos(pi*g/2)
                D = 0.5*v*self.dt*cp.sqrt(gc)*gh*k0g
                x = (cp.sin(D*Kg)/D)**2
                # delete GPU variables
                del k0g, Kg, gc, gh, D
            else:
                Vt = 0.5*v*self.dt
                x = (cp.sin(K*Vt)/Vt)**(2+2*g)
                # delete GPU variables
                del Vt
        y = x.get()
        
        # delete GPU variables
        del x, v, g, K
        # free all blocks
        self.mp.free_all_blocks()

        return y
    
    def _Vlts(self,k,px):
        r'''loss-related compensated mixed-domain operator calculation'''
        v = cp.expand_dims(px[:,0],axis=0)
        g = cp.expand_dims(px[:,1],axis=0)
        K = cp.expand_dims(k,axis=1)
        if self.typ_tsc == 0:
            x = K**(1+2*g)
        else:
            if self.typ_tsc == 1:
                gh = cp.cos(pi*g/2)
                Kg = K**(1+2*g)
                L = 0.5*v*self.dt*gh*K
                x = (cp.sinc(L/pi))**2*Kg
                # delete GPU variables
                del Kg, gh, L
            else:
                Vt = 0.5*v*self.dt
                x = (cp.sin(K*Vt)/Vt)**(1+2*g)
                # delete GPU variables
                del Vt
        y = x.get()
        
        # delete GPU variables
        del x, v, g, K
        # free all blocks
        self.mp.free_all_blocks()
        
        return y
        
    def lrW(self,tp=1):
        r'''tp: 1--dispersion term;
                2--loss term'''
        if tp not in (1,2):
            raise ImportError(f'tp={tp}, must be 1 or 2.')
        ksidx = np.linspace(0,self.Ku.size-1,4*self.n,dtype=np.int32)
        xsidx = np.linspace(0,self.Pxu.shape[0]-1,4*self.m,dtype=np.int32)
        if tp == 1:
            Wks_fun = self._Vdts
        else:
            Wks_fun = self._Vlts
        
        # compose W1
        print('Ws for W1')
        Ws = Wks_fun(self.Ku[ksidx], self.Pxu)

        print('RRQR for Ws')
        _, P1 = la.qr(Ws, overwrite_a=True, mode='r', pivoting=True)
        P1n = P1[:self.n]
        Pxun = self.Pxu[P1n,:]
        for i in range(self.n):
            Pxui = Pxun[i,:]
            mi = (cp.abs(Pxui[0]-self.Px[:,0])<=self.dvg[0])*(cp.abs(Pxui[1]-self.Px[:,1])<=self.dvg[1])
            Pxun[i] = self.Px[mi,:][0]
        print(f'Principal Px combinations:{Pxun}')
        W1 = Wks_fun(self.K, Pxun)
        del Pxui, mi
        
        # compose W2
        print('Ws for W2')
        Ws = Wks_fun(self.Ku, self.Pxu[xsidx])
        print('RRQR for Ws')
        _, P2 = la.qr(Ws.T, overwrite_a=True, mode='r', pivoting=True)
        P2m = P2[:self.m]
        Kum = self.Ku[P2m]
        print(f'Principal K components:{Kum}')
        W2 = Wks_fun(Kum, self.Px)
        # calculate A
        Ai = Wks_fun(Kum, Pxun)
        A = la.pinv(Ai)
        del Kum, Pxun
        
        # reshape W1 and pk
        rs = tuple(np.insert(self.ms,0,self.n))
        W1 = np.reshape(W1.T,rs)
        pk = np.reshape(A@W2,rs)
        
        # trancate the accuracy for efficiency
        W1 = np.array(W1,dtype=np.float32)
        pk = np.array(pk,dtype=np.float32)
        
        # free all blocks
        self.mp.free_all_blocks()
        
        return W1, pk

# LR-approximated vsicoacoustic modeling
class LRmodeling:
    def __init__(self
                 ,basic_parm
                 ,source
                 ,acqgeo
                 ,vmh
                 ,Qmh
                 ,typ_tsc=1
                 ,abc={'naABCs':None}
                 ,mp=cp.get_default_memory_pool()
                 ,inpath='./resources'
                 ,outpath='./outputs'):
        r'''intialize the modeling using following parms:
            basic_parm--basic parameters:
                dt, nt, f0, ln, lm
            source--source function information (source class)
            acqgeo--cquisation geometry class (acqgeo class)
            vmh--velocity model headfile (str)
            Qmh--Q model headfile (str)
            typ_tsc--temporal compensator type: 0-no compensator; 1-Compensators A; 2-Compensators B
            abc--absorbing boundary condition type and parameter (dict)
            mp--GPU memory pool (cupy.memoryPool class)
            inpath--input path (str)
            outpath--output path (str)'''
        
        self.mp = mp
        self.outpath = outpath
        
        # unpack basic parms
        self.dt, self.nt, self.f0, self.ln, self.lm = basic_parm
        self.w0 = self.f0*2*pi
        # check given source function time step size
        if not(source.dt == self.dt):
            raise ImportError('Given source sequence has different time inverval!')
        # source and geometry class
        self.source = source
        self.acgeo = acqgeo
        # determine abc type
        typ_abc, = abc
        if typ_abc not in ('naABCs','hABCs','hnaABCs'):
            raise ImportError(f'The import abc type is {typ_abc}, but it has to be naABCs or hABCs!')
        # determine abc parameters
        par_abc, = abc.values()
        # readin original model arries
        vm = model(vmh, path=inpath, f0=self.f0)
        Qm = model(Qmh, path=inpath)
        # initially assume no hard boundary
        ts0 = time.time() # test runtime
        if typ_abc == 'naABCs':
            if par_abc == None:
                self.vgme = expand_model_naABC(typ_tsc,vm,Qm,self.dt,source.ws)
                print('Cyclic boundary condition...')
            else:
                print('Model expanding for naABC...')
                self.vgme = expand_model_naABC(typ_tsc,vm,Qm,self.dt,source.ws,vareps=par_abc)
        elif typ_abc == 'hABCs':
            print('Model expanding for hABC...')
            self.vgme = expand_model_hABC(vm,Qm,par_abc)
            vmno = [i-2*(self.vgme.Nmax) for i in self.vgme.ne]
            self.ABC = hABC(vmno, self.vgme.Nmax)
        else: 
            print('Model expanding for hnaABC...')
            vareps,hr0,hN = par_abc # retrive the naABC tolerance and hABC reflectivity and number of hABC layers
            self.vgme = expand_model_hnaABC(typ_tsc,vm,Qm,self.dt,source.ws,hr0,hN,vareps=vareps)
            vmno = [i-2*hN for i in self.vgme.ne] # get the expanded model dimension for pure naABC
            self.ABC = hABC(vmno, hN, linw=True) # create the weight and boundary node indices for outtermost hABC
        self.typ_abc = typ_abc
        print(f'Absorbing layer No: {self.vgme.Nmax}')
        print(f'Runtime for preparing ABCs: {time.time()-ts0} s') # test runtime
        
        # model size after expanding for ABC
        self.ms = self.vgme.ne
            
        # LR decomposition
        print(f'Low-rank decomposition...')
        ts0 = time.time() # test runtime
        LRD = LRdecomp(typ_tsc, self.ln, self.lm, self.dt, self.vgme, mp=self.mp)
        self.Wd, self.pkd = LRD.lrW(tp=1)
        self.Wl, self.pkl = LRD.lrW(tp=2)
        LRD.free_gpublocks()
        print(f'Runtime for low-rank decomposition: {time.time()-ts0} s') # test runtime
        
        # auxilary spatial parameters
        print(f'Model parameter calculation...')
        v = cp.array(self.vgme.vme)
        g = cp.array(self.vgme.gme)   
        self.g = g.get()
        v2 = (v*cp.cos(0.5*pi*g))**2
        self.v2 = v2.get()
        yit = -(v/self.w0)**(2*g)*cp.cos(pi*g)
        tau = -v**(2*g-1)/self.w0**(2*g)*cp.sin(pi*g)
        self.yit = yit.get()
        self.tau = tau.get()
        if self.typ_abc != 'naABCs':
            # cO for habc
            cO = cp.sqrt(v2)#v*cp.cos(pi*g/2)*(source.ws[1]/self.f0)**g
            self.cO = cO.get()
            del cO
        del v, g, v2, yit, tau
        self.mp.free_all_blocks()
        
        # interior domain coordinates
        self.xyz = [self.vgme.xe[self.vgme.Nmax:-self.vgme.Nmax],\
               self.vgme.ye[self.vgme.Nmax:-self.vgme.Nmax],\
               self.vgme.ze[self.vgme.Nmax:-self.vgme.Nmax]]
        
        print('Modeling preparation done!')

    def LRiteration(self, ssample=None, tsample=None, clip=None, clim=None):
        r'''shot and time iteration using low-rank approximation viscoacoustic modeling
                ssample: sample shot No. (int list)
                tsample: sample time step. (int list)'''
        
        # get the default sample index if necessary
        if ssample is None:
            ssample=[self.acgeo.ns//2]
        if tsample is None:
            tsample=[self.nt//2]
        if clip is None:
            clip = 1
        
        # figure and axes handles
        fig = []
        ax = []
        
        # copy LR matrices into GPU
        Wd = cp.array(self.Wd)
        pkd = cp.array(self.pkd) 
        Wl = cp.array(self.Wl)
        pkl = cp.array(self.pkl)
        yit = cp.array(self.yit)
        tau = cp.array(self.tau)
        v2 = cp.array(self.v2)
        if self.typ_abc != 'naABCs':
            cO = cp.array(self.cO)
            r'''here we assume d used in habc are the same along three dimensions,
            this is a flaw of this package that requires futrue modification'''
            if len(list(set(self.vgme.d))) == 1:
                d = self.vgme.d[0]
            else:
                raise ImportError('''The modeling grid is not cubic! 
                                     The three dimensional grid size must be the same!''')
            wp = cp.array(self.ABC.wp)
            we = cp.array(self.ABC.we)
            wc = cp.array(self.ABC.wc)
            r = cp.array(self.cO*self.dt/d)
            a = 0.5
            b = np.sqrt(2)
            c = 1+np.sqrt(0.5)

        # outter loop for shot number
        barS = progressbar.ProgressBar(maxval=self.acgeo.ns, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        barT = progressbar.ProgressBar(maxval=self.nt, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        print('Source No. progress:')
        barS.start()
        for ish in range(self.acgeo.ns):
            barS.update(ish+1)
            # get shot position index and corresponding receiver position index
            sidx = cubeidx(self.acgeo.ids[ish]+self.vgme.Nmax)
            ridx = tuple((self.acgeo.idr_shot[ish]+self.vgme.Nmax).T)
            # create initial time step wavefield
            p0 = cp.zeros(self.ms,dtype=np.float32)
            p1 = cp.array(p0)
            if self.typ_abc != 'naABCs':
                pO = cp.array(p0)
            # create record
            record = cp.zeros((self.nt,self.acgeo.nr_shot[ish]),dtype=np.float32)
            
            # whether ish is the sample shot
            if ish in ssample:
                sb = True
                xyzi = self.acgeo.ids[ish]
            else:
                sb = False
            print('Time progress:') 
            barT.start()

            # inner loop for time step
            for it in range(self.nt):
                # progress bar
                barT.update(it+1)
                # dispersion term calculation
                P1 = fft.fftn(p1,axes=(0,1,2))
                pd = 0
                for i in range(self.ln):
                    pdi = fft.ifftn(P1*Wd[i],axes=(0,1,2)).real
                    pd += pkd[i]*pdi
                # clear some memory
                del P1, pdi
                # loss term calculation
                Pt = fft.fftn((p1-p0)/self.dt,axes=(0,1,2))
                pl = 0
                for i in range(self.ln):
                    pli = fft.ifftn(Pt*Wl[i],axes=(0,1,2)).real
                    pl += pkl[i]*pli
                # clear some memory
                del Pt, pli
                # update next time step wavefield
                sp = yit*pd+tau*pl
                # clear some memory
                del pl, pd
                p2 = self.dt**2*v2*sp+2*p1-p0
                del sp
                # special treatment if using hABCs
                if self.typ_abc != 'naABCs':
                    # absorbing boundary condition
                    #(1) for plane boundary
                    pO[self.ABC.Bp] = \
                        (r[self.ABC.Bp]*(p2[self.ABC.Ip]-(p0[self.ABC.Ip]-p0[self.ABC.Bp]))\
                         -(p2[self.ABC.Ip]-2*p1[self.ABC.Ip]+p0[self.ABC.Ip]-2*p1[self.ABC.Bp]+p0[self.ABC.Bp])\
                         +a*r[self.ABC.Bp]**2\
                         *(p1[self.ABC.Bnp[0]]+p1[self.ABC.Bnp[1]]+p1[self.ABC.Bnp[2]]+p1[self.ABC.Bnp[3]]\
                           +p1[self.ABC.Inp[0]]+p1[self.ABC.Inp[1]]+p1[self.ABC.Inp[2]]+p1[self.ABC.Inp[3]]\
                           -4*p1[self.ABC.Bp]-4*p1[self.ABC.Ip]))/(1+r[self.ABC.Bp])
                    #(2) for edge boundary
                    pO[self.ABC.Be] = \
                        (b*p1[self.ABC.Be]\
                         +r[self.ABC.Be]*(p2[self.ABC.Bne[0]]+p2[self.ABC.Bne[1]]))/(b+2*r[self.ABC.Be])
                    #(3) for corner boundary
                    #Bnc:(Bx, By, Bz), Bnnc:(Bxx, Byy, Bzz), Bndc:(Bxy, Byz, Bzx)
                    pO[self.ABC.Bnc[0]] = (c*p1[self.ABC.Bnc[0]]+r[self.ABC.Bnc[0]]\
                           *(pO[self.ABC.Bnnc[0]]+pO[self.ABC.Bndc[0]]+pO[self.ABC.Bndc[2]]))\
                          /(c+3*r[self.ABC.Bnc[0]])
                    pO[self.ABC.Bnc[1]] = (c*p1[self.ABC.Bnc[1]]+r[self.ABC.Bnc[1]]\
                           *(pO[self.ABC.Bnnc[1]]+pO[self.ABC.Bndc[0]]+pO[self.ABC.Bndc[1]]))\
                          /(c+3*r[self.ABC.Bnc[1]])
                    pO[self.ABC.Bnc[2]] = (c*p1[self.ABC.Bnc[2]]+r[self.ABC.Bnc[2]]\
                           *(pO[self.ABC.Bnnc[2]]+pO[self.ABC.Bndc[1]]+pO[self.ABC.Bndc[2]]))\
                          /(c+3*r[self.ABC.Bnc[2]])
                    pO[self.ABC.Bc] = (c*p1[self.ABC.Bc]+r[self.ABC.Bc]*\
                           (pO[self.ABC.Bnc[0]]+pO[self.ABC.Bnc[1]]+pO[self.ABC.Bnc[2]]))\
                          /(c+3*r[self.ABC.Bc])
                    #(4) average TWWE and OWWE
                    p2[self.ABC.Bp] = p2[self.ABC.Bp]*wp+pO[self.ABC.Bp]*(1-wp)
                    p2[self.ABC.Be] = p2[self.ABC.Be]*we+pO[self.ABC.Be]*(1-we)
                    p2[self.ABC.Bc] = p2[self.ABC.Bc]*wc+pO[self.ABC.Bc]*(1-wc)
                
                # implement source
                if it < self.source.sl:
                    p2[sidx] += self.source.ss[it]
                
                # move forward
                p0 = p1
                p1 = p2

                # test whether to sample and display
                if sb:
                    if (it in tsample):
                        p2d = p2.get()
                        smd = p2d[self.vgme.Nmax:-self.vgme.Nmax,\
                                  self.vgme.Nmax:-self.vgme.Nmax,\
                                  self.vgme.Nmax:-self.vgme.Nmax]
                        figt, axt = show3D(smd
                                    ,xyz=self.xyz
                                    ,xyzi=xyzi
                                    ,ea=(30,-15)
                                    ,rcstride=(1,1)
                                    ,clip=clip
                                    ,clim=clim)
                        fig.append(figt)
                        ax.append(axt)
                        smd.tofile(f'{self.outpath}/snapshot_it={it}.dat')
                        
                # record traces
                record[it,:] = p2[ridx]
            # finish source progress bar
            barT.finish()
            # save the record
            rec = record.get()
        # finish time progress bar
        barS.finish()
        
        # delete all variables and free all blocks
        del Wd, pkd, Wl, pkl, yit, tau, v2, p0, p1, p2, record
        if self.typ_abc == 'hABCs':
            del cO, wp, we, wc, r, pO
        self.mp.free_all_blocks()

        return rec, fig, ax