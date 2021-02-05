import numpy as np
from math import isinf, pi
import matplotlib.pyplot as plt
import matplotlib.colors as pcl
import matplotlib.cm as cm
import pandas as pd
from itertools import product
from basicFun import expand
import progressbar
from collections import deque

class stability_cal:
    r'''According to given reference frequency and time step, calculate the stability factors for sampled gamma;
        then, interpolat stability factor for any given gamma'''
    def __init__(self, w0, dt, typ=1, gu=None, nrs=1e4, ngs=1e3):
        # input
        self.w0 = w0 # reference angular frequency (scalar, float)
        self.dt = dt # temporal step size (scalar, float)
        if gu is None:
            self.gu = 0.499
        else:
            self.gu = gu # gamma upper bound for calculating stability factor (scalar, float)
        self.typ = typ # type of compensators: 0: no compensator,1: compensators A,2: compensators B
        # built-in r samples and gamma samples
        self.rs = np.linspace(1e-5,1,nrs) # courant number samples (1darray, float, (nrs))
        self.gs = np.linspace(1e-8,self.gu,ngs) # gamma samples (1darray, float, (ngs))
        self.kmax = np.sqrt(3)*pi # maximum kappa in 3-D (scalar, float)
        
        # calculation
        self.k0 = self.w0*self.dt/self.rs # reference angular wavenumber (1darray, float, (nrs))
        self.sf = self.scal() # stability factor samples corresponding to gamma samples (1darray, float, (ngs))
    
    def sinterp(self, ga):
        r'''interpolate (nearest) s according to given gamma'''
        if ga<self.gs[0]:
            return self.sf[0]
            
        if ga>self.gs[-1]:
            raise ImportError(f'Given gamma is larger than the upper bound ({self.gu})!')
        
        dg = ga-self.gs
        ind = np.where(dg==np.amax(dg[dg<=0]))
        return self.sf[ind][0]
    
    def scal(self):
        r'''calculate stability factor for sampled gamma'''
        s = np.zeros(len(self.gs))
        for i,gt in enumerate(self.gs):
            chg,cg,sg = self.gcscal(gt)
            k0g = self.kgcal(gt)
            D,L = self.DLcal(gt,chg,cg,sg,k0g)
            if self.typ==0:
                fdk = 1
                flk = 1
            else:
                if self.typ==1:
                    fdk, flk = self.fdlkcalA(gt,D,chg)
                else:
                    fdk, flk = self.fdlkcalB(gt)
            g0, g1 = self.g01cal(D,L,fdk,flk)
            mask = g0<=(1-np.abs(g1))
            if all(mask):
                s[i] = self.rs[-1]
            else:
                ind = np.where(~mask)
                s[i] = self.rs[ind[0][0]-1]
                
        return s

    def gcscal(self,g):
        r'''calculate gamma-only ralted terms:
            g-1darray or scalar, float'''
        chg = np.cos(pi*g/2)
        cg = np.cos(pi*g)
        sg = np.sin(pi*g)

        return chg, cg, sg

    def kgcal(self,g):
        r'''calculate k and g related terms:
            g-scalar, float'''
        k0g = self.k0**(-g)

        return k0g
    
    def DLcal(self,g,chg,cg,sg,k0g):
        r'''calculate D and L terms:
            g-scalar, float
            chg,cg,sg,k0g-scalar, float'''
        D = self.rs*np.sqrt(cg)*chg*k0g*self.kmax**(1+g)
        L = self.rs*sg*chg**2*k0g**2*self.kmax**(1+2*g)
        
        return D, L
    
    def fdlkcalA(self,g,D,chg):
        r'''calculate compensators A derivative:
            g-float
            D-1darray or scalar, float
            L-1darray or scalar, float'''
        fdk = np.sinc(D/2/pi)**2
        flk = np.sinc(chg*self.rs*self.kmax/2/pi)**2
        
        return fdk, flk
    
    def fdlkcalB(self,g):
        r'''calculate compensators B derivative:
            g-float'''
        rk2 = self.rs*self.kmax/2
        fdk = np.sinc(rk2/pi)**(2+2*g)
        flk = np.sinc(rk2/pi)**(1+2*g)

        return fdk, flk

    # g0 and g1 calculation
    def g01cal(self,D,L,fdk,flk):
        r'''calculate g0 and g1 according to given disy and losy:
            D,L,fdk,flk-1darray or scalar, float'''
        
        g0 = L*flk-1
        g1 = -D**2*fdk-g0+1

        return g0, g1

class vgopt_model:
    r'''Global velocity-gamma optimization class'''
    def __init__(self, typ, d, dt, w0, w, vareps, delg=0.2, M=20, R=2, gu=0.499):
        # overall model parameters
        self.d = d # grid size (scalar, float)
        self.dt = dt # modeling temporal step size (scalar, float)
        self.w0 = w0 # reference angular frequency (scalar, float)
        self.ws = w # intial angular frequency samples (1darray, float, (m))
        
        # optimization parameters
        self.delg = delg # search radius for gamma (scalar, float)
        self.M = M # sampling interval number within testing gamma range (scalar, int)
        self.R = R # maximum times of subsampling for testing gamma (scalar, int)
        self.gu = gu # gamma upper bound that can be modelled (scalar, float)
        self.vareps = vareps # maximum tolerance of returning residual (scalar, float)
        self.s = stability_cal(self.w0, self.dt, typ, gu=self.gu) # stability test class (stability_cal calss)

class vgopt:
    r'''velocity-gamma optimization'''
    def __init__(self, vb, gb, mp, antialias=True):
        self.mp = mp # model parameters (vgopt_model class)
        self.vb = vb # boundary point velocity (scalar, float)
        self.gb = gb # boundary point gamma (scalar, float)
        self.w = np.array(mp.ws) # angular frequency samples (1darray, float, (<=m,))
        self.vd = self.vb*self.vdfcal(self.gb) # dispersive velocity at sampled frequency (1darray, float, (m))
        self.aa = antialias # whether implement the antialias constraint for the absorbing layers
        if antialias:
            mask = (self.mp.ws*self.mp.d) > (self.vd*pi)
            if any(mask):
                raise ImportError(f'Given (vb:{vb},gb:{gb}) pair will create aliasing for modeled frequencies!')
        
    def vdfcal(self, ga, w=None):
        r'''calculate dispersive velocity factor (beta) according to gamma'''
        if w is None:
            w = self.w
        F = np.cos(0.5*pi*ga)*(w/self.mp.w0)**ga

        return F
    
    def attcal(self, va, ga, F=None):
        r'''calculate attenuation amount within given layer'''
        if F is None:
            F = self.vdfcal(ga)
        vda = va*F
        psi = self.mp.d*ga/vda
        y = np.exp(-pi*self.w*psi)
        
        return y
    
    def Ncal(self):
        r'''optimize N, va, ga according to mp'''
        I = 0
        va = []
        ga = []
        eps = []
        while(1):
            if I == 0:
                alp = self.attcal(self.vb, self.gb)
                vdo = self.vd
                go = self.gb
            else:
                alp *= (1-r**2)*Ao
            epsi = self.mp.vareps/len(self.w)
            eps.append(epsi)
            C = epsi/alp
            # for frequency samples whose C larger than 1, it means this frequency has satisfied the tolerance and does not need to be considered anymore
            mask = C<1
            if np.sum(mask)==0:
                N = I
                break
            else:
                I += 1
            alp = alp[mask]
            C = C[mask]
            vdo = vdo[mask]
            self.w = self.w[mask]
            vo, go, Ao, vdo, r = self.vgIcal(alp, C, vdo, go)
            # save vo and go
            va.append(vo)
            ga.append(go)
        # restore w
        self.w = np.array(self.mp.ws)
        return N, va, ga, eps
    
    def vgIcal(self, alp, C, vd, ga):
        r'''calculate optimal va and ga according to given C and vd
            alp-current accumulated attenuation (1darray, float, (nw--,))
            C-eps/alp (1darray, float, (nw--,))
            vd-nearest interior dispersive velocity (1darray, float, (nw--,))
            ga-nearest interior gamma (scalar, float)'''
        # sample ga vincinity
        gl = ga-self.mp.delg
        if gl<0:
            gl = 0
        gr = ga+self.mp.delg
        if gr>self.mp.gu:
            gr = self.mp.gu
        Jm = float("inf")
        fC = (1-C)/(1+C)
        fCl = vd*fC
        fCu = vd/fC
        M = 0
        dM = self.mp.M
        R = self.mp.R
        while(1):
            M += dM
            while(R):
                flag = False
                dg = (gr-gl)/M
                gas = np.linspace(gl, gr, M+1)
                for gs in gas:
                    fs = self.vdfcal(gs)
                    vlb1 = np.amax(fCl/fs)
                    vub1 = np.amin(fCu/fs)
                    vlb2 = np.amax(self.w*self.mp.d/pi/fs)
                    sg = self.mp.s.sinterp(gs)
                    vub2 = sg*self.mp.d/self.mp.dt
                    if self.aa:
                        vlb = max([vlb1,vlb2])
                    else:
                        vlb = vlb1
                    vub = min([vub1,vub2])
                    if vlb>vub:
                        continue
                    vs = vlb
                    As = self.attcal(vs,gs,fs)
                    Js = np.amax(As)
                    if Js<=Jm:
                        Jm = Js
                        vo = vs
                        go = gs
                        Ao = As
                        vdo = vo*fs
                        flag = True
                if flag:
                    R -= 1
                    gl = go-dg
                    if gl<0:
                        gl = 0
                    gr = go+dg
                    if gr>self.mp.gu:
                        gr = self.mp.gu
                else:
                    break
            # if already got the valid Jm, no need to increase M    
            if not isinf(Jm):
                break
        # compute reflectivity
        r = np.abs(vdo-vd)/(vdo+vd)
        return vo, go, Ao, vdo, r

    def ABCtest(self, va, ga, disp=True):
        r'''Test ABC for different frequency samples and different layers'''
        N = len(va)
        nw = len(self.mp.ws)
        vd = np.array(self.vd)
        psi = self.gb*self.mp.d/vd
        alp = np.exp(-pi*psi*self.mp.ws)
        y = np.zeros((N+1,nw))
        for i in range(N):
            fa = self.vdfcal(ga[i],w=self.mp.ws)
            vda = va[i]*fa
            r = np.abs(vda-vd)/(vda+vd)
            y[i] = r*alp
            vd = np.array(vda)
            psi = ga[i]*self.mp.d/vd
            alp *= (1-r**2)*np.exp(-pi*psi*self.mp.ws)
        y[N] = alp
        
        # display y
        y = y.T
        if disp:
            cN = pcl.Normalize(vmin=np.amin(y), vmax=np.amax(y))
            fig, ax = plt.subplots(1,1,figsize=(10,3))
            ax.imshow(y,cmap='gray',origin='lower',extent=[1,N,self.mp.ws[0]/(2*pi),self.mp.ws[-1]/(2*pi)],aspect=3/7*(N-1)*2*pi/(w[-1]-w[0]))
            ax.set_xticks(np.array(np.linspace(1,N,5),dtype=np.int16))
            ax.set_xlabel('Absorbing layer No.',fontsize=15)
            ax.set_ylabel('Frequency (Hz)',fontsize=15)
            fig.colorbar(cm.ScalarMappable(norm=cN, cmap='gray'))
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(13)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(13)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)
        
        return y
        
# expand the original v and Q model properly for Natural-attenuation absorbing boundaruy condition
class expand_model_naABC:
    r'''expand the velocity and Q model according to given absorbing parameters:
        the obtained class is a new model class includes both expanded velocity and gamma model information'''
    def __init__(self, typ_tsc, vm, Qm, dt, w, vareps=None, dvg=[2,1e-4]):
        r'''typ_tsc-temporal compensator type: (0-no compensator; 1-Compensator A; 2-Compensators B)
            vm-velocity model (model class)
            Qm-Q model (model class)
            dt-temporal step size (float scalar)
            w-frequency samples (1darray, float, (nw,))
            vareps-initial tolerance of returning amplitude from absorbing layers (float scalar)
            dvg-v and gamma intervals:
                dv-velocity interval for solving new boundary parameter (float scalar)
                dg-gamma interval for solving new boundary parameter (float scalar)'''
        
        # test the consistency between vm and Qm
        if not(vm.d == Qm.d) or not(vm.n == Qm.n) or not(vm.o == Qm.o):
            raise ImportError('vm and Qm are not in consistent space!')
        if typ_tsc not in (0,1,2):
            raise ImportError(f'typ_tsc={typ_tsc}, but it must among 0, 1 and 2!')
        else:
            self.typ_tsc = typ_tsc # compensator type
            
        self.vm = np.array(vm.md) # velocity array (3darray, float, n)
        self.gm = np.array(np.arctan(1/Qm.md)/pi) # gamma array (3darray, float, n)
        self.dt = dt # temporal step size (scalar, float)
        self.w = np.array(w) # valid frequency samples (1darray, float, (nw,))
        self.n = np.array(vm.n) # models' original dimension (list, int, (3))
        self.o = np.array(vm.o) # models' original origin (list, float, (3))
        self.d = np.array(vm.d) # models' original cell size (list, float, (3))
        self.x, self.y, self.z = np.array(vm.x), np.array(vm.y), np.array(vm.z) # models' original grid coordinates (1darray, float, (n[012],))
        self.w0 = vm.w0 # velocity model reference frequency

        if (vareps==None) or (vareps==0):
            # no abc (cyclic boundary)
            self.Nmax = 0
            self.vme, self.gme = self.vm, self.gm
        else:
            # abc
            self.dv,self.dg = dvg
            self.eps = vareps
            self.vgu = self._findboundary() # unique boundary (v,g) pairs (2darray, float, (nu,2))
            self.Nmax, self.vga = self.vgacal() # abc layer number and (va,ga) arrays for every (v,g) pairs (3darray, float, (nu, (2,Na))); Nmax: maximum absorbing layer number for all (v,g) pairs (scalar, int) 
            self.vme, self.gme = self.expand_abcs() # abc expanded velocity and gamma arraies (3darray, float, ne)
            
        self.ne = np.array(self.n)
        self.oe = np.array(self.o)
        for i in range(3):
            self.ne[i] += 2*(self.Nmax) # expanded model dimension (list, int, (3))
            self.oe[i] -= self.Nmax*self.d[i] # expanded model origin (list, float, (3))
        # expanded model grid coordinates (1darray, float, (ne[012],))
        self.xe,self.ye,self.ze = deque(np.arange(self.oe[i],self.oe[i]+self.d[i]*self.ne[i],self.d[i]) for i in range(3))
                
    def _findboundary(self):
        r'''find unique boundary (v,g) pairs'''
        # first dimension as boundary
        vg0 = [[self.vm[i,j,k],self.gm[i,j,k]] for i,j,k in product([0,self.n[0]-1], range(self.n[1]), range(self.n[2]))]
        # second dimension as boundary
        vg1 = [[self.vm[i,j,k],self.gm[i,j,k]] for i,j,k in product(range(self.n[0]), [0,self.n[1]-1], range(self.n[2]))]
        # third dimension as boundary
        vg2 = [[self.vm[i,j,k],self.gm[i,j,k]] for i,j,k in product(range(self.n[0]), range(self.n[1]), [0,self.n[2]-1])]
        # combine vg012
        vg = np.concatenate((vg0,vg1,vg2),axis=0)
        # round up vg according to dv and dg
        vr = np.around(vg[:,0]/self.dv)*self.dv
        gr = np.around(vg[:,1]/self.dg)*self.dg
        vgrc = vr+1j*gr
        # find unique (vr,qr) pairs
        vguc = np.unique(vgrc)
        vgu = np.stack((vguc.real,vguc.imag),axis=0).T
        
        return vgu
    
    def vgacal(self):
        r'''calculate optimal absorbing layer parameters according to frequency samples and absorbing residual eps'''
        N = self.vgu.shape[0]
        print(f'Total No. of unique roundup (v,g) pairs: {N}.')
        y = np.empty(N,dtype=np.ndarray)
        Na = np.zeros(N,dtype=np.int16)
        mp = vgopt_model(self.typ_tsc, d=self.d[0], dt=self.dt, w0=self.w0, w=self.w, vareps=self.eps)
        #********progress bar***********#
        barN = progressbar.ProgressBar(maxval=N, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        barN.start()
        #********progress bar***********#
        for i in range(N):
            barN.update(i+1)
            vb = self.vgu[i,0]
            gb = self.vgu[i,1]
            opt = vgopt(vb,gb,mp)
            Na[i],va,ga, _ = opt.Ncal()          
            y[i] = np.stack((va,ga))
        barN.finish() # finish progress bar
        # expand y according to maximum of Na
        Nm = np.max(Na)
        vA = np.zeros((N,Nm))
        gA = np.zeros((N,Nm))
        for i in range(N):
            vA[i][-Na[i]:] = y[i][0]
            gA[i][-Na[i]:] = y[i][1]
            if Na[i]<Nm:
                vA[i][:-Na[i]] = self.vgu[i,0]
                gA[i][:-Na[i]] = self.vgu[i,1]
        vga = np.stack((vA,gA))
        return Nm, vga
    
    def _vgaselect(self,i,j,k):
        vb = np.around(self.vm[i,j,k]/self.dv)*self.dv
        gb = np.around(self.gm[i,j,k]/self.dg)*self.dg
        mask = (np.abs(self.vgu[:,0]-vb)<1e-2)*(np.abs(self.vgu[:,1]-gb)<1e-8)
        va = self.vga[0][mask]
        ga = self.vga[1][mask]
        
        return va, ga

    def expand_abcs(self):
        r'''expanding v and Q models according to obtained optimal velocity and gamma abcs'''
        # create the expanded models
        V = expand(self.vm, self.Nmax)
        G = expand(self.gm, self.Nmax)
        # loop through all boundary nodes
        # plane boundary expanding
        print('Plane boundary expanding...')
        # 1st-dim
        I0 = (np.flip(range(self.Nmax)))
        I1 = (np.array([range(self.Nmax+self.n[0],2*self.Nmax+self.n[0])]))
        for i,j,k in product([0,self.n[0]-1], range(self.n[1]), range(self.n[2])):
            va, ga = self._vgaselect(i,j,k)
            if i:
                I = I1
            else:
                I = I0
            J = j+self.Nmax
            K = k+self.Nmax
            V[I,J,K] = va
            G[I,J,K] = ga
        # 2nd-dim
        J0 = (np.flip(range(self.Nmax)))
        J1 = (np.array(range(self.Nmax+self.n[1],2*self.Nmax+self.n[1])))
        for i,j,k in product(range(self.n[0]),[0,self.n[1]-1], range(self.n[2])):
            va, ga = self._vgaselect(i,j,k)
            if j:
                J = J1
            else:
                J = J0
            I = i+self.Nmax
            K = k+self.Nmax
            V[I,J,K] = va
            G[I,J,K] = ga
        # 3rd-dim
        K0 = (np.flip(range(self.Nmax)))
        K1 = (np.array(range(self.Nmax+self.n[2],2*self.Nmax+self.n[2])))
        for i,j,k in product(range(self.n[0]),range(self.n[1]),[0,self.n[2]-1]):
            va, ga = self._vgaselect(i,j,k)
            if k:
                K = K1
            else:
                K = K0
            I = i+self.Nmax
            J = j+self.Nmax
            V[I,J,K] = va
            G[I,J,K] = ga
        
        # edge boundary
        print('Edge boundary expanding...')
        # 1-2 dims
        K = (np.array(range(self.n[2]))+self.Nmax)
        for i,j in product([[0,-1],[self.n[0]-1,1]], [[0,-1],[self.n[1]-1,1]]):
            I0 = i[0]+self.Nmax
            J0 = j[0]+self.Nmax
            for inc,jnc in product(range(1,self.Nmax+1),range(1,self.Nmax+1)):
                I = I0+inc*i[1]
                J = J0+jnc*j[1]
                if inc>=jnc:
                    V[I,J,K] = V[I,J0,K]
                    G[I,J,K] = G[I,J0,K]
                else:
                    V[I,J,K] = V[I0,J,K]
                    G[I,J,K] = G[I0,J,K]
        # 1-3 dims
        J = (np.array(range(self.n[1]))+self.Nmax)
        for i,k in product([[0,-1],[self.n[0]-1,1]], [[0,-1],[self.n[2]-1,1]]):
            I0 = i[0]+self.Nmax
            K0 = k[0]+self.Nmax
            for inc,knc in product(range(1,self.Nmax+1),range(1,self.Nmax+1)):
                I = I0+inc*i[1]
                K = K0+knc*k[1]
                if inc>=knc:
                    V[I,J,K] = V[I,J,K0]
                    G[I,J,K] = G[I,J,K0]
                else:
                    V[I,J,K] = V[I0,J,K]
                    G[I,J,K] = G[I0,J,K]
        # 2-3 dims
        I = (np.array(range(self.n[0]))+self.Nmax)
        for j,k in product([[0,-1],[self.n[1]-1,1]], [[0,-1],[self.n[2]-1,1]]):
            J0 = j[0]+self.Nmax
            K0 = k[0]+self.Nmax
            for jnc,knc in product(range(1,self.Nmax+1),range(1,self.Nmax+1)):    
                J = J0+jnc*j[1]
                K = K0+knc*k[1]
                if jnc>=knc:
                    V[I,J,K] = V[I,J,K0]
                    G[I,J,K] = G[I,J,K0]
                else:
                    V[I,J,K] = V[I,J0,K]
                    G[I,J,K] = G[I,J0,K]
                
        # corner boundary
        print('Corner boundary expanding...')
        for i,j,k in product([[0,-1],[self.n[0]-1,1]], [[0,-1],[self.n[1]-1,1]], [[0,-1],[self.n[2]-1,1]]):
            I0 = i[0]+self.Nmax
            J0 = j[0]+self.Nmax
            K0 = k[0]+self.Nmax
            for inc,jnc,knc in product(range(1,self.Nmax+1),range(1,self.Nmax+1),range(1,self.Nmax+1)):
                I = I0+inc*i[1]
                J = J0+jnc*j[1]
                K = K0+knc*k[1]
                ijknc = np.array([inc,jnc,knc])
                maxnc = np.amax(ijknc)
                maxid = np.where(maxnc==ijknc)[0][0]
                if maxid == 0:
                    V[I,J,K] = V[I,J0,K0]
                    G[I,J,K] = G[I,J0,K0]
                else:
                    if maxid == 1:
                        V[I,J,K] = V[I0,J,K0]
                        G[I,J,K] = G[I0,J,K0]
                    else:
                        V[I,J,K] = V[I0,J0,K]
                        G[I,J,K] = G[I0,J0,K]
                
        return V, G        
    
    
    
    
    
    