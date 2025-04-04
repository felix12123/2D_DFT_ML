import numpy as np
from scipy import special as sp
from scipy import fft


def initial_kernel(Nx,Ny,NNy,Lx,Ly,R):
    kx = np.zeros(Nx)
    ky = np.zeros(NNy)
    
    w1 = np.zeros((Nx,NNy))
    w2 = np.zeros((Nx,NNy))
    wx = np.zeros((Nx,NNy))
    wy = np.zeros((Nx,NNy))
    wxx = np.zeros((Nx,NNy))
    wyy = np.zeros((Nx,NNy))
    wxy = np.zeros((Nx,NNy))    
    
    with np.errstate(divide='ignore', invalid='ignore'):
        
        for i in range(Nx):
            n=i
            if (i>Nx/2):
                n = i-Nx
            kx[i] = n*2*np.pi/Lx
        for j in range(NNy):
            ky[j] = j*2*np.pi/Ly
            
        
        for i in range(Nx):
            for j in range(NNy):
                k = np.sqrt(kx[i]**2+ky[j]**2)
                w1[i][j]=2*np.pi*R*sp.j0(k*R)
                
        for i in range(Nx):
            for j in range(NNy):
                k = np.sqrt(kx[i]**2+ky[j]**2)
                w2[i][j]=2*np.pi/k*R*sp.j1(k*R)
        w2[0][0]=np.pi*R**2
        
        for i in range(Nx):
            for j in range(NNy):
                k = np.sqrt(kx[i]**2+ky[j]**2)
                wx[i][j]=-2*np.pi*R*kx[i]/k*sp.j1(k*R)
        wx[0][0]=0
        
        for i in range(Nx):
            for j in range(NNy):
                k = np.sqrt(kx[i]**2+ky[j]**2)
                wy[i][j]=-2*np.pi*R*ky[j]/k*sp.j1(k*R)
        wy[0][0]=0
        
        for i in range(Nx):
            for j in range(NNy):
                k = np.sqrt(kx[i]**2+ky[j]**2)
                wxx[i][j]=2*np.pi*(-R*kx[i]**2/k**2*sp.jv(2,k*R)+1.0/k*sp.j1(k*R))
        wxx[0][0]=np.pi*R
        
        for i in range(Nx):
            for j in range(NNy):
                k = np.sqrt(kx[i]**2+ky[j]**2)
                wxy[i][j]=2*np.pi*(-R*kx[i]*ky[j]/k**2*sp.jv(2,k*R))
        wxy[0][0]=0
        
        for i in range(Nx):
            for j in range(NNy):
                k = np.sqrt(kx[i]**2+ky[j]**2)
                wyy[i][j]=2*np.pi*(-R*ky[j]**2/k**2*sp.jv(2,k*R)+1.0/k*sp.j1(k*R))
        wyy[0][0]=np.pi*R
    return w1,w2,wx,wy,wxx,wyy,wxy



def get_weighted_density_functions(Nx,Ny,NNy,w1,w2,wx,wy,wxx,wyy,wxy):
    def cal_n1 (rho):
        return  fft.irfft2(fft.rfft2(rho)*w1)
    def cal_n2 (rho):
        return  fft.irfft2(fft.rfft2(rho)*w2)
    def cal_nx (rho):
        n1x = fft.rfft2(rho)*wx*1j
        for j in range(NNy):
            n1x[int(Nx/2)][j]=0 #kill nyquist frequency
        return  fft.irfft2(n1x)
    def cal_ny (rho):
        n1y = fft.rfft2(rho)*wy*1j #1j for wy is imagenary part
        for i in range(Nx):
            n1y[i][int(Ny/2)]=0
        return  fft.irfft2(n1y)
    
    def cal_nxx (rho):
        n = fft.rfft2(rho)*wxx
        return  fft.irfft2(n)
    
    def cal_nyy (rho):
        n = fft.rfft2(rho)*wyy
        return  fft.irfft2(n)
    
    def cal_nxy (rho):
        n = fft.rfft2(rho)*wxy
        for i in range(Nx):
            n[i][int(Ny/2)]=0
        for j in range(NNy):
            n[int(Nx/2)][j]=0
        return  fft.irfft2(n)
    
    def cal_conv_nx (rho):
        n1x = fft.rfft2(rho)*wx*(-1j) # -1 for w(-r)=-w(r)
        for j in range(NNy):
            n1x[int(Nx/2)][j]=0
        return  fft.irfft2(n1x)
    
    def cal_conv_ny (rho):
        n1y = fft.rfft2(rho)*wy*(-1j)
        for i in range(Nx):
            n1y[i][int(Ny/2)]=0
        return  fft.irfft2(n1y)
    return cal_n1, cal_n2, cal_nx, cal_ny, cal_nxx, cal_nyy, cal_nxy, cal_conv_nx, cal_conv_ny

def get_c1_func(R,cal_n1, cal_n2, cal_nx, cal_ny, cal_nxx, cal_nyy, cal_nxy, cal_conv_nx, cal_conv_ny):
    
    def cal_c1c(rho): #calculate delta F_exc / delta rho
        a = 11/4
        c0 = (a+2)/3
        c1 = (a-4)/3
        c2 = (2-2*a)/3
        
        n1 = cal_n1(rho)
        n2 = cal_n2(rho)
        nx = cal_nx(rho)
        ny = cal_ny(rho)
        nxx = cal_nxx(rho)
        nxy = cal_nxy(rho)
        nyy = cal_nyy(rho)

        """partial Fx partial nx"""
        F1_n1 = -np.log(1-n2)/2.0/np.pi/R
        F1_n2 = n1/(1-n2)/2/np.pi/R
        F2_n1 = 1.0/4/np.pi/(1-n2)*(2*c0*n1)
        F2_n2 = 1.0/4/np.pi/(1-n2)**2*(c0*n1**2+c1*(nx**2+ny**2)
                                       +c2*(nxx**2+nyy**2+2*nxy**2))
        F2_nx = 1.0/4/np.pi/(1-n2)*(2*c1*nx)
        F2_ny = 1.0/4/np.pi/(1-n2)*(2*c1*ny)
        F2_nxx = 1.0/4/np.pi/(1-n2)*(2*c2*nxx)
        F2_nyy = 1.0/4/np.pi/(1-n2)*(2*c2*nyy)
        F2_nxy = 1.0/4/np.pi/(1-n2)*(2*c2*nxy)*2.0
        
        return -(cal_n1(F1_n1)+cal_n1(F2_n1)+cal_n2(F1_n2)+cal_n2(F2_n2) 
                +cal_conv_nx(F2_nx)+cal_conv_ny(F2_ny)
                +cal_nxx(F2_nxx)+cal_nyy(F2_nyy)+cal_nxy(F2_nxy)
               )
    return cal_c1c

def picard_update(rho,mu,alpha,Nx,Ny,expmvext,cal_c1c):
    c1c = cal_c1c(rho)
    if c1c.shape[1] == c1c.shape[0] - 1: # for odd Nx/Ny we need to pad c1c. i suspect the fourier transforms reduce the size by one
        c1c = np.pad(c1c, ((0, 0), (0, 1)), 'constant')
    rho_new = np.exp(mu+c1c) * expmvext
    rho = (1-alpha)*rho+alpha*rho_new
    rho[rho<0] = 0 # komplett neue Zeile um keine Negativen Dichten zu kriegen
    rho[expmvext==0] = 0 # not needed
    error = np.max(np.abs(rho-rho_new))
    return rho,error

def picard_iteration(rho0,mu,Nx,Ny,expmvext,cal_c1c,tol):
    rho = np.full((Nx,Ny),rho0)#lazy way
    alpha = 10**-4
    alpha_min = alpha
    alpha_max = 0.05
    error_old = 100.0
    for i in range (100000):
        rho,error = picard_update(rho,mu,alpha,Nx,Ny,expmvext,cal_c1c)
        if (error < error_old):
            alpha = min(1.1*alpha, alpha_max)
        else:
            alpha = max(alpha/5, alpha_min)
        error_old = error
        if error<tol or i == 99999:
            return rho, rho - picard_update(rho,mu,alpha,Nx,Ny,expmvext,cal_c1c)[0] # rho - rho_new
        if np.isnan(error):
            print("ERROR: error not a real number")
            return None
    
    return None

def get_c1_and_rho_func(Nx,Ny,Lx,Ly,R):
    NNy = int(Ny/2)+1 # I still do not know exactly what that is
    w1,w2,wx,wy,wxx,wyy,wxy = initial_kernel(Nx,Ny,NNy,Lx,Ly,R)
    cal_n1, cal_n2, cal_nx, cal_ny, cal_nxx, cal_nyy, cal_nxy, cal_conv_nx, cal_conv_ny = get_weighted_density_functions(Nx,Ny,NNy,w1,w2,wx,wy,wxx,wyy,wxy)
    cal_c1c = get_c1_func(R,cal_n1, cal_n2, cal_nx, cal_ny, cal_nxx, cal_nyy, cal_nxy, cal_conv_nx, cal_conv_ny)

    X,Y = np.meshgrid(np.linspace(Lx/Nx/2,Lx-Lx/Nx/2,Nx),np.linspace(Ly/Ny/2,Ly-Ly/Ny/2,Ny))
    
    def f(mu,V,tol):
        expmvext = np.exp(-V(X,Y) if callable(V) else -V)        
        rho0 = 0.5
        rho, error = picard_iteration(rho0,mu,Nx,Ny,expmvext,cal_c1c,tol)
        return rho, cal_c1c(rho)
    return f

def get_rho_func(Nx, Ny, Lx, Ly, R):
    NNy = int(Ny/2)+1 # I still do not know exactly what that is
    w1,w2,wx,wy,wxx,wyy,wxy = initial_kernel(Nx,Ny,NNy,Lx,Ly,R)
    cal_n1, cal_n2, cal_nx, cal_ny, cal_nxx, cal_nyy, cal_nxy, cal_conv_nx, cal_conv_ny = get_weighted_density_functions(Nx,Ny,NNy,w1,w2,wx,wy,wxx,wyy,wxy)
    cal_c1c = get_c1_func(R,cal_n1, cal_n2, cal_nx, cal_ny, cal_nxx, cal_nyy, cal_nxy, cal_conv_nx, cal_conv_ny)

    X,Y = np.meshgrid(np.linspace(Lx/Nx/2,Lx-Lx/Nx/2,Nx),np.linspace(Ly/Ny/2,Ly-Ly/Ny/2,Ny))
    def f(mu,V,tol):
        expmvext = np.exp(-V(X,Y) if callable(V) else -V)        
        rho0 = 0.5
        rho, error = picard_iteration(rho0,mu,Nx,Ny,expmvext,cal_c1c,tol)
        return rho, np.abs(error)
    return f