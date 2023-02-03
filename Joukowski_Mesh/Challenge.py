from __future__ import division, print_function
import numpy as npy
from numpy import pi, sin, cos, tan, log10
from scipy import integrate
from scipy import optimize

#import pylab as pyl


#-----------------------------------
def Distance(i, dx_te, ratio):
    return dx_te*(ratio**i - 1) / (ratio-1)

def GradDist(i, dx_te, ratio):
    return dx_te*((i-1)*ratio**i - i*ratio**(i-1) + 1) / (ratio-1)**2


def FindStretching(n, h_min, Hc):
    # Find the ratio of successive cell sizes to get a total length of Hc from
    # n cells, starting with the first cell size = h_min.
    
    assert( h_min > 0 )
    # This guess is exact for very large size ratios
    guess = (Hc / h_min)**(1/(n-1))
    finished = False
    while not finished:
        func = Distance(n, h_min, guess) - Hc
        grad = GradDist(n, h_min, guess)
        delta = - func / grad
        guess += delta

        if (abs(delta) < 1.e-12):
            finished = True

    return guess;
    

#-----------------------------------
def tanh(i, nn, delta):
    return 1 + npy.tanh(delta*(i/nn - 1))/npy.tanh(delta)

def dtanhddelta(i, nn, delta):
    return (-1 + i/nn)/(npy.tanh(delta) * npy.cosh(delta* (-1 + i/nn))**2) - npy.tanh(delta* (-1 + i/nn))/(npy.sinh(delta)**2)

def find_tanh_delta( ds, nn ):
    
    guess = 2
    itmax = 1000
    it = 0
    finished = False
    while not finished and it < itmax:
        func = tanh(1, nn, guess) - ds
        grad = dtanhddelta(1, nn, guess)
        delta = - func / grad
        guess += delta
        it = it + 1
        #print guess, ' ', delta
        if (abs(delta) < 1e-12):
            finished = True
    
    #if it == itmax:
    #    assert(it < itmax)
    return guess

def coarsen(re, ref, maxref):
    i = maxref
    while i > ref:
        re = npy.delete(re, npy.s_[1::2], 0 )
        i = i-1
    return re

#-----------------------------------
def Bezier(nn, smax=1, ds0=-0.2, ds1=-0.2):
   
    s0 = npy.linspace(0,smax,nn+1)
      
    #Use a Bezier curve to cluster at LE and TE: ds = -1 gives a linear distribution. Clustering is added as ds->0 from -1
    #ds0 = -0.2
    #ds1 = -0.2
    P0 = 1
    P1 = (3 + ds1)/3
    P2 = -(ds0/3)
    s1 = P0*(1 - s0)**3 + P1*3*s0*(1 - s0)**2 + P2*3*s0**2*(1 - s0)
    return s1

#-----------------------------------
# def Bezier(nn, smax=1, ds0=0.2, dds0=0, ds1=0.2, dds1=0):
#   
#     ds0 = -ds0
#     ds1 = -ds1
#       
#     #ds0 = 2.5
#     #dds0 = 0.0
#     #ds1 = 0.175
#     #dds1 = 2.0
#      
#     ds0 = 0.2
#     dds0 = 0.0
#     ds1 = 0.1
#     dds1 = 0.0
#       
#     s = npy.linspace(0,smax,nn+1)
#       
#     #Use a Bezier curve to cluster at LE and TE: ds = 1 gives a linear distribution. Clustering is added as ds->0 from 1
#     P0 = 0
#     P1 = ds0/5
#     P2 = (dds1 + 8*ds0)/20.
#     P3 = (20 + dds0 - 8*ds1)/20.
#     P4 = (5 - ds1)/5.
#     P5 = 1
#     t = P0*(1 - s)**5 + P1*5*s*(1-s)**4 + P2*10*s**2*(1-s)**3 + P3*10*s**3*(1-s)**2+P4*5*s**4*(1-s) + P5*s**5
#     return 1-t

#-----------------------------------
def Cos(nn, smax=1):
    s0 = npy.linspace(0,smax,nn+1)
    return 1-0.5*(1-npy.cos(pi*s0))

#-----------------------------------
def Joukowski_wake_x(nchordwise, nn, Hc, ds0, ds1, AR = 1):
    
    frac = 2
    nAf = int(nchordwise/frac)
    
    #s = 1-npy.linspace(0,1/frac,nAf+1)
    #s = Cos(nAf,1/frac)
    s = Bezier(nAf,1/frac,ds0=ds0,ds1=ds1)
    
    # Joukwoski transformation
    a = 0.1
    den  = 1 + 2*a*(1 + a)*(1 + cos(pi*s)) ;
    xnum = (1 + a*(1 + 2*a)*(1 + cos(pi*s)))*(sin(0.5*pi*s))**2 ;
    x = 1-xnum/den;
 
    #for i in range(7,nn+1):
    #    x[i] = (i-6)*x[i]
    
    #x = x/x[-1]
    
    x /= AR
    nWake = nn-nAf
    dx = x[-1] - x[-2]
    
    re = npy.zeros(nn+1)
    re[0:nAf+1] = x
    ratio = FindStretching(nWake, dx, Hc-x[-1])
    for i in range(nWake+1):
        re[nAf+i] = x[-1] + Distance(i, dx, ratio)
    
    return re/Hc

def make_joukowski_challenge(ref, Q, reynolds=1.e6):
    #
    # Makes a quad or tri grid for an airfoil. This file must have two numbers
    # per line, each representing an (x,y) coordinate of a point.
    # The points should start at the trailing edge and loop clockwise.  
    # The trailing-edge is assumed closed (no gap), and the trailing-
    # edge point should not be repeated.  The number of points in
    # the file should be sufficient to represent the geometry well, but
    # it need not be a multiple of Q as the points will be re-splined.
    # An optional hard-coded analytical geometry function can be used to
    # nudge points to the true geometry (if using the spline is not enough).
    # The spacing of points on the geometry is done via a quasi-curvature
    # based method -- the optional pg2d size input controls this.
    # The generated pg2d is of the "C" type (see graphic make_airfoil.png).
    #
    # INPUTS:
    #   ref        : refinement number (useful for convergence studies)
    #   Q          : geometry order (e.g. 1,2,3,4,...)
    #   TriFlag    : False = quad, True = tri
    #   FileFormat : String file format switch
    #   reynolds   : Switch for choosing grid spacings
    #   filename_base : File name base for dump
    #
    
    maxref = 6
    assert( ref <= maxref )
    Dfarfield = 100    # farfield distance from the airfoil
    farang=0.0*pi/180.
    nchordwise=8       # number of elements along one side of the airfoil geometry
    nxwake=8           # x-wake on centerline
    nnormal=16         # points normal to airfoil surface
    wakeangle=0.0      # angle of wake leaving the airfoil
    rxwakefary = 0.35  # x-wake stretching far from airfoil (+/-y)
    
    # Trailing edge spacing
    if (reynolds > 5e5):
        # Turbulent. 
        ds1 = -0.2
        ds0 = -0.2
    else:
        # Laminar.  
        ds1 = -0.2
        ds0 = -0.2
    
    #--------------------#
    # load/spline points #
    #--------------------#
    X, saf = Joukowski(nchordwise*2**ref,Q,ds0,ds1) #Don't use the max refinement to make sure the high-order nodes are distributted well
    
    c = max(X[:,0]) - min(X[:,0])          # chord length
    Hc = Dfarfield*c                       # farfield distance
    
    xte = X[0,:];                          # TE point
    dx_te = X[0,0] - X[Q,0];
    XLE = X[npy.append(range(len(X)),0),:] # rest of airfoil
    nLE = len(XLE)

    #-------------------------------------#
    # put points down along farfield, FLE #
    #-------------------------------------#
    x0     = tan(farang)*Hc
    radius = (x0**2 + Hc**2)**0.5
    t0 = npy.linspace( 3.*pi/2.+farang, 5.*pi/2.-farang, nLE)

    FLE      = npy.zeros([nLE,2])
    FLE[:,0] = x0 - radius*cos(t0)
    FLE[:,1] =      radius*sin(t0)

    #pyl.plot(FLE[:,0],FLE[:,1],'-o')
    #pyl.show()
    
    #----------------------#
    # x-wake on centerline #
    #----------------------#
    nr0 = nxwake*2**maxref 

    re = Joukowski_wake_x(nchordwise*2**maxref, nr0, Hc, ds0, ds1)

    re = coarsen(re, ref, maxref)
    rw = spaceq(re, Q)
    
    #----------------------------------#
    # C-grid: put points on wake first #
    #----------------------------------#
    
    XWK = npy.flipud(npy.array([rw*Hc+xte[0], npy.zeros(len(rw))]).transpose())
    XWK[:,1] = (XWK[:,0]-xte[0])*tan(wakeangle)
    XWK2 = npy.flipud(XWK)

    nWK = len(XWK)
    
    #----------------------------------------#
    # x-wake spacing far from airfoil (+/-y) #
    #----------------------------------------#
    a  = 0.1
    b  = rxwakefary
    re = (npy.logspace(a,b,nr0+1) - 10**a)/(10**b-10**a)
    re = coarsen(re, ref, maxref)
    rbot = npy.flipud(spaceq(re, Q)*(Hc+xte[0]))
    
    FWK1 = npy.array([rbot,              XWK[:,1] - Hc - rbot*x0/Hc]).transpose()
    FWK2 = npy.array([npy.flipud(rbot), XWK2[:,1] + Hc + npy.flipud(rbot)*x0/Hc]).transpose()
    
    #-------------------#
    # Wake and boundary #
    #-------------------#
    XWB = npy.append(XWK,  XLE[1:-1,:], axis = 0)
    XWB = npy.append(XWB,  XWK2,        axis = 0)
    FWB = npy.append(FWK1, FLE[1:-1,:], axis = 0)
    FWB = npy.append(FWB,  FWK2,        axis = 0)
    
    nWB = len(XWB)
    
    #------------------#
    # points on C grid #
    #------------------#
    nr0 = nnormal*2**ref
    nr = 1 + nr0*Q
    XC = npy.zeros([nWB, nr])
    YC = npy.array(XC)


    # Spacing estations
    if (reynolds > 5e5):
        # Turbulent.  y+=1 for the first cell at the TE on the coarse pg2d
        coarse_yplus = 1
        dy_te = 5.82 * (coarse_yplus / reynolds**0.9) / 2**maxref
        wake_power = 0.8
        
        nr0 = nnormal*2**maxref
        #re = npy.zeros(nr0+1)
        #delta = find_tanh_delta( dy_te/Hc, nr0 )
        #for i2 in range(0, nr0+1):
        #    re[i2] = tanh(i2, nr0, delta)
        #ratio = FindStretching(nr0, dy_te, Hc)
        #for i2 in range(0, nr0+1):
        #    re[i2] = Distance(i2, dy_te, ratio)/Hc
        AR = 50
            
        re = Joukowski_wake_x(nchordwise*2**maxref, nr0, Hc, ds0, ds1, AR)

    else:
        # Laminar.  Put two cells across the BL at the TE on the coarse mesh
        AR = 1
        dy_te = 0.1 / reynolds**0.5 / 2**maxref
        wake_power = 0.5
        
        nr0 = nnormal*2**maxref
        re = Joukowski_wake_x(nchordwise*2**maxref, nr0, Hc, ds0, ds1, AR)

    #print "dy_te = ", dy_te, re[1]*Hc

    re = coarsen(re, ref, maxref)
    r0 = spaceq(re, Q)

    for i in range(nWB):
        #iplus = min(nWB-1, i+1)
        #iminus = max(0, i-1)
        #ds = ((XWB[iplus,0] - XWB[iminus,0])**2 +
        #      (XWB[iplus,1] - XWB[iminus,1])**2)**0.5/ (iplus - iminus)
              
        #dy = dy_te * max(XWB[i,0],1)**wake_power
        # print(XWB[iplus,0], XWB[iminus,0], ds, dy, iplus, iminus)
        #re = npy.zeros(nr0+1)
        #ratio = FindStretching(nr0, dy, Hc)
        #for i2 in range(0, nr0+1):
            #print(i2, Distance(i2, dy, ratio)/Hc)
        #    re[i2] = Distance(i2, dy, ratio)/Hc
            
        #delta = find_tanh_delta( dy/Hc, nr0 )
        #for i2 in range(0, nr0+1):
        #    re[i2] = tanh(i2, nr0, delta)

        #re = coarsen(re, ref, maxref)
        #r0 = spaceq(re, Q)
        #print(re)
    
        r = r0
        #if i < nWK-1 or i > nWB-nWK-1:
        #    xx = (XWB[i,0]-XWK[-1,0])/max(XWB[:,0])
        #    r = r0 * (1-xx) + r1 * xx
        XC[i,:] = XWB[i,0] + r*(FWB[i,0]-XWB[i,0])
        YC[i,:] = XWB[i,1] + r*(FWB[i,1]-XWB[i,1])
    
    return XC, YC
    
#-----------------------------------
def block_elem(N, Q):
    nx, ny = N.shape;
    #if (Q != 1) and ((mod(nx,Q) != 1) or (mod(ny,Q) != 1)): print('ERROR 2'); return;
    mx = int((nx-1)/Q);
    my = int((ny-1)/Q);
    E = npy.zeros( (mx*my,(Q+1)*(Q+1)),int);
    i = 0;
    for imy in range(my):
        for imx in range(mx):
            ix = Q*(imx+1)-(Q-1)-1;
            iy = Q*(imy+1)-(Q-1)-1;
            k = 0;
            for ky in range(Q+1):
                for kx in range(Q+1):
                    E[i,k] = N[ix+kx,iy+ky]
                    k = k+1;

            i = i + 1;
      
    return E



#-----------------------------------
def Joukowski_xy(s,a):
    den  = 1 + 2*a*(1 + a)*(1 + cos(pi*s)) ;
    xnum = (1 + a*(1 + 2*a)*(1 + cos(pi*s)))*(sin(0.5*pi*s))**2 ;
    ynum = 0.5*a*(1 + 2*a)*(1 + cos(pi*s))*sin(pi*s) ;
    x = xnum/den ;
    y = ynum/den ;
    
    return x, y

#-----------------------------------
def Joukowski_dxy_ds(s,a):
    den  = 1 + 2*a*(1 + a)*(1 + cos(pi*s)) ;
    xnum = (1 + a*(1 + 2*a)*(1 + cos(pi*s)))*(sin(0.5*pi*s))**2 ;
    ynum = 0.5*a*(1 + 2*a)*(1 + cos(pi*s))*sin(pi*s) ;

    den_ds  = -2*a*(1 + a)*pi*sin(pi*s) ;
    xnum_ds = pi*cos((pi*s)/2.)*(1 + a*(1 + 2*a)*(1 + cos(pi*s)))*sin((pi*s)/2.) - a*(1 + 2*a)*pi*sin((pi*s)/2.)**2*sin(pi*s) ;
    ynum_ds = (a*(1 + 2*a)*pi*cos(pi*s)*(1 + cos(pi*s)))/2. - (a*(1 + 2*a)*pi*sin(pi*s)**2)/2. ;

    dxds = xnum_ds/den - xnum*den_ds/den**2 ;
    dyds = ynum_ds/den - ynum*den_ds/den**2 ;
    
    return dxds, dyds

#-----------------------------------
def Joukowski(nn, Q, ds0 = -0.2, ds1 = -0.2):
    # hardcoded analytical function
    
    X = npy.zeros([2*nn*Q,2])
    a = 0.1

    # The Joukowski airfoil is already defined in a cosine parametric space,
    # so linspace is correct here, not cos(linspace).
    #s = 1-npy.linspace(0,1,nn+1)
    #print nn, s

    #Use a cos curve to cluster at LE and TE. def Joukowski_wake_x must use the same function.
    #s = Cos(nn)
    
    #Use a Bezier curve to cluster at LE and TE. def Joukowski_wake_x must use the same function.
    s = Bezier(nn, ds0=ds0, ds1=ds1)
    
    #print nn, s
    sL = spaceqarc(s, a, Q)
    #print sL;
    sU = sL[::-1]

    xL, yL = Joukowski_xy(sL,a)
    xU, yU = Joukowski_xy(sU,a)
    yL = -yL
    #print xL;

    s = npy.append(sL,-sU[1:])
    
    X[:,0] = npy.append(xL,xU[1:-1])
    X[:,1] = npy.append(yL,yU[1:-1])

    return X, sL

#===============================================================================
def spaceq(re, Q):
    nsub = Q
    nre = len(re) - 1
    nr  = nsub*nre
    r = npy.zeros(nr+1)
    for i in range(nre):
        for j in range(nsub):
            f = j/nsub
            r[i*nsub+j] = re[i]*(1.0-f) + re[i+1]*f
    r[nr] = re[nre]
    
    return r

#===============================================================================
def spaceqarc(se, a, Q):
    
    def arc(s):
        dxds, dyds = Joukowski_dxy_ds(s,a)
        return npy.sqrt( dxds**2 + dyds**2 )

    nsub = Q
    ns = len(se) - 1
    nr  = nsub*ns
    s = npy.zeros(nr+1)
    for i in range(ns):
        
        arclength = integrate.quad( arc, se[i], se[i+1] )[0]
        
        s[i*nsub] = se[i]
        for j in range(1,nsub):
            f = j/float(nsub)
            
            s[i*nsub+j] = optimize.bisect(lambda t:integrate.quad( arc, se[i], t )[0]-arclength*f, se[i] + 1e-8*arclength, se[i+1]-1e-8*arclength)

    s[nr] = se[ns]
    
    return s


#-----------------------------------
def meshplot(X, Y, edgecolor='k'):
    """Plot a mapped Cartesian grid."""
    import matplotlib.pyplot as plt
    plt.clf()
    plt.axis('equal')
    plt.pcolor(X, Y, 0*X, edgecolor=edgecolor, cmap='Greens')
    plt.show()
    plt.draw()
    
if __name__ == '__main__':
    #nnormal = 16
    #reynolds = 1e3
    #maxref = 6
    #dy_te = 0.1 / reynolds**0.5 / 2**maxref
    #print nnormal*2**maxref, dy_te
    
    Q = 1
    X, Y = make_joukowski_challenge(3, 1, 1.e6)
    meshplot(X, Y)

    #for ref in range(0,1):
    #    make_joukowski_challenge(ref, Q, reynolds=1.e6)
    #    print("Done with level " + str(ref));
#     import pylab as pyl
#     X, sL = Joukowski(500, Q)
#     pyl.plot(X[:,0],X[:,1],'-o')
#     pyl.axis( [0,1,-0.5,0.5] )
#     pyl.show()
