from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

# Top half C-mesh for Joukowski airfoil
# Based on conformal mappings

#-----------------------------------
def Distance(i, dx_te, ratio):
    return dx_te*(ratio**i - 1) / (ratio-1)

def GradDist(i, dx_te, ratio):
    return dx_te*((i-1)*ratio**i - i*ratio**(i-1) + 1) / (ratio-1)**2


def FindStretching(n, h_min, Hc):
    # Find the ratio of successive cell sizes to get a total length of Hc from
    # n cells, starting with the first cell size = h_min.
    
    # This guess is exact for very large size ratios
    guess = (Hc / h_min)**(1./(n-1))
    finished = False
    while not finished:
        func = Distance(n, h_min, guess) - Hc
        grad = GradDist(n, h_min, guess)
        delta = - func / grad
        guess += delta

        if (abs(delta) < 1.e-12):
            finished = True

    return guess;

#===============================================================================
def spaceq(re, Q):
    nsub = Q
    nre = len(re) - 1
    nr  = nsub*nre
    r = np.zeros(nr+1)
    for i in range(nre):
        for j in range(nsub):
            f = j/nsub
            r[i*nsub+j] = re[i]*(1.0-f) + re[i+1]*f
    r[nr] = re[nre]
    
    return r

def coarsen(re, ref, maxref):
    i = maxref
    while i > ref:
        re = np.delete(re, np.s_[1::2], 0 )
        i = i-1
    return re

#-----------------------------------
def Bezier(nn, smax=1, ds0=0.2, dds0=0, ds1=0.2, dds1=0):

    s = np.linspace(0,smax,nn+1)
    
    #Use a Bezier curve to cluster at LE and TE: ds = 1 gives a linear distribution. Clustering is added as ds->0 from 1
    #ds0 = -0.2
    #ds1 = -0.2
    P0 = 0
    P1 = ds0/5
    P2 = (dds1 + 8*ds0)/20.
    P3 = (20 + dds0 - 8*ds1)/20.
    P4 = (5 - ds1)/5.
    P5 = 1
    t = P0*(1 - s)**5 + P1*5*s*(1-s)**4 + P2*10*s**2*(1-s)**3 + P3*10*s**3*(1-s)**2+P4*5*s**4*(1-s) + P5*s**5
    return t

#-----------------------------------
def meshplot(X, Y, edgecolor='k'):
    """Plot a mapped Cartesian grid."""
    plt.clf()
    plt.axis('equal')
    plt.pcolor(X, Y, 0*X, edgecolor=edgecolor, cmap='Greens')
    plt.show()
    plt.draw()

def joukowski_conformal(S, T, joux=0.1):
    """Conformal mapping (S,T) -> (X,Y) for Joukowski mesh.

    joux:   Joukowski x-shift
    """

    # Special case for (0,0)
    zero_ix = np.logical_and(S == 0.0, T == 0.0).ravel().nonzero()

    # Map to circle
    z = S + 1j * T
    w = z**2
    w.ravel()[zero_ix] = 1.0
    A = (w - 1) / w
    B = np.sqrt(A)
    c = (1 + B) / (1 - B)
    c.ravel()[zero_ix] = -1.0

    # Map to Joukowski
    r = joux + 1
    cs = c*r - complex(joux, 0.0)
    g = cs + 1.0 / cs

    # Scale airfoil to x = [0,1]
    endpnts = np.array([-1, 1]) * r - joux
    left_right = endpnts + 1.0 / endpnts
    g = (g - left_right[0]) / np.diff(left_right)

    # Return physical and parameter meshes
    return np.real(g), np.imag(g)

def joukowski_inverse(X, Y, joux=0.1):
    """Inverse of joukowski_conformal (X,Y) -> (S,T).

    joux:   Joukowski x-shift
    """
    # Maple auto-generated
    zz = X + 1j * Y
    t1 = (joux ** 2)
    t2 = 2 * t1
    t3 = (zz ** 2)
    t4 = t1 * t3
    t5 = 2 * t4
    t7 = 2 * joux * t3
    t8 = t1 * zz
    t9 = 4 * t8
    t10 = zz * joux
    t11 = 2 * t10
    t12 = t1 * joux
    t15 = t1 ** 2
    t20 = t3 ** 2
    t24 = t3 * zz
    t35 = -2 * t12 * zz - t8 + t15 + 6 * t3 * t15 - 4 * zz * t15 + t15 * t20 + 2 * t12 * t20 - 4 * t15 * t24 + t1 * t20 + 6 * t12 * t3 - 6 * t12 * t24 - 3 * t1 * t24 + 3 * t4
    t36 = np.sqrt(t35)
    t37 = 0.2e1 * t36
    t43 = 1 / (-4 * t1 + 8 * t8 + 4 * t10 + 1)
    t45 = np.sqrt(t43 * (-t2 + t5 + t7 + t9 + t11 + zz + t37))
    t48 = np.sqrt(-t43 * (t2 - t5 - t7 - t9 - t11 - zz + t37))
    out = np.array([t45,-t45,t48,-t48])

    # Test closest match between positive roots
    map1 = joukowski_conformal(np.real(out[0]), np.imag(out[0]), joux)
    map2 = joukowski_conformal(np.real(out[2]), np.imag(out[2]), joux)

    d1 = (map1[0] - X)**2 + (map1[1] - Y)**2
    d2 = (map2[0] - X)**2 + (map2[1] - Y)**2

    out1 = out[0]
    ix = (d2 < d1).ravel().nonzero()
    out1.ravel()[ix] = out[2].ravel()[ix]
    
    return np.real(out1), np.imag(out1)

def joukowski_parameter(ref, Q, reynolds, growth=1.3, R=100, joux=0.1):
    """Make parameter space mesh (S,T) for Joukowski mapping.

    nchord: Number of streamwise points along the chord (cos-distributed)
    growth: Element size growth ratio in wake + normal directions
    R:      Farfield distance (at least)
    """
    
    refmax = 6
    assert(ref <= refmax)
    
    nchord=8*2**refmax           # number of elements along one side of the airfoil geometry
    nxwake=16*2**refmax           # x-wake on centerline
    nnormal=16*2**refmax         # points normal to airfoil surface
    
    # Trailing edge spacing
    if (reynolds > 5e5):
        # Turbulent. 
        AR = 0.5
        ds0 = 1.0
        dds0 = 0.0
        ds1 = 0.2
        dds1 = 0.
    else:
        # Laminar.  
        AR = 1
        ds0 = 2.5
        dds0 = 0.0
        ds1 = 0.175
        dds1 = 2.0

    # Chord distribution
    #phi = np.linspace(np.pi, 0.0, nchord+1)
    #sAf = (np.cos(phi) + 1) / 2
    phi = Bezier(nchord,ds0=ds0,dds0=dds0,ds1=ds1,dds1=dds1)
    sAf = (1 - np.cos(np.pi*phi)) / 2

    sAf_half = 1-sAf[:int(nchord/2)-1:-1]
    
    ds = sAf_half[-1] - sAf_half[-2]
    
    sx = np.zeros(nchord+nxwake+1)
    sx[0:nchord+1] = sAf
    sx[nchord:nchord+len(sAf_half)] = 1+sAf_half
    nWake = nxwake+1-len(sAf_half)
    ratio = FindStretching(nWake, ds, np.sqrt((R + 1+sAf_half[-1]))-(1+sAf_half[-1]))
    for i in range(1,nWake+1):
        sx[nchord+len(sAf_half)+i-1] = 1+sAf_half[-1] + Distance(i, ds, ratio)

    sx = coarsen(sx, ref, refmax)
    sx = spaceq(sx, Q)

    # Wake distribution
    #sx = sAf
    #sx = np.append(sx, 1.0 + sAf_half[1:])
    #while sx[-1]**2 < (R + 1)/1.5:
    #    sx = np.append(sx, sx[-1] + growth * (sx[-1] - sx[-2]))
    
    
#    sy = np.zeros(nnormal+1)
#    sy[0:len(sAf_half)] = sAf_half/AR
#    sy_Af = sy[len(sAf_half)-1]
#    ds = sy[len(sAf_half)-1] - sy[len(sAf_half)-2]
#    nNormal = nnormal+1-len(sAf_half)
#    ratio = FindStretching(nNormal, ds, np.sqrt(R)-sy_Af)
#    for i in range(nNormal+1):
#        sy[len(sAf_half)+i-1] = sy_Af + Distance(i, ds, ratio)

    sy = np.zeros(nnormal+1)
    ds = (sAf_half[1] - sAf_half[0])/AR
    ratio = FindStretching(nnormal, ds, np.sqrt(R))
    for i in range(1,nnormal+1):
        sy[i] = Distance(i, ds, ratio)


    sy = coarsen(sy, ref, refmax)
    sy = spaceq(sy, Q)
    
    # Normal distribution
    #sy = sAf_half.copy()
    #growth_normal = 1.0 + (growth - 1.0) / 2.0  # empirical
    #while sy[-1]**2 < R/1.5:
    #    sy = np.append(sy, sy[-1] + growth_normal * (sy[-1] - sy[-2]))


    lx0 = sx / sx.max()
    ly0 = sy / sy.max()
    lx = -2 * lx0**3 + 3 * lx0**2
    ly = -2 * ly0**3 + 3 * ly0**2

    # Bottom and left
    bottom = [sx, 0*sx]
    left = [0*sy, sy]
    
    #left[1] = np.zeros(nnormal+1)
    #left[1][0:len(sAf_half)] = sAf_half
    #nNormal = nnormal+1-len(sAf_half)
    #ratio = FindStretching(nNormal, ds, R-sAf_half[-1])
    #for i in range(nNormal+1):
    #    left[1][len(sAf_half)+i-1] = sAf_half[-1] + Distance(i, ds, ratio)

    #left[1] = coarsen(left[1], ref, refmax)
    #sy2 = left[1] = spaceq(left[1], Q)
    
    #left[1] = joukowski_inverse(left[1], left[0], joux)[0]

    # Find parameters for straight vertical outflow boundary
    xright = joukowski_conformal(sx[-1], 0.0, joux)[0]
    yright = ly0 * -joukowski_conformal(0.0*sy[-1], sy[-1], joux)[0]
    right = joukowski_inverse(xright + 0*sy, yright, joux)
    right_eps = joukowski_inverse(xright, yright[-1] - 1e-5, joux)

    # Top boundary

    # Straight
    #top = [np.linspace(left[0][-1], right[0][-1], sx.shape[0]),
    #       np.linspace(left[1][-1], right[1][-1], sx.shape[0])]
    # Hermite
    lxtop = lx.copy()
    # Some smoothing, but smoothing removes grid nesting
    #for i in range(1000): # empirical
    #    lxtop[1:-1] = (lxtop[0:-2] + lxtop[2:]) / 2.0
    lxtop = np.linspace(0, lxtop[-1], lxtop.shape[0]) # straight
    lxtop1 = -2 * lxtop**3 + 3 * lxtop**2
    # Make orthogonal to right boundary
    #slope = (right[0][-1] - right[0][-2]) / (right[1][-1] - right[1][-2])
    slope = (right[0][-1] - right_eps[0]) / (right[1][-1] - right_eps[1])
    newslope = slope * right[0][-1]
    lxtop2 = lxtop**3 - lxtop**2
    top = [lxtop * right[0][-1], \
           left[1][-1]*(1-lxtop1) + right[1][-1]*lxtop1 - newslope*lxtop2]
    
    if False:  #debugging
        plt.clf()
        plt.plot(bottom[0], bottom[1], '.-',label="bot")
        plt.plot(top[0], top[1], '.-',label="top")
        plt.plot(left[0], left[1], '.-',label="left")
        plt.plot(right[0], right[1], '.-',label="right")
        plt.axis('equal')
        plt.legend()
        plt.draw()
        plt.show()

    # TFI mapping
    X1 = np.outer(1-lx, left[0]) + np.outer(lx, right[0])
    Y1 = np.outer(1-lx, left[1]) + np.outer(lx, right[1])

    X2 = np.outer(bottom[0], 1-ly) + np.outer(top[0], ly)
    Y2 = np.outer(bottom[1], 1-ly) + np.outer(top[1], ly)

    X12 = np.outer(1-lx, 1-ly) * X1[0, 0] + np.outer(lx, 1-ly) * X1[-1, 0] + \
          np.outer(1-lx,   ly) * X1[0,-1] + np.outer(lx,   ly) * X1[-1,-1]
    Y12 = np.outer(1-lx, 1-ly) * Y1[0, 0] + np.outer(lx, 1-ly) * Y1[-1, 0] + \
          np.outer(1-lx,   ly) * Y1[0,-1] + np.outer(lx,   ly) * Y1[-1,-1]

    X = X1 + X2 - X12
    Y = Y1 + Y2 - Y12

   # meshplot(X, Y)


    # Find parameters for straight vertical outflow boundary with uniform y-distribution
    #sy = np.linspace(0, sy[-1], len(sy))
    #sy = Bezier(len(sy)-1,ds0=0.025,dds0=1.0,ds1=2.5,dds1=0.0)*sy[-1]
    ds = sy[-1]/len(sy)/(10)
    ratio = FindStretching(len(sy)-1, ds, sy[-1])
    for i in range(1,len(sy)):
        sy[i] = Distance(i, ds, ratio)

    ly0 = sy / sy.max()

    yright = ly0 * -joukowski_conformal(0.0*sy[-1], sy[-1], joux)[0]
    right = joukowski_inverse(xright + 0*sy, yright, joux)
    
    s = np.zeros(nchord+int(nxwake/2)+1)
    s = coarsen(s, ref, refmax)
    s = spaceq(s, Q)
    
    noffset = len(s)-1

    left = (X[noffset,:], Y[noffset,:])
    top = (X[noffset:,-1], Y[noffset:,-1])
    bottom = (X[noffset:,0], Y[noffset:,0])

    if False:  #debugging
        plt.clf()
        plt.plot(bottom[0], bottom[1], '.-',label="bot")
        plt.plot(top[0], top[1], '.-',label="top")
        plt.plot(left[0], left[1], '.-',label="left")
        plt.plot(right[0], right[1], '.-',label="right")
        plt.axis('equal')
        plt.legend()
        plt.draw()
        plt.show()

    lx = lx[noffset:]
    lx = (lx - lx.min())
    lx /= lx.max()

    # TFI mapping
    X1 = np.outer(1-lx, left[0]) + np.outer(lx, right[0])
    Y1 = np.outer(1-lx, left[1]) + np.outer(lx, right[1])

    X2 = np.outer(bottom[0], 1-ly) + np.outer(top[0], ly)
    Y2 = np.outer(bottom[1], 1-ly) + np.outer(top[1], ly)
 
    X12 = np.outer(1-lx, 1-ly) * X1[0, 0] + np.outer(lx, 1-ly) * X1[-1, 0] + \
          np.outer(1-lx,   ly) * X1[0,-1] + np.outer(lx,   ly) * X1[-1,-1]
    Y12 = np.outer(1-lx, 1-ly) * Y1[0, 0] + np.outer(lx, 1-ly) * Y1[-1, 0] + \
          np.outer(1-lx,   ly) * Y1[0,-1] + np.outer(lx,   ly) * Y1[-1,-1]

    Xwake = X1 + X2 - X12
    Ywake = Y1 + Y2 - Y12

    #meshplot(Xwake, Ywake)

    X[len(s)-1:,:] = Xwake
    Y[len(s)-1:,:] = Ywake
    
    
    #meshplot(X, Y)

    return X, Y

def make_joukowski_classic(ref, Q, reynolds=1.e6):
    S, T = joukowski_parameter(ref, Q, reynolds)
    X, Y = joukowski_conformal(S, T)
    
    X = np.concatenate(( np.flipud(np.delete(X, 0, axis=0)), X), axis=0)
    Y = np.concatenate((-np.flipud(np.delete(Y, 0, axis=0)), Y), axis=0)

    return X, Y

if __name__ == "__main__":
    X, Y = make_joukowski_classic(2, 1, 1.e6)
    meshplot(X, Y)
