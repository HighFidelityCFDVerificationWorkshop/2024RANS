from __future__ import print_function
import numpy as npy

#-----------------------------------------------------------
# writes a Pointwise segment geometry files
def writeSegment(X, Y, nWK):
    
    # Airfoil
    Afx = X[nWK-1:-nWK+1,0]
    Afy = Y[nWK-1:-nWK+1,0]
    nAf = len(Afx)

    # Inflow
    Inx = X[-1::-1,-1]
    Iny = Y[-1::-1,-1]
    nIn = len(Inx)

    # Outflow Upper
    Outx = X[0,-1:0:-1]
    Outy = Y[0,-1:0:-1]

    # Outflow Lower
    Outx = npy.append(Outx, X[-1,:] )
    Outy = npy.append(Outy, Y[-1,:])
    nOut = len(Outx)

    filename = 'joukowski.dat'
    Af = open(filename, 'w')
    print('Writing ', filename)

    #----------#
    # Vertices #
    #----------#
    floatformat = "{:3.16e}"
    
    Af.write(str(nAf)+"\n")
    for i in range(nAf):
        Af.write(floatformat.format(Afx[i]) + ' ' + floatformat.format(Afy[i]) + ' 0' + '\n')

    Af.write(str(nIn)+"\n")
    for i in range(nIn):
        Af.write(floatformat.format(Inx[i]) + ' ' + floatformat.format(Iny[i]) + ' 0' + '\n')

    Af.write(str(nOut)+"\n")
    for i in range(nOut):
        Af.write(floatformat.format(Outx[i]) + ' ' + floatformat.format(Outy[i]) + ' 0' + '\n')
    Af.close()
