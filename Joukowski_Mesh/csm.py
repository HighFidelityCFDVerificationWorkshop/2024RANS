from __future__ import print_function
import numpy as npy

#-----------------------------------------------------------
# writes a csm geometry files
def writeCSM(X, Y, nWK):

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

    filename = 'joukowski.csm'
    f = open(filename, 'w')
    print('Writing ', filename)

    #----------#
    # Vertices #
    #----------#
    floatformat = "{:3.16e}"

    f.write("#Joukowski Airfoil\n")
    f.write("\n")
    f.write("#Inflow\n")
    f.write("SKBEG " + floatformat.format(Inx[0]) + ' ' + floatformat.format(Iny[0]) + ' 0' + '\n')
    for i in range(1,nIn):
        f.write('SPLINE ' + floatformat.format(Inx[i]) + ' ' + floatformat.format(Iny[i]) + ' 0' + '\n')
    f.write("SKEND\n")
    f.write("\n")
    f.write("#Outflow\n")
    f.write("SKBEG " + floatformat.format(Outx[0]) + ' ' + floatformat.format(Outy[0]) + ' 0' + '\n')
    for i in range(1,nOut):
        f.write('SPLINE ' + floatformat.format(Outx[i]) + ' ' + floatformat.format(Outy[i]) + ' 0' + '\n')
    f.write("SKEND\n")
    f.write("JOIN\n")
    f.write("COMBINE\n")
    f.write("\n")
    f.write("#Airfoil\n")
    f.write("SKBEG " + floatformat.format(Afx[0]) + ' ' + floatformat.format(Afy[0]) + ' 0' + '\n')
    for i in range(1,nAf):
        f.write('SPLINE ' + floatformat.format(Afx[i]) + ' ' + floatformat.format(Afy[i]) + ' 0' + '\n')
    f.write("SKEND\n")
    f.write("BOX -0.1 0 -0.1  0.2 0 0.2\n")
    f.write("SUBTRACT\n")
    f.write("SUBTRACT\n")
    f.write("SELECT edge 1\n")
    f.write("SELECT add 2\n")
    f.write("   ATTRIBUTE .nPos 15\n")
    #f.write("SELECT edge 3\n")
    #f.write("SELECT add 4\n")
    #f.write("   ATTRIBUTE .nPos " + str(1) + "\n")
    f.write("DUMP joukowski.egads\n")    
    f.write("END\n")

    f.close()
