from __future__ import division, print_function
import numpy as npy

#-----------------------------------------------------------
# writes a 2D ascii ugrid grid file
# http://www.simcenter.msstate.edu/software/downloads/doc/ug_io/2d_grid_file_type_ugrid.html
def writeUGRID2D(filename_base, ref, Q, TriFlag, E, V, nLE, NC, nWK, nWB, nr):
    #=========================#
    # Write out the grid file #
    #=========================#

    filename = filename_base + '_ref'+str(ref)+ '_Q'+str(Q)+'.ugrid'
    print('Writing ', filename)
    f = open(filename, 'w')
    
    nnode = V.shape[0];
    ntri = 0;
    nquad = 0;
    ntet = 0;
    npyr = 0;
    npri = 0;
    nhex = 0;
    nline = 0;

    
    nelem = E.shape[0];
    
    if TriFlag: ntri  = 2*nelem 
    else:       nquad = nelem
    
    f.write(str(nnode) + ' ' +
            str(ntri) + ' ' +
            str(nquad) + ' ' +
            str(ntet) + ' ' +
            str(npyr) + ' ' +
            str(npri) + ' ' +
            str(nhex) + '\n')
     
    #----------#
    # Vertices #
    #----------#
    floatformat = "{:3.16e}"
    
    for i in range(nnode):
        f.write(floatformat.format(V[i,0]) + ' ' + floatformat.format(V[i,1]) + ' 0.0\n')

    #----------#
    # Elements #
    #----------#
      
    if TriFlag:
        ni = int((NC.shape[0]-1))
        nj = int((NC.shape[1]-1))
        
        nodemap = (0,  1, 
                   2, -1)
        nodemap2= (0, 1, 
                  -1, 2)

        #Invert the map
        nodemapinv  = []
        for k in range(3):
            j = 0
            while nodemap[j] != k: j += 1
            nodemapinv.append(j)
    
        nodemapinv2  = []
        for k in range(3):
            j = 0
            while nodemap2[j] != k: j += 1
            nodemapinv2.append(j)

        for j in range(nj):
            for i in range(int(ni/2)):
                e = i + ni*j
                
                #Write nodes
                for k in range(3):
                    f.write(str(E[e,nodemapinv[k]])+' ')
                f.write('\n')
        
                #Write nodes
                for k in range(3):
                    f.write(str(E[e,4-1-nodemapinv[k]])+' ')
                f.write('\n')
                
            for i in range(int(ni/2),ni):
                e = i + ni*j
                
                #Write nodes
                for k in range(3):
                    f.write(str(E[e,nodemapinv2[k]])+' ')
                f.write('\n')
                        
                #Write nodes
                for k in range(3):
                    f.write(str(E[e,4-1-nodemapinv2[k]])+' ')
                f.write('\n')
                
    else:
        nodemap = (0,  1, 
                   3,  2)

        for e in range(nelem):
            for k in range(4):
                f.write(str(E[e,nodemap[k]])+' ')
            f.write('\n')


      # write dummy markers for tri+quad
    for e in range(ntri+nquad):
        f.write('0\n')

    #----------------#
    # Boundary faces #
    #----------------#
     
    # write the number of lines
    f.write(str((nLE-1)+(nWB-1)+(2*(nr-1))) + '\n')
        
    # Airfoil
    for i in range(int((nLE-1))):
        f.write(str(NC[nWK-1+i,0]) + ' ' + str(NC[nWK-1+(i+1),0]) + ' 1 \n')
    
      
    # Farfield inflow
    for i in range(int((nWB-1))):
        f.write(str(NC[i,nr-1]) + ' ' + str(NC[(i+1),nr-1]) + ' 2 \n')
      
    # Farfield Outflow
    for i in range(nr-1):
        f.write(str(NC[0,i]) + ' ' + str(NC[0,(i+1)]) + ' 3\n')

    for i in range(nr-1):
        f.write(str(NC[nWB-1,i]) + ' ' + str(NC[nWB-1,(i+1)]) + ' 3 \n')
    
    f.close()
    
    
    filename = filename_base + '_ref'+str(ref)+ '_Q'+str(Q)+'.mapbc'
    print('Writing ', filename)
    f = open(filename, 'w')
    f.write('1 4000 Airfoil')
    f.write('2 5050 Farfield')
    f.write('3 5050 Farfield')
    f.close()
