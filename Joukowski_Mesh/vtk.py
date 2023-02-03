from __future__ import division, print_function

def writeVTK(filename_base, ref, Q, E, V):
    #=========================#
    # Write out the grid file #
    #=========================#

    filename = filename_base + '_ref'+str(ref)+ '_Q'+str(Q)+'.vtk'
    print('Writing ', filename)
    f = open(filename, 'w')

    f.write('# vtk DataFile Version 2\n');
    f.write(filename_base + ', level ' + str(ref) + ' order ' + str(Q) + '\n');
    f.write('ASCII\n\n');
    f.write('DATASET UNSTRUCTURED_GRID\n');
    
    nelem = E.shape[0];
    nnode = V.shape[0];

    #----------#
    # Vertices #
    #----------#
    f.write('POINTS ' + str(nnode) + ' float\n');
    floatformat = "{:3.16e}"
    for i in range(nnode):
        f.write(floatformat.format(V[i,0]) + ' ' +
                floatformat.format(V[i,1]) + ' 0\n')
      
    #----------#
    # Elements #
    #----------#
    # Write as linear quads
    
    f.write('CELLS '+ str(nelem)+ ' ' + str(5*nelem) + '\n');
    for e in range(nelem):
        f.write('4   ');
        f.write(str(E[e,1-1]-1)+' ');
        f.write(str(E[e,Q+1-1]-1)+' ');
        f.write(str(E[e,(Q+1)*(Q+1)-1]-1)+' ');
        f.write(str(E[e,Q*(Q+1)+1-1]-1)+' ');
        f.write('\n');

    #------------------------------------------------#
    # Element types: Linear tri = 5; linear quad = 9 #
    #------------------------------------------------#
    f.write('CELL_TYPES ' + str(nelem) + '\n');
    for e in range(nelem):
        f.write('9\n');
                
    f.close()
    return
