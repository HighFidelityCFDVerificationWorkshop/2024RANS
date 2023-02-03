from __future__ import division, print_function

import numpy as np
from h5py import File

def _cst_size_str(s, n):
    return np.bytes_(bytes(s, "ascii").ljust(n, b"\x00"))

def create_group(parent, group_name, label, name, typ, flags=True):
    parent.create_group(group_name)
    grp = parent[group_name]
    grp.attrs['label'] = _cst_size_str(label, 33)
    grp.attrs['name'] = _cst_size_str(name, 33)
    grp.attrs['type'] = _cst_size_str(typ, 3)
    if flags:
        grp.attrs['flags'] = np.array([1], dtype=np.int32)
    return grp

def make_baseline_file(name):
    f = File(name, 'w')

    f.attrs['label'] = _cst_size_str('Root Node of HDF5 File', 33)
    f.attrs['name'] = _cst_size_str('HDF5 MotherNode', 33)
    f.attrs['type'] = _cst_size_str('MT', 3)

    # header information needed for CGNS readers
    # format is 'IEEE_LITTLE_32\x00' using integers for ASCII characters
    f.create_dataset(' format', shape=(15,), dtype='|i1',
                     data=[73, 69, 69, 69, 95, 76, 73, 84, 84, 76, 69, 95, 51, 50, 0])
    # hdf5version is 'HDF5 Version 1.8.16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    f.create_dataset(' hdf5version', shape=(33,), dtype='|i1',
                     data=[72, 68, 70, 53, 32, 86, 101, 114, 115, 105, 111, 110, 32, 49, 46, 56,
                           46, 49, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # create the CGNS version group
    vers = create_group(f, 'CGNSLibraryVersion', 'CGNSLibraryVersion_t', 'CGNSLibraryVersion', 'R4')
    vers.create_dataset(' data', shape=(1,), dtype='<f4', data=3.4)

    # create the base group
    b = create_group(f, 'Base', 'CGNSBase_t', 'Base', 'I4')
    b.create_dataset(' data', shape=(2,), dtype='<i4', data=[2, 2])

    # create the zone group
    z = create_group(b, 'Zone', 'Zone_t', 'Zone', 'I4')
    # this will be filled with (num points, num elements, 0)
    z.create_dataset(' data', shape=(3,1), dtype='<i4', data=[0, 0, 0])

    # create the zone type group
    zt = create_group(z, 'ZoneType', 'ZoneType_t', 'ZoneType', 'C1')
    # ZoneType is 'Unstructured'
    zt.create_dataset(' data', shape=(12,), dtype='|i1',
                      data=[85, 110, 115, 116, 114, 117, 99, 116, 117, 114, 101, 100])

    # Create skeleton for grid coordinates, to be filled in later
    gc = create_group(z, 'GridCoordinates', 'GridCoordinates_t', 'GridCoordinates', 'MT')
    create_group(gc, 'CoordinateX', 'DataArray_t', 'CoordinateX', 'R8')
    create_group(gc, 'CoordinateY', 'DataArray_t', 'CoordinateY', 'R8')

    # Create ZoneBC group to house BC info
    bc = create_group(z, 'ZoneBC', 'ZoneBC_t', 'ZoneBC', 'MT')

    # Create farfield inflow/outflow BC
    for inout in ('inflow', 'outflow'):
        far = create_group(bc, f'farfield_{inout}', 'BC_t', f'farfield_{inout}', 'C1')
        # set data to BCFarfield
        far.create_dataset(' data', shape=(10,), dtype='|i1',
                           data=[66, 67, 70, 97, 114, 102, 105, 101, 108, 100])
        pr = create_group(far, 'PointRange', 'IndexRange_t', 'PointRange', 'I4')
        pr.create_dataset(' data', shape=(2, 1), dtype='<i4')
        # set grid location to EdgeCenter, which means that PointRange is edge element range
        gl = create_group(far, 'GridLocation', 'GridLocation_t', 'GridLocation', 'C1')
        gl.create_dataset(' data', shape=(10,), dtype='|i1',
                          data=[69, 100, 103, 101, 67, 101, 110, 116, 101, 114])


    # Create wall BC
    wall = create_group(bc, 'wall', 'BC_t', 'wall', 'C1')
    # set data BCWallViscous
    wall.create_dataset(' data', shape=(13,), dtype='|i1',
                        data=[66, 67, 87, 97, 108, 108, 86, 105, 115, 99, 111, 117, 115])
    pr = create_group(wall, 'PointRange', 'IndexRange_t', 'PointRange', 'I4')
    pr.create_dataset(' data', shape=(2, 1), dtype='<i4')
    # set grid location to EdgeCenter, which means that PointRange is edge element range
    gl = create_group(wall, 'GridLocation', 'GridLocation_t', 'GridLocation', 'C1')
    gl.create_dataset(' data', shape=(10,), dtype='|i1',
                      data=[69, 100, 103, 101, 67, 101, 110, 116, 101, 114])

    return f


def writeCGNS(filename_base, x, y, q, ref, tri, dist):
    f = make_baseline_file(filename_base + '_ref'+str(ref)+ '_Q'+str(q)+'.cgns')

    # get some handy references
    Z = f['Base']['Zone']
    GCX = Z['GridCoordinates']['CoordinateX']
    GCY = Z['GridCoordinates']['CoordinateY']
    BC_in = Z['ZoneBC']['farfield_inflow']['PointRange'][' data']
    BC_out = Z['ZoneBC']['farfield_outflow']['PointRange'][' data']
    BC_wall = Z['ZoneBC']['wall']['PointRange'][' data']

    # constants
    if dist == "Classic":
        nWake = 16*q*2**ref+1
    elif dist == "Challenge":
        nWake = 8*q*2**ref+1
    nWB = x.shape[0]
    nr  = x.shape[1]

    # flatten arrays, transpose to fix normals, remove duplicates and place data into file
    x = x.T.flatten()
    y = y.T.flatten()
    nNodes = len(x) - nWake
    x = np.hstack((x[:nWB-nWake], x[nWB:]))
    y = np.hstack((y[:nWB-nWake], y[nWB:]))
    GCX.create_dataset(' data', dtype='<f8', data=x)
    GCY.create_dataset(' data', dtype='<f8', data=y)

    # get the proper node numbering accounting for the duplicate wake nodes
    nodes = np.arange(1, nNodes+nWake+1)
    nodes[nWB-nWake:nWB] = nodes[:nWake][::-1]
    nodes[nodes > nWB] -= nWake

    # make the field cells
    if tri:
        field, nElems_field = write_tri_field(Z, q, nodes, nWB, nr)
    else:
        field, nElems_field = write_quad_field(Z, q, nodes, nWB, nr)

    # make boundary elements and set boundary condition ranges
    far_in, nElems_far_in = write_bar_boundary(Z, q, nodes[-nWB:], f'farfield_inflow_bar_q{q}',
                                               nElems_field+1)
    BC_in[0] = nElems_field+1
    BC_in[1] = nElems_field+nElems_far_in
    outNodes = [nodes[-(i*nWB+1)] for i in range(nr)]
    outNodes.extend([nodes[i*nWB] for i in range(1,nr)])
    far_out, nElems_far_out = write_bar_boundary(Z, q, outNodes, f'farfield_outflow_bar_q{q}',
                                                 nElems_field+nElems_far_in+1)
    BC_out[0] = nElems_field+nElems_far_in+1
    BC_out[1] = nElems_field+nElems_far_in+nElems_far_out
    wall, nElems_wall = write_bar_boundary(Z, q, nodes[nWake-1:nWB-nWake+1], f'wall_bar_q{q}',
                                           nElems_field+nElems_far_in+nElems_far_out+1)
    BC_wall[0] = nElems_field+nElems_far_in+nElems_far_out+1
    BC_wall[1] = nElems_field+nElems_far_in+nElems_far_out+nElems_wall

    Z[' data'][0] = nNodes
    Z[' data'][1] = nElems_field

    f.close()


def write_bar_boundary(zone, q, nodes, name, elem_start):
    elem_type = {1:3,   # Q1 refers to BAR_2
                 2:4,   # Q2 refers to BAR_3
                 3:24,  # Q3 refers to BAR_4
                 4:40}  # Q4 refers to BAR_5
    elem_map = {1:lambda n: (nodes[n], nodes[n+1]),
                2:lambda n: (nodes[n], nodes[n+2], nodes[n+1]),
                3:lambda n: (nodes[n], nodes[n+3], nodes[n+1], nodes[n+2]),
                4:lambda n: (nodes[n], nodes[n+4], nodes[n+1], nodes[n+2], nodes[n+3])}
    nElemNodes = q + 1
    nElems = (len(nodes) - 1) // q

    bar = create_group(zone, name, 'Elements_t', name, 'I4')
    bar.create_dataset(' data', dtype='<i4', data=[elem_type[q], 0])

    # create element range group
    er = create_group(bar, 'ElementRange', 'IndexRange_t', 'ElementRange', 'I4')

    # create element connectivity group
    ec = create_group(bar, 'ElementConnectivity', 'DataArray_t', 'ElementConnectivity', 'I4')

    er.create_dataset(' data', dtype='<i4', data=[elem_start, elem_start+nElems-1])
    ec.create_dataset(' data', shape=(nElems*nElemNodes,), dtype='<i4')
    ec = ec[' data']
    loc = 0
    for i in range(0, len(nodes)-1, q):
        ec[loc:loc+nElemNodes] = elem_map[q](i)
        loc += nElemNodes

    return bar, nElems


def write_quad_field(zone, q, nodes, nWB, nr):
    elem_type = {1:7,   # Q1 refers to QUAD_4
                 2:9,   # Q2 refers to QUAD_9
                 3:28,  # Q3 refers to QUAD_16
                 4:44}  # Q4 refers to QUAD_25
    elem_map = {1:lambda n: (nodes[n], nodes[n+1], nodes[n+nWB+1], nodes[n+nWB]),
                2:lambda n: (nodes[n], nodes[n+2], nodes[n+2*nWB+2], nodes[n+2*nWB],
                             nodes[n+1], nodes[n+nWB+2], nodes[n+2*nWB+1], nodes[n+nWB],
                             nodes[n+nWB+1]),
                3:lambda n: (nodes[n], nodes[n+3], nodes[n+3*nWB+3], nodes[n+3*nWB],
                             nodes[n+1], nodes[n+2], nodes[n+nWB+3], nodes[n+2*nWB+3],
                             nodes[n+3*nWB+2], nodes[n+3*nWB+1], nodes[n+2*nWB], nodes[n+nWB],
                             nodes[n+nWB+1], nodes[n+nWB+2], nodes[n+2*nWB+2], nodes[n+2*nWB+1]),
                4:lambda n: (nodes[n], nodes[n+4], nodes[n+4*nWB+4], nodes[n+4*nWB],
                             nodes[n+1], nodes[n+2], nodes[n+3],
                             nodes[n+nWB+4], nodes[n+2*nWB+4], nodes[n+3*nWB+4],
                             nodes[n+4*nWB+3], nodes[n+4*nWB+2], nodes[n+4*nWB+1],
                             nodes[n+3*nWB], nodes[n+2*nWB], nodes[n+nWB],
                             nodes[n+nWB+1], nodes[n+nWB+2], nodes[n+nWB+3],
                             nodes[n+2*nWB+3], nodes[n+3*nWB+3],
                             nodes[n+3*nWB+2], nodes[n+3*nWB+1],
                             nodes[n+2*nWB+1], nodes[n+2*nWB+2])}
    nElemNodes = int((q+1)**2)
    nElems = int(((nWB - 1)/q * (nr - 1)/q))

    field = create_group(zone, f'quad_q{q}', 'Elements_t', f'quad_q{q}', 'I4')
    field.create_dataset(' data', dtype='<i4', data=[elem_type[q], 0])

    # create element range group
    er = create_group(field, 'ElementRange', 'IndexRange_t', 'ElementRange', 'I4')

    # create element connectivity group
    ec = create_group(field, 'ElementConnectivity', 'DataArray_t', 'ElementConnectivity', 'I4')

    er.create_dataset(' data', dtype='<i4', data=[1, nElems])
    ec.create_dataset(' data', shape=(nElems*nElemNodes,), dtype='<i4')
    ec = ec[' data']
    loc = 0
    for j in range(int((nr-1)/q)):
        for i in range(int((nWB-1)/q)):
            n = (j*nWB + i) * q
            ec[loc:loc+nElemNodes] = elem_map[q](n)
            loc += nElemNodes

    return field, nElems


def write_tri_field(zone, q, nodes, nWB, nr):
    elem_type = {1:5,   # Q1 refers to TRI_3
                 2:6,   # Q2 refers to TRI_6
                 3:26,  # Q3 refers to TRI_10
                 4:42}  # Q4 refers to TRI_15
    elem_map = {1:lambda n: (nodes[n], nodes[n+1], nodes[n+nWB], # Triangle 1
                             nodes[n+1], nodes[n+nWB+1], nodes[n+nWB]), # Triangle 2
                2:lambda n: (nodes[n], nodes[n+2], nodes[n+2*nWB], # Triangle 1
                             nodes[n+1], nodes[n+nWB+1], nodes[n+nWB],
                             nodes[n+2], nodes[n+2*nWB+2], nodes[n+2*nWB], # Triangle 2
                             nodes[n+nWB+2], nodes[n+2*nWB+1], nodes[n+nWB+1]),
                3:lambda n: (nodes[n], nodes[n+3], nodes[n+3*nWB], # Triangle 1
                             nodes[n+1], nodes[n+2], nodes[n+nWB+2], nodes[n+2*nWB+1],
                             nodes[n+2*nWB], nodes[n+nWB], nodes[n+nWB+1],
                             nodes[n+3], nodes[n+3*nWB+3], nodes[n+3*nWB], # Triangle 2
                             nodes[n+nWB+3], nodes[n+2*nWB+3], nodes[n+3*nWB+2], nodes[n+3*nWB+1],
                             nodes[n+2*nWB+1], nodes[n+nWB+2], nodes[n+2*nWB+2]),
                4:lambda n: (nodes[n], nodes[n+4], nodes[n+4*nWB], # Triangle 1
                             nodes[n+1], nodes[n+2], nodes[n+3],
                             nodes[n+nWB+3], nodes[n+2*nWB+2], nodes[n+3*nWB+1],
                             nodes[n+3*nWB], nodes[n+2*nWB], nodes[n+nWB],
                             nodes[n+nWB+1], nodes[n+nWB+2], nodes[n+2*nWB+1],
                             nodes[n+4],nodes[n+4*nWB+4],nodes[n+4*nWB], # Triangle 2
                             nodes[n+nWB+4], nodes[n+2*nWB+4], nodes[n+3*nWB+4],
                             nodes[n+4*nWB+3], nodes[n+4*nWB+2], nodes[n+4*nWB+1],
                             nodes[n+3*nWB+1], nodes[n+2*nWB+2], nodes[n+nWB+3],
                             nodes[n+2*nWB+3], nodes[n+3*nWB+3], nodes[n+3*nWB+2])}

    nElemNodes = int((q+1)*(q+2)/2)
    nElems = int(((nWB - 1)/q * (nr - 1)/q)) * 2

    field = create_group(zone, f'tri_q{q}', 'Elements_t', f'tri_q{q}', 'I4')
    field.create_dataset(' data', dtype='<i4', data=[elem_type[q], 0])

    # create element range group
    er = create_group(field, 'ElementRange', 'IndexRange_t', 'ElementRange', 'I4')

    # create element connectivity group
    ec = create_group(field, 'ElementConnectivity', 'DataArray_t', 'ElementConnectivity', 'I4')

    er.create_dataset(' data', dtype='<i4', data=[1, nElems])
    ec.create_dataset(' data', shape=(nElems*nElemNodes,), dtype='<i4')
    ec = ec[' data']
    loc = 0
    for j in range(int((nr-1)/q)):
        for i in range(int((nWB-1)/q)):
            n = (j*nWB + i) * q
            ec[loc:loc+nElemNodes*2] = elem_map[q](n)
            loc += nElemNodes*2

    return field, nElems


