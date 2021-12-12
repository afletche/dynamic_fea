import numpy as np
import matplotlib.pyplot as plt
import time

from dynamic_fea.fea import FEA
from dynamic_fea.material import IsotropicMaterial
from dynamic_fea.component import Component
from dynamic_fea.element import Element, QuadElement
# from mesh import UnstructuredMesh
from dynamic_fea.examples.bridge_example.unstructured_matrix import UnstructuredMesh
from dynamic_fea.design_representation import FERepresentation


from lsdo_geo.core.geometry import Geometry
from lsdo_geo.core.mesh import Mesh

from examples.bridge_example.create_bridge_mesh import create_bridge_mesh
from vedo import Points, Plotter, LegendBox



''' 3 load cases '''
# path_name = 'fea/CAD/'
# file_name = 'box_10x5x1.stp'
# geo = Geometry(path_name + file_name)

steel_z_ratio = 0.03/0.2
steel_E = steel_z_ratio*200.e9 + (1-steel_z_ratio)*40.e9
concrete = IsotropicMaterial(name='concrete', E=40.e9, nu=0.2)  # replace with concrete material properties
steel = IsotropicMaterial(name='steel', E=steel_E, nu=0.3)  # replace with steel material properties
materials = [concrete, steel]
thickness = 1

# nodes_per_length = 75
# bridge_mesh = create_bridge_mesh(geo, [nodes_per_length])
# deck_pointset = bridge_mesh[0].pointset_list[0]
# corner_pointset = bridge_mesh[0].pointset_list[1]
# barrier_pointset = bridge_mesh[0].pointset_list[2]

# geo.assemble()
# geo.evaluate()

# deck_nodes = deck_pointset.physical_coordinates.reshape(deck_pointset.shape)
# corner_nodes = corner_pointset.physical_coordinates.reshape(corner_pointset.shape)
# barrier_nodes = barrier_pointset.physical_coordinates.reshape(barrier_pointset.shape)
# deck_nodes = np.swapaxes(deck_nodes, 0, 1)
# corner_nodes = np.swapaxes(corner_nodes, 0, 1)
# barrier_nodes = np.swapaxes(barrier_nodes, 0, 1)

# # Plot mesh
# vp = Plotter(axes=1)
# nodes1 = Points(deck_nodes.reshape((-1,3)), r = 7, c = 'plum').legend('mesh')
# nodes2 = Points(corner_nodes.reshape((-1,3)), r = 7, c = 'blue').legend('mesh')
# nodes3 = Points(barrier_nodes.reshape((-1,3)), r = 7, c = 'cyan').legend('mesh')

# vp.show(nodes1, nodes2, nodes3, 'Mesh', at=0, viewup="z", interactive = True)
# vp_test.show(nodes1, 'Mesh', at=0, viewup="z", interactive = True)


# mesh = UnstructuredMesh()
    
# mesh.structs.append(deck_nodes)
# mesh.structs.append(corner_nodes)
# mesh.structs.append(barrier_nodes)



# STICH_TOL = 1e-12
# mesh.stitch(STICH_TOL, materials, thickness)
# mesh.nodes = mesh.nodes[:,:2]

# rebar_thickness = 0.03
# bar1_left = np.array([0., 0.2])
# bar1_right = np.array([3.2, 0.7])
# bar2_left = np.array([0., 0.4])
# bar2_right = np.array([3.2, 0.8])
# bar3_left = np.array([0., 0.6])
# bar3_right = np.array([3.2, 0.9])
# bar4_left = np.array([0., 0.8])
# bar4_right = np.array([3.2, 0.98])
# mesh.addRebar(bar1_left, bar1_right, rebar_thickness, concrete, steel)
# mesh.addRebar(bar2_left, bar2_right, rebar_thickness, concrete, steel)
# mesh.addRebar(bar3_left, bar3_right, rebar_thickness, concrete, steel)
# mesh.addRebar(bar4_left, bar4_right, rebar_thickness, concrete, steel)
# # mesh.plotStructures()
# # mesh.plotMesh()



# bridge_prob_bc = []
# for i in range(deck_pointset.shape[1]):
#     bridge_prob_bc.append((i, 0, 0))
#     bridge_prob_bc.append((i, 1, 0))


# ''' Load case Ex: downward on upper right tip'''
# upper_right_element = mesh.elements[-1]
# upper_right_node = upper_right_element.node_map[-1]
# bridge_prob_loads = [
#     (upper_right_node, np.array([0., -5.e6]))
# ]

# load_cases = []

# ''' Load case 1: Distributed load on the barrier interior'''
# x_bounds = np.array([2.99, 3.01])
# y_bounds = np.array([0.99, 1.41])
# mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([510e3, 0.])
# load_per_node = load/load_node_indices.shape[0]
# bridge_prob_loads = []
# for node in load_node_indices[:,0]:
#     bridge_prob_loads.append((node, load_per_node))

# load_cases.append(bridge_prob_loads)

# ''' Load case 2: Truck load'''
# WEIGHT_PER_TIRE = 17.2e3

# x_bounds1 = np.array([0.54, 0.66])
# y_bounds1 = np.array([0.99, 1.01])
# mask_x1 = mesh.nodes[:,0] >= x_bounds1[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds1[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds1[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds1[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([0., -WEIGHT_PER_TIRE])
# load_per_node = load/load_node_indices.shape[0]
# bridge_prob_loads = []
# for node in load_node_indices[:,0]:
#     bridge_prob_loads.append((node, load_per_node))

# x_bounds2 = np.array([2.34, 2.46])
# y_bounds2 = np.array([0.99, 1.01])
# mask_x1 = mesh.nodes[:,0] >= x_bounds2[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds2[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds2[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds2[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)


# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([0., -WEIGHT_PER_TIRE])
# load_per_node = load/load_node_indices.shape[0]
# for node in load_node_indices[:,0]:
#     bridge_prob_loads.append((node, load_per_node))


# load_cases.append(bridge_prob_loads)

# ''' Load case 2b: Car load'''
# WEIGHT_PER_TIRE = 3.3e3

# x_bounds1 = np.array([0.54, 0.66])
# y_bounds1 = np.array([0.99, 1.01])
# mask_x1 = mesh.nodes[:,0] >= x_bounds1[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds1[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds1[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds1[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([0., -WEIGHT_PER_TIRE])
# load_per_node = load/load_node_indices.shape[0]
# bridge_prob_loads = []
# for node in load_node_indices[:,0]:
#     bridge_prob_loads.append((node, load_per_node))

# x_bounds2 = np.array([2.34, 2.46])
# y_bounds2 = np.array([0.99, 1.01])
# mask_x1 = mesh.nodes[:,0] >= x_bounds2[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds2[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds2[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds2[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([0., -WEIGHT_PER_TIRE])
# load_per_node = load/load_node_indices.shape[0]
# for node in load_node_indices[:,0]:
#     bridge_prob_loads.append((node, load_per_node))

# load_cases.append(bridge_prob_loads)


# ''' Load case 3: Hanging load'''
# x_bounds = np.array([2.99, 3.51])
# y_bounds = np.array([0., 0.61])
# mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)

# HANGING_WEIGHT = 100000.

# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([0., -HANGING_WEIGHT])
# load_per_node = load/load_node_indices.shape[0]
# bridge_prob_loads = []
# for node in load_node_indices[:,0]:
#     bridge_prob_loads.append((node, load_per_node))

# load_cases.append(bridge_prob_loads)

''' Run for each load case'''
# for i, load_case in enumerate(load_cases):
#     print(f'------load case {i}-----')
#     bridge_prob = FEA(mesh=mesh, loads=load_case, boundary_conditions=bridge_prob_bc)
#     bridge_prob.setup()
#     bridge_prob.evaluate()
#     print('___Displacements___')
#     U_reshaped = bridge_prob.U.reshape((bridge_prob.num_nodes, -1))
#     print('max U_x: ', max(abs(U_reshaped[:,0])))
#     print('max U_y: ', max(abs(U_reshaped[:,1])))
#     # bridge_prob.plot_displacements()
#     bridge_prob.calc_stresses()
#     print('___Stresses___')
#     print('max sigma_xx: ', max(bridge_prob.stresses[:,:,0].reshape((-1,))))
#     print('max sigma_yy: ', max(bridge_prob.stresses[:,:,1].reshape((-1,))))
#     print('max sigma_xy: ', max(bridge_prob.stresses[:,:,2].reshape((-1,))))
#     print('min sigma_xx: ', min(bridge_prob.stresses[:,:,0].reshape((-1,))))
#     print('min sigma_yy: ', min(bridge_prob.stresses[:,:,1].reshape((-1,))))
#     print('min sigma_xy: ', min(bridge_prob.stresses[:,:,2].reshape((-1,))))
#     bridge_prob.plot_stresses('xx')
#     bridge_prob.plot_stresses('yy')
#     bridge_prob.plot_stresses('xy')



''' Run convergence study'''
path_name = 'fea/CAD/'
file_name = 'box_10x5x1.stp'
geo = Geometry(path_name + file_name)

plt.figure()
nodes_per_length_queries = [10, 20, 30, 60, 90, 125]
# nodes_per_length_queries = np.linspace(10, 100, 10).astype(int)
# nodes_per_length_queries = np.delete(nodes_per_length_queries, [8])

meshes = create_bridge_mesh(geo, nodes_per_length_queries)
geo.assemble()
geo.evaluate()


solution_times = []
displacements = np.array([])
for i, mesh in enumerate(meshes):
    # bridge_mesh = create_bridge_mesh(geo, nodes_per_length)
    deck_pointset = mesh.pointset_list[0]
    corner_pointset = mesh.pointset_list[1]
    barrier_pointset = mesh.pointset_list[2]


    deck_nodes = deck_pointset.physical_coordinates.reshape(deck_pointset.shape)
    corner_nodes = corner_pointset.physical_coordinates.reshape(corner_pointset.shape)
    barrier_nodes = barrier_pointset.physical_coordinates.reshape(barrier_pointset.shape)
    deck_nodes = np.swapaxes(deck_nodes, 0, 1)
    corner_nodes = np.swapaxes(corner_nodes, 0, 1)
    barrier_nodes = np.swapaxes(barrier_nodes, 0, 1)

    mesh = UnstructuredMesh()
        
    mesh.structs.append(deck_nodes)
    mesh.structs.append(corner_nodes)
    mesh.structs.append(barrier_nodes)

    STICH_TOL = 1e-12
    mesh.stitch(STICH_TOL, materials, thickness)
    mesh.nodes = mesh.nodes[:,:2]


    bridge_prob_bc = []
    for j in range(deck_pointset.shape[1]):
        bridge_prob_bc.append((j, 0, 0))
        bridge_prob_bc.append((j, 1, 0))

    load_cases = []
    ''' Load case 2: Truck load'''
    WEIGHT_PER_TIRE = 17.2e3

    x_bounds1 = np.array([0.54, 0.66])
    y_bounds1 = np.array([0.99, 1.01])
    mask_x1 = mesh.nodes[:,0] >= x_bounds1[0]
    mask_x2 = mesh.nodes[:,0] <= x_bounds1[1]
    mask_y1 = mesh.nodes[:,1] >= y_bounds1[0]
    mask_y2 = mesh.nodes[:,1] <= y_bounds1[1]
    mask_x = np.logical_and(mask_x1, mask_x2)
    mask_y = np.logical_and(mask_y1, mask_y2)
    mask = np.logical_and(mask_x, mask_y)

    load_nodes = mesh.nodes[mask]
    load_node_indices = np.argwhere(mask)
    load = np.array([0., -WEIGHT_PER_TIRE])
    load_per_node = load/load_node_indices.shape[0]
    bridge_prob_loads = []
    for node in load_node_indices[:,0]:
        bridge_prob_loads.append((node, load_per_node))

    x_bounds2 = np.array([2.34, 2.46])
    y_bounds2 = np.array([0.99, 1.01])
    mask_x1 = mesh.nodes[:,0] >= x_bounds2[0]
    mask_x2 = mesh.nodes[:,0] <= x_bounds2[1]
    mask_y1 = mesh.nodes[:,1] >= y_bounds2[0]
    mask_y2 = mesh.nodes[:,1] <= y_bounds2[1]
    mask_x = np.logical_and(mask_x1, mask_x2)
    mask_y = np.logical_and(mask_y1, mask_y2)
    mask = np.logical_and(mask_x, mask_y)


    load_nodes = mesh.nodes[mask]
    load_node_indices = np.argwhere(mask)
    load = np.array([0., -WEIGHT_PER_TIRE])
    load_per_node = load/load_node_indices.shape[0]
    bridge_prob_loads = []
    for node in load_node_indices[:,0]:
        bridge_prob_loads.append((node, load_per_node))

    load_cases.append(bridge_prob_loads)


    time_start = time.time()
    bridge_prob = FEA(mesh=mesh, loads=bridge_prob_loads, boundary_conditions=bridge_prob_bc)
    bridge_prob.setup()
    bridge_prob.evaluate()
    time_end = time.time()
    solution_times.append(time_end - time_start)
    print(time_end - time_start)
    displacements = np.append(displacements, bridge_prob.U[-1])

    # plt.plot(nodes_per_length_queries[i], bridge_prob.U[-1], '-bo')
    plt.plot(bridge_prob.num_elements, bridge_prob.U[-1], '-bo')

print('Solution times: ', solution_times)
print("Displacements: ", displacements)

plt.xlabel("Number of Mesh Elements")
plt.ylabel("Displacement of the upper tip node (m)")
plt.title("Mesh Refinement Convergence Study")

plt.show()



''' "Initial" Design '''
# path_name = 'fea/CAD/'
# file_name = 'box_10x5x1.stp'
# geo = Geometry(path_name + file_name)

# nodes_per_length = 75
# bridge_mesh = create_bridge_mesh(geo, [nodes_per_length])[0]
# deck_pointset = bridge_mesh.pointset_list[0]
# corner_pointset = bridge_mesh.pointset_list[1]
# barrier_pointset = bridge_mesh.pointset_list[2]

# geo.assemble()
# geo.evaluate()

# deck_nodes = deck_pointset.physical_coordinates.reshape(deck_pointset.shape)
# corner_nodes = corner_pointset.physical_coordinates.reshape(corner_pointset.shape)
# barrier_nodes = barrier_pointset.physical_coordinates.reshape(barrier_pointset.shape)
# deck_nodes = np.swapaxes(deck_nodes, 0, 1)
# corner_nodes = np.swapaxes(corner_nodes, 0, 1)
# barrier_nodes = np.swapaxes(barrier_nodes, 0, 1)

# mesh = UnstructuredMesh()
    
# mesh.structs.append(deck_nodes)
# mesh.structs.append(corner_nodes)
# mesh.structs.append(barrier_nodes)

# STICH_TOL = 1e-12
# mesh.stitch(STICH_TOL, materials, thickness)
# mesh.nodes = mesh.nodes[:,:2]

# rebar_thickness = 0.03
# bar1_left = np.array([0., 0.2])
# bar1_right = np.array([3.2, 0.2])
# bar2_left = np.array([0., 0.4])
# bar2_right = np.array([3.2, 0.4])
# bar3_left = np.array([0., 0.6])
# bar3_right = np.array([3.2, 0.6])
# bar4_left = np.array([0., 0.8])
# bar4_right = np.array([3.2, 0.8])
# mesh.addRebar(bar1_left, bar1_right, rebar_thickness, concrete, steel)
# mesh.addRebar(bar2_left, bar2_right, rebar_thickness, concrete, steel)
# mesh.addRebar(bar3_left, bar3_right, rebar_thickness, concrete, steel)
# mesh.addRebar(bar4_left, bar4_right, rebar_thickness, concrete, steel)
# # mesh.plotStructures()
# # mesh.plotMesh()


# bridge_prob_bc = []
# for i in range(deck_pointset.shape[1]):
#     bridge_prob_bc.append((i, 0, 0))
#     bridge_prob_bc.append((i, 1, 0))


# load_cases = []

# ''' Load case 1: Distributed load on the barrier interior'''
# x_bounds = np.array([2.99, 3.01])
# y_bounds = np.array([0.99, 1.41])
# mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([510e3, 0.])
# load_per_node = load/load_node_indices.shape[0]
# bridge_prob_loads = []
# for node in load_node_indices[:,0]:
#     bridge_prob_loads.append((node, load_per_node))

# load_cases.append(bridge_prob_loads)

# ''' Load case 2: Truck load'''
# WEIGHT_PER_TIRE = 17.2e3

# x_bounds1 = np.array([0.54, 0.66])
# y_bounds1 = np.array([0.99, 1.01])
# mask_x1 = mesh.nodes[:,0] >= x_bounds1[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds1[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds1[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds1[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([0., -WEIGHT_PER_TIRE])
# load_per_node = load/load_node_indices.shape[0]
# bridge_prob_loads = []
# for node in load_node_indices[:,0]:
#     bridge_prob_loads.append((node, load_per_node))

# x_bounds2 = np.array([2.34, 2.46])
# y_bounds2 = np.array([0.99, 1.01])
# mask_x1 = mesh.nodes[:,0] >= x_bounds2[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds2[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds2[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds2[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)


# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([0., -WEIGHT_PER_TIRE])
# load_per_node = load/load_node_indices.shape[0]
# for node in load_node_indices[:,0]:
#     bridge_prob_loads.append((node, load_per_node))


# load_cases.append(bridge_prob_loads)

# ''' Load case 2b: Car load'''
# WEIGHT_PER_TIRE = 3.3e3

# x_bounds1 = np.array([0.54, 0.66])
# y_bounds1 = np.array([0.99, 1.01])
# mask_x1 = mesh.nodes[:,0] >= x_bounds1[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds1[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds1[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds1[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([0., -WEIGHT_PER_TIRE])
# load_per_node = load/load_node_indices.shape[0]
# bridge_prob_loads = []
# for node in load_node_indices[:,0]:
#     bridge_prob_loads.append((node, load_per_node))

# x_bounds2 = np.array([2.34, 2.46])
# y_bounds2 = np.array([0.99, 1.01])
# mask_x1 = mesh.nodes[:,0] >= x_bounds2[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds2[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds2[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds2[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([0., -WEIGHT_PER_TIRE])
# load_per_node = load/load_node_indices.shape[0]
# for node in load_node_indices[:,0]:
#     bridge_prob_loads.append((node, load_per_node))

# load_cases.append(bridge_prob_loads)


# ''' Run for each load case'''
# for i, load_case in enumerate(load_cases):
#     print(f'------load case {i}-----')
#     bridge_prob = FEA(mesh=mesh, loads=load_case, boundary_conditions=bridge_prob_bc)
#     bridge_prob.setup()
#     bridge_prob.evaluate()
#     print('___Displacements___')
#     U_reshaped = bridge_prob.U.reshape((bridge_prob.num_nodes, -1))
#     print('max U_x: ', max(abs(U_reshaped[:,0])))
#     print('max U_y: ', max(abs(U_reshaped[:,1])))
#     bridge_prob.plot_displacements()
#     bridge_prob.calc_stresses()
#     print('___Stresses___')
#     print('max sigma_xx: ', max(bridge_prob.stresses[:,:,0].reshape((-1,))))
#     print('max sigma_yy: ', max(bridge_prob.stresses[:,:,1].reshape((-1,))))
#     print('max sigma_xy: ', max(bridge_prob.stresses[:,:,2].reshape((-1,))))
#     print('min sigma_xx: ', min(bridge_prob.stresses[:,:,0].reshape((-1,))))
#     print('min sigma_yy: ', min(bridge_prob.stresses[:,:,1].reshape((-1,))))
#     print('min sigma_xy: ', min(bridge_prob.stresses[:,:,2].reshape((-1,))))
#     # bridge_prob.plot_stresses('xx')
#     # bridge_prob.plot_stresses('yy')
#     # bridge_prob.plot_stresses('xy')






# ''' Homework example '''
# nodes = np.array([
#     [0., 0.],
#     [0., 0.0025],
#     [0., 0.005],
#     [0.00333, 0.],
#     [0.00333, 0.0025],
#     [0.00333, 0.005],
#     [0.00666, 0.],
#     [0.00666, 0.0025],
#     [0.01, 0.],
#     [0.01, 0.005]
# ])

# num_elements = 5

# element_nodes = np.array([
#     [0, 1, 3, 4],
#     [1, 2, 4, 5],
#     [3, 4, 6, 7],
#     [4, 5, 7, 9],
#     [6, 7, 8, 9]
# ])

# hw5_mat = IsotropicMaterial(name='hw5_mat', E=100e9, nu=0.3)
# thickness = 0.001   # m

# elements = []
# for i in range(num_elements):
#     elements.append(QuadElement(nodes=nodes[element_nodes[i,:]], node_map=element_nodes[i,:], material=hw5_mat, thickness=thickness))




# hw5_component = Component(name='hw5_structure')
# hw5_component_properties_dict = {
#     'material' : hw5_mat,
#     'thickness': thickness
# }
# hw5_component.add_properties(hw5_component_properties_dict)

# # hw5_mesh = UnstructuredMesh(name='hw_mesh', nodes=nodes, element_nodes=element_nodes)
# hw5_mesh = UnstructuredMesh(name='hw_mesh', nodes=nodes, elements=elements)
# # hw5_component.add_design_representations(hw5_mesh)

# hw5_fe_representation = FERepresentation(name='hw5_fe_representation', components=[hw5_component], meshes=[hw5_mesh])

# load_nodes = np.array([2, 5, 9, 8, 9])
# load_forces = np.array([
#     [0., 30./4e3],
#     [0., 11.25e3],
#     [0., 11.25e3],
#     [0.4e3, 0.],
#     [1.6e3, 0.]
#     ])

# hw5_loads = [
#     (2, np.array([0., 5.e3])),
#     (5, np.array([0., 15.e3])),
#     (9, np.array([0., 10.e3])),
#     (8, np.array([0.4e3, 0.])),
#     (9, np.array([1.6e3, 0.]))
#     ]


# hw5_boundary_conditions = [
#     (0, 0, 0),
#     (1, 0, 0),
#     (2, 0, 0),
#     (0, 1, 0),
#     (3, 1, 0),
#     (6, 1, 0),
#     (8, 1, 0)
# ]

# hw5 = FEA(design_representation=hw5_fe_representation, loads=hw5_loads, boundary_conditions=hw5_boundary_conditions)
# hw5.setup()
# hw5.evaluate()
# print(hw5.U)
# hw5.plot()

# hw5.calc_stresses()
# print(hw5.stresses[4,:,:2])     # normal stressses of element A
