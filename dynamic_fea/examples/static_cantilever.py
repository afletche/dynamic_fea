import numpy as np
import matplotlib.pyplot as plt
import time

from dynamic_fea.fea import FEA
from dynamic_fea.material import IsotropicMaterial
from dynamic_fea.component import Component
from dynamic_fea.element import Element, QuadElement
from dynamic_fea.mesh import UnstructuredMesh
from dynamic_fea.design_representation import FERepresentation


from lsdo_geo.core.geometry import Geometry
from lsdo_geo.core.mesh import Mesh

from vedo import Points, Plotter, LegendBox


path_name = 'CAD/'
file_name = 'box_10x5x1.stp'
geo = Geometry(path_name + file_name)

aluminum = IsotropicMaterial(name='aluminum', E=68e9, nu=0.36)
pla = IsotropicMaterial(name='pla', E=4.1e9)
thickness = 1


''' Inputs '''
NODES_PER_LENGTH = 10


down_direction = np.array([0., 0., -1.])
''' Points to be projected'''
point_00 = np.array([0., -2.5, 10.])
point_01 = np.array([0., 2.5, 10.])
point_10 = np.array([10, -2.5, 10.])
point_11 = np.array([10, 2.5, 10.])


'''
Calcualtes the number of divisions
'''
left_edge_length = np.linalg.norm(point_01-point_00)
bot_edge_length = np.linalg.norm(point_10-point_00)
num_pts_00_10 = [np.rint(bot_edge_length*NODES_PER_LENGTH).astype(int)]
num_pts_00_01 = [np.rint(left_edge_length*NODES_PER_LENGTH).astype(int)]


'''Project points'''
point_00_ptset, top_right_coord = geo.project_points(point_00, projection_direction = down_direction)
point_01_ptset, top_right_coord = geo.project_points(point_01, projection_direction = down_direction)
point_10_ptset, top_right_coord = geo.project_points(point_10, projection_direction = down_direction)
point_11_ptset, top_right_coord = geo.project_points(point_11, projection_direction = down_direction)

surface_shape1 = np.append(num_pts_00_10,num_pts_00_01)
top_edge_curve1 = geo.perform_linear_interpolation(point_01_ptset, point_11_ptset, num_pts_00_10)
bot_edge_curve1 = geo.perform_linear_interpolation(point_00_ptset, point_10_ptset, num_pts_00_10)
left_edge_curve1 = geo.perform_linear_interpolation(point_00_ptset, point_01_ptset, num_pts_00_01)
right_edge_curve1 = geo.perform_linear_interpolation(point_10_ptset, point_11_ptset, num_pts_00_01)
cantilever_pointset = geo.perform_2d_transfinite_interpolation(bot_edge_curve1, top_edge_curve1, left_edge_curve1, right_edge_curve1)

cantilever_mesh = Mesh(f'cantilever_mesh_{NODES_PER_LENGTH}')

geo.assemble()
geo.evaluate()

mesh_nodes = cantilever_pointset.physical_coordinates.reshape(cantilever_pointset.shape)

# # Plot mesh
# vp = Plotter(axes=1)
# nodes1 = Points(mesh_nodes.reshape((-1,3)), r = 7, c = 'plum').legend('mesh')
# vp.show(nodes1, 'Mesh', at=0, viewup="z", interactive = True)

elements = []
for u in range(mesh_nodes.shape[0]-1):
    for v in range(mesh_nodes.shape[1]-1):
        node_00 = mesh_nodes[u,v,:2]
        node_01 = mesh_nodes[u,v+1,:2]
        node_10 = mesh_nodes[u+1,v,:2]
        node_11 = mesh_nodes[u+1,v+1,:2]
        nodes = np.array([node_00, node_01, node_10, node_11])

        node_00_index = u*mesh_nodes.shape[1] + v
        node_01_index = u*mesh_nodes.shape[1] + (v+1)
        node_10_index = (u+1)*mesh_nodes.shape[1] + v
        node_11_index = (u+1)*mesh_nodes.shape[1] + (v+1)
        node_map = np.array([node_00_index, node_01_index, node_10_index, node_11_index])

        elements.append(QuadElement(nodes, node_map, pla, thickness))

mesh = UnstructuredMesh(name='cantilever_mesh', nodes=cantilever_pointset.physical_coordinates[:,:2] ,elements=elements)


bridge_prob_bc = []
for i in range(mesh_nodes.shape[1]):
    bridge_prob_bc.append((i, 0, 0))
    bridge_prob_bc.append((i, 1, 0))


load_cases = []

# ''' Load case Ex: downward on upper right tip'''
# upper_right_element = mesh.elements[-1]
# upper_right_node = upper_right_element.node_map[-1]
# bridge_prob_loads = [
#     (upper_right_node, np.array([0., 5.e6]))
# ]
# load_cases.append(bridge_prob_loads)

''' Load case 1: Distributed load on motor mount'''
x_bounds = np.array([8.99, 10.01])
y_bounds = np.array([2.49, 2.51])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)

load_nodes = mesh.nodes[mask]
load_node_indices = np.argwhere(mask)
load = np.array([0., 5.e6])
load_per_node = load/load_node_indices.shape[0]
bridge_prob_loads = []
for node in load_node_indices[:,0]:
    bridge_prob_loads.append((node, load_per_node))

load_cases.append(bridge_prob_loads)


''' Run for each load case'''
for i, load_case in enumerate(load_cases):
    print(f'------load case {i}-----')
    bridge_prob = FEA(mesh=mesh, loads=load_case, boundary_conditions=bridge_prob_bc)
    bridge_prob.setup()
    bridge_prob.evaluate()
    print('___Displacements___')
    U_reshaped = bridge_prob.U.reshape((bridge_prob.num_nodes, -1))
    print('max U_x: ', max(abs(U_reshaped[:,0])))
    print('max U_y: ', max(abs(U_reshaped[:,1])))
    bridge_prob.plot_displacements()
    bridge_prob.calc_stresses()
    print('___Stresses___')
    print('max sigma_xx: ', max(bridge_prob.stresses[:,:,0].reshape((-1,))))
    print('max sigma_yy: ', max(bridge_prob.stresses[:,:,1].reshape((-1,))))
    print('max sigma_xy: ', max(bridge_prob.stresses[:,:,2].reshape((-1,))))
    print('min sigma_xx: ', min(bridge_prob.stresses[:,:,0].reshape((-1,))))
    print('min sigma_yy: ', min(bridge_prob.stresses[:,:,1].reshape((-1,))))
    print('min sigma_xy: ', min(bridge_prob.stresses[:,:,2].reshape((-1,))))
    # bridge_prob.plot_stresses('xx')
    # bridge_prob.plot_stresses('yy')
    # bridge_prob.plot_stresses('xy')

