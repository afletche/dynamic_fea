from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import time

from dynamic_fea.fea import FEA
from dynamic_fea.material import IsotropicMaterial
from dynamic_fea.component import Component
from dynamic_fea.element import Element, QuadElement
from dynamic_fea.mesh import UnstructuredMesh
from dynamic_fea.design_representation import FERepresentation


from lsdo_kit.design.design_geometry.design_geometry import DesignGeometry
from lsdo_kit.simulation.mesh.mesh import Mesh

from vedo import Points, Plotter, LegendBox


path_name = 'CAD/'
file_name = 'box_10x5x1.stp'
geo = DesignGeometry(path_name + file_name)

aluminum = IsotropicMaterial(name='aluminum', E=68e9, nu=0.36)
pla = IsotropicMaterial(name='pla', E=4.1e9)
nylon = IsotropicMaterial(name='nylon', E=2.7e9, nu = 0.39)
thickness = 0.1


''' Inputs '''
NODES_PER_LENGTH = 50.
# NODES_PER_LENGTH = 5. BAD


down_direction = np.array([0., 0., -1.])
''' Points to be projected'''
# point_00 = np.array([0., -2.5, 10.])
# point_01 = np.array([0., 2.5, 10.])
# point_10 = np.array([10, -2.5, 10.])
# point_11 = np.array([10, 2.5, 10.])

point_00 = np.array([0., 0., 10.])
point_01 = np.array([0., 0.1, 10.])
point_10 = np.array([0.5, 0., 10.])
point_11 = np.array([0.5, 0.1, 10.])


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

# Plot mesh
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

        elements.append(QuadElement(nodes, node_map, nylon, thickness))

mesh = UnstructuredMesh(name='cantilever_mesh', nodes=cantilever_pointset.physical_coordinates[:,:2], elements=elements)


cantilever_prob_bc = []
# for i in range(mesh_nodes.shape[1]):
#     cantilever_prob_bc.append((i, 0, 0))
#     cantilever_prob_bc.append((i, 1, 0))

x_bounds = np.array([0.24, 0.26])
y_bounds = np.array([-5.01, 5.01])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)

bc_node_indices = np.argwhere(mask)
for node in bc_node_indices[:,0]:
    cantilever_prob_bc.append((node, 0, 0))
    cantilever_prob_bc.append((node, 1, 0))

print('bc_nodes', bc_node_indices)
''' Load case 2: Distributed load on motor mount'''
# x_bounds = np.array([8.99, 10.01])
# y_bounds = np.array([2.49, 2.51])
# mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([0., 1.e2])
# # load = np.array([1.e7, 0.])
# load_per_node = load/load_node_indices.shape[0]
# cantilever_prob_loads = []
# for node in load_node_indices[:,0]:
#     cantilever_prob_loads.append((node, load_per_node))

# x_bounds = np.array([-0.01, .01])
# y_bounds = np.array([2.49, 2.51])
# mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
# mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
# mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
# mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
# mask_x = np.logical_and(mask_x1, mask_x2)
# mask_y = np.logical_and(mask_y1, mask_y2)
# mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
# load_node_indices = np.argwhere(mask)
# load = np.array([0., 1.e2])
# # load = np.array([1.e7, 0.])
# load_per_node = load/load_node_indices.shape[0]
# for node in load_node_indices[:,0]:
#     cantilever_prob_loads.append((node, load_per_node))

# load_cases.append(cantilever_prob_loads)


''' Run for each load case'''
# for i, load_case in enumerate(load_cases):
    # print(f'------load case {i}-----')
    # cantilever_prob = FEA(mesh=mesh, loads=load_case, boundary_conditions=cantilever_prob_bc)
    # cantilever_prob.setup()
    # cantilever_prob.evaluate()
    # print('___Displacements___')
    # U_reshaped = cantilever_prob.U.reshape((cantilever_prob.num_nodes, -1))
    # print('max U_x: ', max(abs(U_reshaped[:,0])))
    # print('max U_y: ', max(abs(U_reshaped[:,1])))
    # cantilever_prob.plot_displacements()
    # cantilever_prob.calc_stresses()
    # print('___Stresses___')
    # print('max sigma_xx: ', np.max(cantilever_prob.stresses_per_point[:,0].reshape((-1,))))
    # print('max sigma_yy: ', np.max(cantilever_prob.stresses_per_point[:,1].reshape((-1,))))
    # print('max sigma_xy: ', np.max(cantilever_prob.stresses_per_point[:,2].reshape((-1,))))
    # print('min sigma_xx: ', np.min(cantilever_prob.stresses_per_point[:,0].reshape((-1,))))
    # print('min sigma_yy: ', np.min(cantilever_prob.stresses_per_point[:,1].reshape((-1,))))
    # print('min sigma_xy: ', np.min(cantilever_prob.stresses_per_point[:,2].reshape((-1,))))
    # cantilever_prob.plot_stresses('xx')
    # cantilever_prob.plot_stresses('yy')
    # cantilever_prob.plot_stresses('xy')
''' End of static case'''

# disps = cantilever_prob.U_f
# velocities = np.zeros_like(disps)
# steady_sol = np.append(disps, velocities).reshape((-1, 1))
# print(load_cases)
# load_cases = []
# cantilever_prob_loads = [(0, np.array([0., 0.]))]
# load_cases.append(cantilever_prob_loads)

t0 = 0.
tf = 0.1
nt = (tf-t0)*1000
t_eval = np.linspace(t0, tf, nt+1)
load_cases = []
dynamic_loads = []
input_loads = [[np.array([0., 1.e2]), np.array([0., 0.e1]), np.array([0., 1.e2]), np.array([0., 0.e1])],
                [np.array([0., 1.e2]), np.array([0., 0.e1]), np.array([0., 1.e2]), np.array([0., 0.e1])]]
t_loads = [0., tf/4, tf/2, tf*3/4]
num_loads = len(input_loads)
for i, t in enumerate(t_loads):
    # t = t_loads[i]
    dynamic_loads_per_time = []
    x_bounds = np.array([-0.001, 0.031])
    y_bounds = np.array([0.09, 0.11])
    mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
    mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
    mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
    mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
    mask_x = np.logical_and(mask_x1, mask_x2)
    mask_y = np.logical_and(mask_y1, mask_y2)
    mask = np.logical_and(mask_x, mask_y)

    load_nodes = mesh.nodes[mask]
    load_node_indices = np.argwhere(mask)
    load = input_loads[0][i]
    load_per_node = load/load_node_indices.shape[0]
    for node in load_node_indices[:,0]:
        dynamic_loads_per_time.append((node, load_per_node))

    x_bounds = np.array([0.469, .501])
    y_bounds = np.array([0.09, 0.11])
    mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
    mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
    mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
    mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
    mask_x = np.logical_and(mask_x1, mask_x2)
    mask_y = np.logical_and(mask_y1, mask_y2)
    mask = np.logical_and(mask_x, mask_y)

    load_nodes = mesh.nodes[mask]
    load_node_indices = np.argwhere(mask)
    load = input_loads[1][i]
    load_per_node = load/load_node_indices.shape[0]
    for node in load_node_indices[:,0]:
        dynamic_loads_per_time.append((node, load_per_node))

    dynamic_loads.append([t, dynamic_loads_per_time])
load_cases.append(dynamic_loads)

''' Run dynamics for each load case'''
for i, load_case in enumerate(load_cases):
    print(f'------load case {i}-----')
    cantilever_prob = FEA(mesh=mesh, boundary_conditions=cantilever_prob_bc)
    cantilever_prob.setup_dynamics()
    cantilever_prob.evaluate_dynamics(loads=load_case, t_eval=t_eval)
    U_reshaped = cantilever_prob.U_per_dim_per_time

    stresses = cantilever_prob.calc_dynamic_stresses()
    cantilever_prob.plot_dynamic_stresses('xx', time_step=-1)
    cantilever_prob.plot_dynamic_stresses('yy', time_step=-1)
    cantilever_prob.plot_dynamic_stresses('xy', time_step=-1)
    cantilever_prob.plot_dynamic_stresses('yy', dof=-1)

    # print('___Displacements___')
    # print(cantilever_prob.U)
    # print(cantilever_prob.U.shape)
    # print('max U_x: ', max(abs(U_reshaped[:,0])))
    # print('max U_y: ', max(abs(U_reshaped[:,1])))
    # print('___Stresses___')
    # print('max sigma_xx: ', max(cantilever_prob.stresses[:,:,0].reshape((-1,))))
    # print('max sigma_yy: ', max(cantilever_prob.stresses[:,:,1].reshape((-1,))))
    # print('max sigma_xy: ', max(cantilever_prob.stresses[:,:,2].reshape((-1,))))
    # print('min sigma_xx: ', min(cantilever_prob.stresses[:,:,0].reshape((-1,))))
    # print('min sigma_yy: ', min(cantilever_prob.stresses[:,:,1].reshape((-1,))))
    # print('min sigma_xy: ', min(cantilever_prob.stresses[:,:,2].reshape((-1,))))


plt.plot(t_eval, cantilever_prob.U[-1, :], '-r', label='Upper right node')
plt.plot(t_eval, cantilever_prob.U[-3, :], '-b', label='Node below upper right node')

plt.title(f'Y-Displacement of Adjacent Nodes')
plt.xlabel('t (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.show()


# plt.plot(t, cantilever_prob.rigid_body_disp[1,:].reshape((-1,)), '-bo')
# plt.show()

cantilever_prob.plot_dynamics(show_plots=False, video_file_name='displacement_animation.avi', fps=20)