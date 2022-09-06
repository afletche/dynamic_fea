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

# aluminum = IsotropicMaterial(name='aluminum', E=68e9, nu=0.36)
# pla = IsotropicMaterial(name='pla', E=4.1e9)
nylon = IsotropicMaterial(name='nylon', E=2.7e9, nu = 0.39, density=115.)
thickness = 0.04


''' Inputs '''
# NODES_PER_LENGTH = 50.
NODES_PER_LENGTH = 75.
# NODES_PER_LENGTH = 30.
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
quadrotor_2d_pointset = geo.perform_2d_transfinite_interpolation(bot_edge_curve1, top_edge_curve1, left_edge_curve1, right_edge_curve1)

quadrotor_2d_mesh = Mesh(f'quadrotor_2d_mesh_{NODES_PER_LENGTH}')

geo.assemble()
geo.evaluate()

mesh_nodes = quadrotor_2d_pointset.physical_coordinates.reshape(quadrotor_2d_pointset.shape)

# # Plot mesh
# vp = Plotter(axes=1)
# nodes1 = Points(mesh_nodes.reshape((-1,3)), r = 7, c = 'plum').legend('mesh')
# vp.show(nodes1, 'Mesh', at=0, viewup="z", interactive = True)

plane_stress_or_plane_strain = 'strain'
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

        elements.append(QuadElement(plane_stress_or_plane_strain, nodes, node_map, nylon, thickness))

mesh = UnstructuredMesh(name='quadrotor_2d_mesh', nodes=quadrotor_2d_pointset.physical_coordinates[:,:2], elements=elements)

rotor_mass = 0.064 + 0.01786    # kg
battery_mass = 0.529    # kg

left_motor_point_mass = []
x_bounds = np.array([-0.001, 0.031])
y_bounds = np.array([0.09, 0.11])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
load_node_indices = np.argwhere(mask)
load = rotor_mass
load_per_node = load/load_node_indices.shape[0]
for node in load_node_indices[:,0]:
    left_motor_point_mass.append((node, load_per_node))

right_motor_point_mass = []
x_bounds = np.array([0.469, .501])
y_bounds = np.array([0.09, 0.11])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
load_node_indices = np.argwhere(mask)
load = rotor_mass
load_per_node = load/load_node_indices.shape[0]
for node in load_node_indices[:,0]:
    right_motor_point_mass.append((node, load_per_node))

mesh.add_point_masses(left_motor_point_mass)
mesh.add_point_masses(right_motor_point_mass)

battery_point_mass = []
x_bounds = np.array([0.2, .3])
y_bounds = np.array([0.09, 0.11])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
load_node_indices = np.argwhere(mask)
load = battery_mass
load_per_node = load/load_node_indices.shape[0]
for node in load_node_indices[:,0]:
    battery_point_mass.append((node, load_per_node))

mesh.add_point_masses(left_motor_point_mass)
mesh.add_point_masses(right_motor_point_mass)
mesh.add_point_masses(battery_point_mass)



quadrotor_2d_prob_bc = []

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
    quadrotor_2d_prob_bc.append((node, 0, 0))
    quadrotor_2d_prob_bc.append((node, 1, 0))


t0 = 0.
# tf = 0.0001
tf = 10.
# nt = np.rint((tf-t0)*100).astype(int) # To see displacement
# nt = np.rint((tf-t0)*1000).astype(int) # To see displacement
# nt = np.rint((tf-t0)*1000000).astype(int)      # to see stresses (time step must be small too)
# t_eval = np.linspace(t0, tf, nt+1)
# dynamic_loads = []
input_loads = [[np.array([0., 1.e2]), np.array([0., 0.2e1]), np.array([0., 1.e2]), np.array([0., 3.4e2])],
                [np.array([0., 1.2e2]), np.array([0., 0.e1]), np.array([0., 1.4e2]), np.array([0., 3.e2])]]

# t_loads = [0., tf/4, tf/2, tf*3/4]
t_loads = np.linspace(t0, tf*3/4, 4)


dynamic_loads = []
for i, t in enumerate(t_loads):
    # t = time[i]
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

    # load_nodes = mesh.nodes[mask]
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

    # load_nodes = mesh.nodes[mask]
    load_node_indices = np.argwhere(mask)
    load = input_loads[1][i]
    load_per_node = load/load_node_indices.shape[0]
    for node in load_node_indices[:,0]:
        dynamic_loads_per_time.append((node, load_per_node))

    dynamic_loads.append([t, dynamic_loads_per_time])

from dynamic_fea.io.import_loads import import_loads
dynamic_loads, t_eval = import_loads(mesh=mesh, file_path='quadrotor_input_profile.json')
# print(dynamic_loads)
# t_eval = dynamic_loads[:][0]
t_eval = np.linspace(0., 10., 201)

''' Run dynamics'''
# for i, load_case in enumerate(load_cases):
print(f'------load case-----')
quadrotor_2d_prob = FEA(mesh=mesh, boundary_conditions=quadrotor_2d_prob_bc)
quadrotor_2d_prob.setup_dynamics()
quadrotor_2d_prob.apply_self_weight(g=9.81)
quadrotor_2d_prob.evaluate_dynamics(loads=dynamic_loads, t_eval=t_eval)
U_reshaped = quadrotor_2d_prob.U_per_dim_per_time

stresses = quadrotor_2d_prob.evaluate_stresses()
quadrotor_2d_prob.plot(stress_type='xx', time_step=-1)
# quadrotor_2d_prob.plot(stress_type='yy', time_step=-1)
# quadrotor_2d_prob.plot(stress_type='xy', time_step=-1)
# quadrotor_2d_prob.plot(stress_type='yy', dof=-1)
# quadrotor_2d_prob.plot(dof=[-1, -3], show_dislpacements=True)
# quadrotor_2d_prob.plot(dof=[-1, -3], show_dislpacements=True, stress_type='yy')
# quadrotor_2d_prob.plot(stress_type='vm', time_step=-1)
# quadrotor_2d_prob.plot(stress_type='avm', time_step=-1)
# quadrotor_2d_prob.plot(stress_type='vm', dof=-1)
# quadrotor_2d_prob.plot(stress_type='avm', dof=-1)
# quadrotor_2d_prob.plot(show_dislpacements=True, show_connections=False, stress_type='xx', video_file_name='displacement_animation.avi', video_fps=20, show=False)
# quadrotor_2d_prob.plot_rigid_body_displacement()
# quadrotor_2d_prob.plot_rigid_body_displacement(x_axis='t', y_axis='y')
# quadrotor_2d_prob.plot_rigid_body_displacement(x_axis='t', y_axis='x')
# quadrotor_2d_prob.plot_rigid_body_displacement(x_axis='t', y_axis='rot_z')

# print('___Displacements___')
# print(quadrotor_2d_prob.U)
# print(quadrotor_2d_prob.U.shape)
# print('max U_x: ', max(abs(U_reshaped[:,0])))
# print('max U_y: ', max(abs(U_reshaped[:,1])))
# print('___Stresses___')
# print('max sigma_xx: ', max(quadrotor_2d_prob.stresses[:,:,0].reshape((-1,))))
# print('max sigma_yy: ', max(quadrotor_2d_prob.stresses[:,:,1].reshape((-1,))))
# print('max sigma_xy: ', max(quadrotor_2d_prob.stresses[:,:,2].reshape((-1,))))
# print('min sigma_xx: ', min(quadrotor_2d_prob.stresses[:,:,0].reshape((-1,))))
# print('min sigma_yy: ', min(quadrotor_2d_prob.stresses[:,:,1].reshape((-1,))))
# print('min sigma_xy: ', min(quadrotor_2d_prob.stresses[:,:,2].reshape((-1,))))