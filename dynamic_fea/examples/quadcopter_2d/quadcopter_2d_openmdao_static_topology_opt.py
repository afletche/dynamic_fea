'''
The 2d quadcopter arm optimization, but with openmdao.
'''
import numpy as np
import openmdao.api as om

from dynamic_fea.fea import FEA
from dynamic_fea.material import IsotropicMaterial
from dynamic_fea.component import Component
from dynamic_fea.element import Element, QuadElement
from dynamic_fea.mesh import UnstructuredMesh
from dynamic_fea.design_representation import FERepresentation

from dynamic_fea.examples.quadcopter_2d.evaluate_fea import EvaluateFea

# from optimization_framework.optimization_problem import OptimizationProblem
# from optimization_framework.optimizers.gradient_descent_optimizer import GradientDescentOptimizer
# from optimization_framework.optimizers.finite_difference import finite_difference

# from vedo import Points, Plotter, LegendBox

import sys


nylon = IsotropicMaterial(name='nylon', E=2.7e9, nu = 0.39, density=1150., sigma_y=40e6)
# height = 10.e-2
# height = 9.9e-2
height = 5.e-2
length = 5.e-1
thickness = 1.e-2

''' Inputs '''
# # NODES_PER_LENGTH = 50.
# NODES_PER_LENGTH = 30.
# # NODES_PER_LENGTH = 5. BAD

'''
Calcualtes the number of divisions
'''
# left_edge_length = np.linalg.norm(point_01-point_00)
# bot_edge_length = np.linalg.norm(point_10-point_00)
# num_pts_00_10 = [np.rint(bot_edge_length*NODES_PER_LENGTH).astype(int)]
# num_pts_00_01 = [np.rint(left_edge_length*NODES_PER_LENGTH).astype(int)]
# num_pts_00_10 = 51
# num_pts_00_10 = 33
# num_pts_00_10 = 26
num_pts_00_10 = 31
# num_pts_00_10 = 33
# num_pts_00_01 = 11
# num_pts_00_01 = 8
# num_pts_00_01 = 6
num_pts_00_01 = 4
# num_pts_00_01 = 4


point_00 = np.array([0., 0., 0.])
point_01 = np.array([0., height, 0.])
point_10 = np.array([length, 0., 0.])
point_11 = np.array([length, height, 0.])

left_edge = np.linspace(point_00, point_01, num_pts_00_01)
right_edge = np.linspace(point_10, point_11, num_pts_00_01)
mesh_nodes = np.linspace(left_edge, right_edge, num_pts_00_10)

# # Plot mesh
# vp = Plotter(axes=1)
# nodes1 = Points(mesh_nodes.reshape((-1,3)), r = 7, c = 'plum').legend('mesh')
# vp.show(nodes1, 'Mesh', at=0, viewup="z", interactive = True)

plane_stress_or_plane_strain = 'stress'
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

# mesh = UnstructuredMesh(name='quadrotor_2d_mesh', nodes=quadrotor_2d_pointset.physical_coordinates[:,:2], elements=elements)
mesh = UnstructuredMesh(name='quadrotor_2d_mesh', nodes=mesh_nodes[:,:,:2].reshape((-1, 2)), elements=elements)
mesh.set_material(material=nylon)

rotor_mass = 0.064 + 0.01786    # kg
battery_mass = 0.529    # kg

left_motor_point_mass = []
x_bounds = np.array([-0.001, 0.031])
y_bounds = np.array([height - 0.001, height + 0.001])
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
x_bounds = np.array([length - 0.031, length + 0.001])
y_bounds = np.array([height - 0.001, height + 0.001])
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
x_bounds = np.array([length/2 - 0.01, length/2 + 0.01])
y_bounds = np.array([height - 0.001, height + 0.001])
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

x_bounds = np.array([length/2 - 0.01, length/2 + 0.01])
y_bounds = np.array([-.001, height + 0.001])
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



input_load = np.array([[0., 30.], [0., 30.]])

static_loads = []
x_bounds = np.array([-0.001, 0.031])
y_bounds = np.array([height - 0.001, height + 0.001])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
load_node_indices = np.argwhere(mask)
load = input_load[0]
load_per_node = load/load_node_indices.shape[0]
for node in load_node_indices[:,0]:
    static_loads.append((node, load_per_node))

x_bounds = np.array([length - 0.031, length + 0.001])
y_bounds = np.array([height - 0.001, height + 0.001])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)

# load_nodes = mesh.nodes[mask]
load_node_indices = np.argwhere(mask)
load = input_load[1]
load_per_node = load/load_node_indices.shape[0]
for node in load_node_indices[:,0]:
    static_loads.append((node, load_per_node))

''' Run dynamics'''
# for i, load_case in enumerate(load_cases):
quadrotor_2d_prob = FEA(mesh=mesh, boundary_conditions=quadrotor_2d_prob_bc)
quadrotor_2d_prob.apply_loads(static_loads)
quadrotor_2d_prob.setup()
quadrotor_2d_prob.apply_self_weight(g=9.81)


prob = om.Problem()

independent_variable_component = om.IndepVarComp()
topology_densities = np.ones((quadrotor_2d_prob.num_elements,))*0.5
# topology_densities = np.array([0.8, 0.6, 0.4, 0.2, 0.02, 0.95, 0.9, 0.8, 0.02, 0.02, 0.05, 0.9])
# topology_densities = np.array([0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.21,0.22,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.36,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.32,0.35,0.  ,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.47,0.38,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.5 ,0.  ,0.37,0.  ,0.  ,0.  ,0.  ,0.  ,0.5 ,0.  ,0.  ,0.34,0.  ,0.  ,
#  0.  ,0.  ,0.5 ,0.  ,0.  ,0.  ,0.36,0.  ,0.  ,0.  ,0.5 ,0.  ,0.  ,0.  ,
#  0.24,0.37,0.  ,0.  ,0.49,0.  ,0.  ,0.  ,0.24,0.  ,0.37,0.  ,0.43,0.2 ,
#  0.21,0.  ,0.24,0.  ,0.  ,0.37,0.28,0.38,0.  ,0.  ,0.29,0.  ,0.  ,0.  ,
#  0.37,0.46,0.  ,0.  ,0.24,0.  ,0.21,0.  ,0.  ,0.37,0.63,0.  ,0.24,0.  ,
#  0.  ,0.  ,0.21,0.  ,0.38,0.59,0.55,0.  ,0.  ,0.  ,0.  ,0.  ,0.22,0.36,
#  0.55,0.  ,0.53,0.  ,0.  ,0.  ,0.  ,0.  ,0.41,0.55,0.  ,0.  ,0.53,0.  ,
#  0.  ,0.  ,0.  ,0.4 ,0.54,0.  ,0.  ,0.  ,0.53,0.  ,0.  ,0.  ,0.4 ,0.53,
#  0.  ,0.  ,0.  ,0.  ,0.53,0.  ,0.  ,0.41,0.53,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.52,0.  ,0.45,0.53,0.  ,0.  ,0.  ,0.  ,0.  ,0.38,0.68,0.53,0.53,0.  ,
#  0.  ,0.  ,0.28,0.38,0.  ,0.  ,0.73,0.54,0.  ,0.  ,0.27,0.  ,0.27,0.  ,
#  0.  ,0.65,0.54,0.  ,0.  ,0.38,0.28,0.  ,0.  ,0.  ,0.61,0.55,0.  ,0.39,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.68,0.46,0.44,0.  ,0.  ,0.  ,0.  ,0.  ,0.42,
#  0.7 ,0.62,0.  ,0.  ,0.  ,0.  ,0.  ,0.42,0.  ,0.69,0.6 ,0.  ,0.  ,0.  ,
#  0.  ,0.42,0.  ,0.  ,0.69,0.64,0.  ,0.  ,0.  ,0.42,0.  ,0.  ,0.  ,0.68,
#  0.71,0.  ,0.  ,0.42,0.  ,0.  ,0.  ,0.  ,0.68,0.81,0.  ,0.42,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.67,0.77,0.64,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.67,0.75,
#  0.  ,0.56,0.  ,0.  ,0.  ,0.  ,0.  ,0.67,0.76,0.  ,0.  ,0.56,0.  ,0.  ,
#  0.  ,0.  ,0.66,0.75,0.  ,0.  ,0.  ,0.56,0.  ,0.  ,0.  ,0.66,0.75,0.  ,
#  0.  ,0.  ,0.  ,0.56,0.  ,0.  ,0.66,0.75,0.  ,0.  ,0.  ,0.  ,0.  ,0.56,
#  0.  ,0.67,0.75,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.73,0.64,0.75,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.46,0.  ,0.91,0.75,0.  ,0.  ,0.  ,0.  ,0.46,0.  ,0.  ,
#  0.88,0.75,0.  ,0.  ,0.  ,0.46,0.  ,0.  ,0.  ,0.85,0.75,0.  ,0.  ,0.46,
#  0.  ,0.  ,0.  ,0.  ,0.83,0.76,0.  ,0.46,0.  ,0.  ,0.  ,0.  ,0.  ,0.82,
#  0.72,0.47,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.82,0.82,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.83,0.84,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.86,0.91,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.89,1.  ,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.94,1.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.97,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,
#  1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.  ,1.  ,0.96,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,
#  0.91,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.97,0.87,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.88,0.83,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.81,0.81,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.8 ,0.79,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.85,0.79,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.94,0.8 ,0.  ,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.96,0.33,0.82,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.93,0.  ,0.85,0.  ,0.  ,0.  ,0.  ,0.  ,0.55,0.  ,0.89,0.73,0.44,0.  ,
#  0.  ,0.  ,0.55,0.  ,0.  ,0.84,0.76,0.  ,0.41,0.  ,0.54,0.  ,0.  ,0.  ,
#  0.79,0.74,0.  ,0.  ,0.59,0.  ,0.31,0.  ,0.  ,0.74,0.73,0.  ,0.49,0.  ,
#  0.37,0.  ,0.37,0.  ,0.71,0.73,0.5 ,0.  ,0.  ,0.  ,0.31,0.  ,0.38,0.67,
#  0.71,0.  ,0.  ,0.  ,0.  ,0.  ,0.22,0.37,0.63,0.67,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.38,0.  ,0.63,0.71,0.  ,0.  ,0.  ,0.  ,0.38,0.  ,0.  ,0.64,0.49,
#  0.57,0.  ,0.26,0.36,0.  ,0.  ,0.  ,0.65,0.56,0.  ,0.5 ,0.  ,0.26,0.  ,
#  0.  ,0.  ,0.66,0.54,0.38,0.  ,0.45,0.  ,0.  ,0.  ,0.  ,0.67,0.55,0.  ,
#  0.39,0.  ,0.37,0.  ,0.  ,0.  ,0.67,0.52,0.  ,0.  ,0.37,0.51,0.  ,0.  ,
#  0.  ,0.68,0.51,0.  ,0.  ,0.  ,0.  ,0.53,0.  ,0.  ,0.69,0.5 ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.53,0.  ,0.7 ,0.49,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.53,
#  0.72,0.49,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.72,0.5 ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.  ,0.56,0.51,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.51,0.34,
#  0.53,0.  ,0.  ,0.  ,0.  ,0.  ,0.5 ,0.  ,0.36,0.55,0.  ,0.  ,0.  ,0.  ,
#  0.5 ,0.  ,0.  ,0.36,0.56,0.  ,0.  ,0.  ,0.5 ,0.  ,0.  ,0.  ,0.36,0.37,
#  0.42,0.  ,0.5 ,0.  ,0.  ,0.  ,0.  ,0.37,0.5 ,0.  ,0.5 ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.37,0.37,0.53,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.38,0.  ,0.53,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.39,0.  ,0.  ,0.53,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.4 ,0.  ,0.  ,0.  ,0.53,0.  ,0.  ,0.  ,0.  ,0.41,0.  ,0.  ,0.  ,
#  0.  ,0.53,0.  ,0.  ,0.  ,0.43,0.  ,0.  ,0.  ,0.  ,0.  ,0.53,0.  ,0.  ,
#  0.42,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.53,0.  ,0.41,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.51,0.35,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.33,0.34,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.36,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.21,0.22])
# topology_densities = np.array([0.  ,0.01,0.  ,0.  ,0.  ,1.  ,0.81,0.  ,0.01,0.01,0.08,1.  ,1.  ,0.  ,
#  0.  ,0.  ,0.  ,0.85,1.  ,1.  ,0.  ,0.  ,0.01,0.  ,1.  ,1.  ,1.  ,0.  ,
#  0.  ,0.14,1.  ,0.8 ,0.83,1.  ,0.  ,0.  ,0.15,1.  ,0.96,0.  ,1.  ,0.  ,
#  0.  ,1.  ,1.  ,0.88,0.25,1.  ,0.  ,0.  ,1.  ,1.  ,0.  ,1.  ,1.  ,0.  ,
#  0.  ,1.  ,0.91,0.  ,0.96,1.  ,0.  ,0.  ,1.  ,0.89,0.  ,0.01,1.  ,0.87,
#  1.  ,1.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,0.01,0.  ,0.01,1.  ,1.  ,
#  1.  ,1.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,0.01,0.  ,0.01,1.  ,1.  ,
#  1.  ,1.  ,0.01,0.  ,0.01,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,1.  ,1.  ,
#  1.  ,1.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,0.01,0.  ,0.01,1.  ,1.  ,
#  1.  ,1.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,0.01,0.  ,0.01,1.  ,1.  ,
#  1.  ,1.  ,0.  ,0.  ,0.01,1.  ,1.  ,0.99,1.  ,0.01,0.  ,0.  ,1.  ,1.  ,
#  0.  ,1.  ,0.8 ,0.  ,0.13,1.  ,0.88,0.  ,1.  ,0.94,0.  ,0.69,1.  ,0.01,
#  0.  ,1.  ,1.  ,0.  ,1.  ,1.  ,0.  ,0.  ,1.  ,1.  ,0.83,0.62,1.  ,0.  ,
#  0.  ,0.35,1.  ,0.86,0.  ,1.  ,0.  ,0.  ,0.1 ,1.  ,0.93,0.  ,1.  ,0.01,
#  0.  ,0.01,0.01,0.99,1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.96,1.  ,1.  ,0.  ,
#  0.  ,0.03,0.01,0.12,1.  ,1.  ,0.  ,0.  ,0.01,0.  ,0.  ,0.01,1.  ,0.84])
topology_densities[topology_densities == 0] = 1.e-3
independent_variable_component.add_output('topology_densities', topology_densities)
prob.model.add_subsystem('independent_variables', independent_variable_component, promotes=['*'])
# quadrotor_2d_prob.evaluate(x=topology_densities)

fea_comp = EvaluateFea(fea_object=quadrotor_2d_prob, simp_penalization_factor=3., filter_radius=height*3/num_pts_00_01)
prob.model.add_subsystem('fea', fea_comp, promotes=['*'])

prob.model.add_objective('weight')
prob.model.add_design_var('topology_densities', lower=1.e-4, upper=1.)
# prob.model.add_constraint('stress_constraint', upper=0.)
prob.model.add_constraint('strain_energy_constraint', upper=0.)

# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['maxiter'] = 6

prob.driver = driver = om.pyOptSparseDriver()
driver.options['optimizer'] = 'SNOPT'
# driver.opt_settings['Verify level'] = 0
# driver.opt_settings['Major iterations limit'] = 20
# driver.opt_settings['Minor iterations limit'] = 20
# driver.opt_settings['Iterations limit'] = 20
# driver.opt_settings['Major step limit'] = 2.0

# driver.opt_settings['Major feasibility tolerance'] = 1.0e-10
# driver.opt_settings['Major optimality tolerance'] =2.e-14

prob.setup()
prob.run_model()
# prob.check_totals('U', 'topology_densities')
prob.check_totals('strain_energy_constraint', 'topology_densities')
# prob.check_totals()
# prob.run_driver()

print(prob['topology_densities'])
solution_densities = prob['topology_densities']
quadrotor_2d_prob.plot_topology(solution_densities)

np.set_printoptions(threshold=sys.maxsize)
save_file_solution = np.array2string(solution_densities, precision=2, separator=',', suppress_small=True)
with open('solution_densities.txt', 'w') as f:
    f.write('Densities')
    f.write('\n')
    f.write(save_file_solution)