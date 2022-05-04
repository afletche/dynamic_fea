import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from dynamic_fea.fea import FEA
from dynamic_fea.material import IsotropicMaterial
from dynamic_fea.component import Component
from dynamic_fea.element import Element, QuadElement
from dynamic_fea.mesh import UnstructuredMesh
from dynamic_fea.design_representation import FERepresentation

from optimization_framework.optimization_problem import OptimizationProblem
from optimization_framework.optimizers.gradient_descent_optimizer import GradientDescentOptimizer
from optimization_framework.optimizers.finite_difference import finite_difference

from lsdo_kit.design.design_geometry.design_geometry import DesignGeometry
from lsdo_kit.simulation.mesh.mesh import Mesh

from vedo import Points, Plotter, LegendBox


FLX9895_DM = IsotropicMaterial(name='FLX9895_DM', E=11.e6, nu = 0.4, sigma_y=15.e6)
thickness = 0.01    # m, so 1 cm

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
# num_pts_00_10 = 4+1
# num_pts_00_10 = 27+1
num_pts_00_10 = 81
# num_pts_00_01 = 4+1
# num_pts_00_01 = 8+1
num_pts_00_01 = 24+1

num_pts_100_110 = np.ceil(2.5/16*(num_pts_00_10-1)+1).astype(int)
# num_pts_100_110 = 3
# num_pts_100_110 = 6
num_pts_100_101 = np.ceil(3/4*(num_pts_00_01-1)+1).astype(int)  # needs to match up with left panel
num_nodes_cut_out = num_pts_100_101
num_points_below_right_panel = num_pts_00_01 - num_pts_100_101

# NOTE: Only going to model half of the domain because of symmetry.
point_00 = np.array([0., 0.04, 0.])
point_01 = np.array([0., 0.08, 0.])
point_10 = np.array([0.16-0.025, 0.04, 0.])
point_11 = np.array([0.16-0.025, 0.08, 0.])

point_100 = np.array([0.16-0.025, 0.04+0.01, 0.])
point_101 = np.array([0.16-0.025, 0.08, 0.])
point_110 = np.array([0.16, 0.04+0.01, 0.])
point_111 = np.array([0.16, 0.08, 0.])

left_edge = np.linspace(point_00, point_01, num_pts_00_01)
right_edge = np.linspace(point_10, point_11, num_pts_00_01)
left_panel = np.linspace(left_edge, right_edge, num_pts_00_10)
num_left_panel_nodes = num_pts_00_10 * num_pts_00_01

smaller_left_edge = np.linspace(point_100, point_101, num_pts_100_101)
smaller_right_edge = np.linspace(point_110, point_111, num_pts_100_101)
right_panel = np.linspace(smaller_left_edge, smaller_right_edge, num_pts_100_110)

right_panel_remove_duplicate = right_panel[1:,:,:]
mesh_nodes = np.vstack((left_panel.reshape((-1, 3)), right_panel_remove_duplicate.reshape((-1, 3))))

# # Plot mesh
# vp = Plotter(axes=1)
# nodes1 = Points(mesh_nodes.reshape((-1,3)), r = 7, c = 'plum').legend('mesh')
# vp.show(nodes1, 'Mesh', at=0, viewup="z", interactive = True)

# For left panel
plane_stress_or_plane_strain = 'stress'
elements = []
for u in range(left_panel.shape[0]-1):
    for v in range(left_panel.shape[1]-1):
        node_00 = left_panel[u,v,:2]
        node_01 = left_panel[u,v+1,:2]
        node_10 = left_panel[u+1,v,:2]
        node_11 = left_panel[u+1,v+1,:2]
        nodes = np.array([node_00, node_01, node_10, node_11])

        node_00_index = u*left_panel.shape[1] + v
        node_01_index = u*left_panel.shape[1] + (v+1)
        node_10_index = (u+1)*left_panel.shape[1] + v
        node_11_index = (u+1)*left_panel.shape[1] + (v+1)
        node_map = np.array([node_00_index, node_01_index, node_10_index, node_11_index])

        elements.append(QuadElement(plane_stress_or_plane_strain, nodes, node_map, FLX9895_DM, thickness))

# for the interface
for v in range(right_panel.shape[1]-1):
    node_00 = left_panel[-1,v+num_points_below_right_panel,:2]
    node_01 = left_panel[-1,v+1+num_points_below_right_panel,:2]
    node_10 = right_panel[1,v,:2]
    node_11 = right_panel[1,v+1,:2]
    nodes = np.array([node_00, node_01, node_10, node_11])

    node_00_index = v + (num_left_panel_nodes - (num_nodes_cut_out))
    node_01_index = (v+1) + (num_left_panel_nodes - (num_nodes_cut_out))
    node_10_index = v + (num_left_panel_nodes)
    node_11_index = (v+1) + (num_left_panel_nodes)
    node_map = np.array([node_00_index, node_01_index, node_10_index, node_11_index])

    elements.append(QuadElement(plane_stress_or_plane_strain, nodes, node_map, FLX9895_DM, thickness))

# For rest of right panel
for u in np.arange(right_panel.shape[0]-2)+1:
    for v in range(right_panel.shape[1]-1):
        node_00 = right_panel[u,v,:2]
        node_01 = right_panel[u,v+1,:2]
        node_10 = right_panel[u+1,v,:2]
        node_11 = right_panel[u+1,v+1,:2]
        nodes = np.array([node_00, node_01, node_10, node_11])

        node_00_index = (u-1)*right_panel.shape[1] + v + num_left_panel_nodes
        node_01_index = (u-1)*right_panel.shape[1] + (v+1) + num_left_panel_nodes
        node_10_index = (u)*right_panel.shape[1] + v + num_left_panel_nodes
        node_11_index = (u)*right_panel.shape[1] + (v+1) + num_left_panel_nodes
        node_map = np.array([node_00_index, node_01_index, node_10_index, node_11_index])

        elements.append(QuadElement(plane_stress_or_plane_strain, nodes, node_map, FLX9895_DM, thickness))

mesh = UnstructuredMesh(name='compliant_mech_mesh', nodes=mesh_nodes[:,:2].reshape((-1, 2)), elements=elements)
mesh.set_material(material=FLX9895_DM)

compliant_mech_prob_bc = []

# symmetry bc
x_bounds = np.array([-0.01, 0.1601])
y_bounds = np.array([0.0399, 0.0401])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)
bc_node_indices = np.argwhere(mask)
for node in bc_node_indices[:,0]:
    compliant_mech_prob_bc.append((node, 1, 0))

# workiece x-bc
x_bounds = np.array([0.134, 0.136])
x_bounds = np.array([0.16-0.0251, 0.1601])
y_bounds = np.array([0.0399, 0.0501])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)
workpiece_bc_node_indices = np.argwhere(mask)
for node in workpiece_bc_node_indices[:,0]:
    compliant_mech_prob_bc.append((node, 0, 0))

# workpiece y-bc
x_bounds = np.array([0.16-0.0251, 0.1601])
# x_bounds = np.array([0.16-0.001, 0.1601])
y_bounds = np.array([0.0499, 0.0501])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)
workpiece_bc_node_indices = np.argwhere(mask)
for node in workpiece_bc_node_indices[:,0]:
    compliant_mech_prob_bc.append((node, 1, 0))


static_loads = []
x_bounds = np.array([-0.01, 0.0401])
x_bounds = np.array([-0.01, 0.0001])
y_bounds = np.array([0.0799, 0.0801])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)
gripping_node_indices = np.argwhere(mask)

# load_nodes = mesh.nodes[mask]
load_node_indices = np.argwhere(mask)
load = np.array([0., -1.e0])
load_per_node = load/load_node_indices.shape[0]
for node in load_node_indices[:,0]:
    static_loads.append((node, load_per_node))

''' Run dynamics'''
# for i, load_case in enumerate(load_cases):
compliant_mech_prob = FEA(mesh=mesh, boundary_conditions=compliant_mech_prob_bc)
compliant_mech_prob.apply_loads(static_loads)
# compliant_mech_prob.setup()

compliant_mech_optimization = OptimizationProblem()
steepest_descent_optimizer = GradientDescentOptimizer(alpha=1e-2)
compliant_mech_optimization.set_model(model=compliant_mech_prob)
compliant_mech_optimization.set_optimizer(steepest_descent_optimizer)
compliant_mech_optimization.setup()
x0 = np.ones(len(elements),)*0.9
# x0 = np.array([0.  ,0.  ,0.07,0.13,0.19,0.23,0.28,0.32,0.36,0.39,0.4 ,0.41,0.4 ,0.37,
#  0.32,0.23,0.09,0.  ,0.22,0.31,0.23,0.27,0.  ,1.  ,0.  ,0.05,0.11,0.17,
#  0.22,0.27,0.32,0.35,0.38,0.4 ,0.41,0.41,0.4 ,0.37,0.33,0.26,0.08,0.  ,
#  0.12,0.  ,0.  ,0.  ,1.  ,1.  ,0.  ,0.06,0.13,0.19,0.25,0.31,0.35,0.38,
#  0.4 ,0.42,0.42,0.4 ,0.37,0.32,0.27,0.27,0.19,0.  ,0.  ,0.  ,1.  ,1.  ,
#  0.  ,1.  ,0.  ,0.06,0.14,0.22,0.28,0.34,0.38,0.41,0.42,0.42,0.41,0.37,
#  0.28,0.13,0.  ,0.  ,0.  ,0.  ,0.02,1.  ,1.  ,0.  ,0.  ,1.  ,0.  ,0.06,
#  0.15,0.24,0.3 ,0.36,0.4 ,0.42,0.43,0.42,0.38,0.3 ,0.12,0.  ,0.  ,0.  ,
#  0.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,1.  ,0.  ,0.06,0.16,0.25,0.32,0.38,
#  0.41,0.44,0.44,0.41,0.34,0.19,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,0.  ,0.  ,1.  ,0.  ,0.07,0.17,0.27,0.34,0.39,0.43,0.44,0.43,0.39,
#  0.28,0.08,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,1.  ,1.  ,
#  0.  ,0.08,0.19,0.28,0.35,0.4 ,0.43,0.44,0.41,0.34,0.2 ,0.  ,0.  ,0.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,1.  ,1.  ,0.  ,0.09,0.19,0.29,
#  0.36,0.4 ,0.43,0.42,0.37,0.26,0.05,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,0.06,0.  ,0.  ,1.  ,1.  ,0.  ,0.07,0.18,0.28,0.36,0.4 ,0.41,0.38,
#  0.3 ,0.14,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.08,0.  ,0.  ,1.  ,
#  1.  ,1.  ,0.  ,0.  ,0.13,0.26,0.35,0.39,0.39,0.31,0.15,0.  ,0.  ,0.  ,
#  0.01,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.04,1.  ,1.  ,1.  ,0.  ,0.  ,
#  0.08,0.23,0.34,0.39,0.36,0.2 ,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.07,0.23,0.34,0.4 ,
#  0.35,0.05,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.02,1.  ,1.  ,1.  ,0.  ,0.  ,0.13,0.23,0.33,0.4 ,0.39,0.  ,0.  ,0.  ,
#  0.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,
#  0.  ,0.09,0.1 ,0.17,0.29,0.37,0.39,0.26,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.13,
#  0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.25,0.11,
#  0.  ,0.06,0.02,1.  ,1.  ,1.  ,0.  ,0.  ,0.07,0.  ,0.  ,0.14,0.6 ,0.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.37,0.38,0.15,0.  ,0.  ,0.  ,1.  ,
#  1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.92,0.41,1.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,0.  ,0.38,0.45,0.36,0.17,0.  ,0.13,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,
#  0.  ,0.  ,0.03,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.15,0.29,0.29,
#  0.26,0.16,0.  ,0.04,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.06,0.  ,0.02,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,0.03,0.04,0.  ,0.  ,0.  ,0.01,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,0.02,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,
#  0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,
#  0.05,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,0.  ,0.  ,0.35,1.  ,1.  ,1.  ,0.31,0.  ,0.  ,0.47,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.3 ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,0.68,0.41,0.45,0.92,1.  ,1.  ,1.  ,1.  ,0.  ,
#  0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.23,0.88,0.99,1.  ,1.  ,
#  1.  ,1.  ,0.93,0.75,0.58,0.43,0.14,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,0.  ,0.19,0.55,0.72,0.85,0.99,1.  ,1.  ,1.  ,1.  ,
#  1.  ,0.72,0.15,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,0.18,0.48,0.61,0.72,0.83,0.93,1.  ,1.  ,1.  ,1.  ,1.  ,0.62,0.  ,
#  0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.1 ,0.34,0.5 ,
#  0.62,0.72,0.79,0.86,0.94,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.17,0.37,0.47,0.57,0.65,0.71,0.75,
#  0.79,0.84,0.98,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,0.  ,0.22,0.38,0.46,0.53,0.6 ,0.66,0.7 ,0.71,0.7 ,0.66,0.62,
#  0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.18,
#  0.33,0.42,0.5 ,0.57,0.63,0.67,0.68,0.65,0.51,0.26,0.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.09,1.  ,0.  ,0.18,0.33,0.41,0.48,0.54,
#  0.6 ,0.64,0.66,0.63,0.45,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,0.01,0.51,0.68,0.  ,0.23,0.34,0.4 ,0.46,0.51,0.56,0.6 ,0.62,0.6 ,
#  0.38,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.29,0.58,0.68,
#  0.  ,0.22,0.32,0.38,0.43,0.48,0.52,0.55,0.56,0.54,0.38,0.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.49,0.63,0.63,0.  ,0.18,0.27,0.34,
#  0.4 ,0.44,0.48,0.5 ,0.5 ,0.41,0.23,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,0.  ,0.18,0.51,0.57,0.57,0.  ,0.15,0.24,0.31,0.37,0.4 ,0.42,0.43,
#  0.41,0.34,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.35,0.46,
#  0.51,0.52,0.  ,0.15,0.22,0.29,0.33,0.36,0.36,0.34,0.3 ,0.24,0.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.28,0.39,0.45,0.46,0.  ,0.14,
#  0.2 ,0.26,0.3 ,0.3 ,0.28,0.23,0.15,0.09,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,0.  ,0.  ,0.17,0.29,0.38,0.4 ,0.  ,0.12,0.16,0.22,0.25,0.24,
#  0.19,0.1 ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,
#  0.  ,0.17,0.3 ,0.34,0.  ,0.06,0.1 ,0.16,0.18,0.16,0.08,0.  ,0.  ,0.  ,
#  0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.05,0.22,0.27,
#  0.  ,0.  ,0.  ,0.07,0.08,0.04,0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.11,0.18,0.01,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.05,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.03,
#  0.03,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,
#  0.04,0.12,0.15,0.13,0.  ,0.  ,0.02,0.11,0.19,0.28,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.29,0.26,
#  0.  ,0.07,0.15,0.26,0.56,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.38,0.  ,0.15,0.27,0.69,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.21,0.52,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,0.  ,0.29,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.58,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.35,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,0.  ,0.03,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,
#  0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,
#  0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,0.18,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.28,0.47,0.48,0.23,
#  0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,0.  ,0.01,0.45,0.41,0.34,0.  ,0.  ,0.  ,0.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,0.  ,0.23,0.25,0.18,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.  ,
#  0.  ,0.  ,0.  ,0.5 ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.04,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.01,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.11,0.09,0.17,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.08,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,0.15,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,0.01,0.01,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,0.01,0.  ,0.49,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,0.  ,0.75,0.68,0.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,0.87,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,0.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,0.99,0.97,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,0.92])
steepest_descent_optimizer.set_initial_guess(x0)
# t_start = time.time()
# print(compliant_mech_optimization.evaluate_model(x0))
# print('time:', time.time() - t_start)
# # compliant_mech_prob.plot_topology(x0)
# compliant_mech_prob.evaluate_gradient(x0)
# exit()
compliant_mech_optimization.run(line_search='GFD', grad_norm_abs_tol=1.e-4, delta_x_abs_tol=1e-7, objective_penalty=1e-3, max_iter=15)
solution = compliant_mech_optimization.report(history=True)
compliant_mech_optimization.plot()

compliant_mech_prob.plot_topology(solution[0])

# np.set_printoptions(threshold=sys.maxsize)
# print(compliant_mech_prob.averaged_von_mises_stresses > compliant_mech_prob.mesh.material.sigma_y)
# print(compliant_mech_prob.averaged_von_mises_stresses)

np.set_printoptions(threshold=sys.maxsize)
save_file_solution = np.array2string(solution[0], precision=2, separator=',', suppress_small=True)

with open('solution_densities.txt', 'w') as f:
    f.write('Densities')
    f.write('\n')
    f.write(save_file_solution)


'''
# Debugging
def evaluate_fd(x0):
    compliant_mech_prob.evaluate(x0)

    f = -compliant_mech_prob.F_p[-1]/100
    

    # return [sim1.theta[-1]]
    return f
    # return [sim1.ydotdot[-1]]
    # return [sim1.xdotdot[-1]]
    # print(lagrange_multipliers.dot(c))
    # return [lagrange_multipliers.dot(c)]


x0 = np.ones(len(elements),)*0.9
# x0[2] = 100
# sim1.simulate_dynamics(x0[:-1])
# sim1.lagrange_multipliers = x0[-1]

print('FD', finite_difference(evaluate_fd, x0, h=1.e-3))
print('ANALYTIC', compliant_mech_prob.evaluate_gradient(x0).todense())
# print('JAC', sim1.evaluate(x0)[3])


# control_optimization = OptimizationProblem()
# steepest_descent_optimizer = GradientDescentOptimizer(alpha=1e-3)
# control_optimization.set_model(model=sim1)
# control_optimization.set_optimizer(steepest_descent_optimizer)
# control_optimization.setup()
# x0 = np.ones(sim1.num_control_inputs + sim1.num_constraints,)*200.
# steepest_descent_optimizer.set_initial_guess(x0)
# print("model outputs",control_optimization.evaluate_model(x0))
'''