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

from dynamic_fea.examples.quadcopter_2d.evaluate_dynamic_fea import EvaluateDynamicFea

# from optimization_framework.optimization_problem import OptimizationProblem
# from optimization_framework.optimizers.gradient_descent_optimizer import GradientDescentOptimizer
# from optimization_framework.optimizers.finite_difference import finite_difference

# from vedo import Points, Plotter, LegendBox

import sys


nylon = IsotropicMaterial(name='nylon', E=2.7e9, nu = 0.39, density=1150., sigma_y=40e6)
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
num_pts_00_10 = 21
# num_pts_00_10 = 33
# num_pts_00_01 = 11
# num_pts_00_01 = 8
# num_pts_00_01 = 6
num_pts_00_01 = 5


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
mass_node_indices = np.argwhere(mask)
mass = rotor_mass
mass_per_node = mass/mass_node_indices.shape[0]
for node in mass_node_indices[:,0]:
    left_motor_point_mass.append((node, mass_per_node))

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

# mass_nodes = mesh.nodes[mask]
mass_node_indices = np.argwhere(mask)
mass = rotor_mass
mass_per_node = mass/mass_node_indices.shape[0]
for node in mass_node_indices[:,0]:
    right_motor_point_mass.append((node, mass_per_node))

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

# mass_nodes = mesh.nodes[mask]
mass_node_indices = np.argwhere(mask)
mass = battery_mass
mass_per_node = mass/mass_node_indices.shape[0]
for node in mass_node_indices[:,0]:
    battery_point_mass.append((node, mass_per_node))

mesh.add_point_masses(left_motor_point_mass)
mesh.add_point_masses(right_motor_point_mass)
mesh.add_point_masses(battery_point_mass)


''' Boundary Conditions '''
quadrotor_2d_prob_bc = []
x_bounds = np.array([length/2 - 0.0001, length/2 + 0.0001])
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


t0 = 0.
# tf = 0.005
# tf = 4.e-3
tf = 1.e-5
# nt_eval = 10000
nt_eval = 6001
# nt_eval = 40001
# nt_eval = 100
# nt_eval = 100

# tf = 10.
# nt = np.rint((tf-t0)*100).astype(int) # To see displacement
# nt = np.rint((tf-t0)*1000).astype(int) # To see displacement
# nt = np.rint((tf-t0)*1000000).astype(int)      # to see stresses (time step must be small too)
# t_eval = np.linspace(t0, tf, nt+1)
# dynamic_loads = []
# input_loads = [[np.array([0., 10.01e2]), np.array([0., 6.e0]), np.array([0., 20.e0]), np.array([0., 6.e0]), np.array([0., 6.01e0])],
#                 [np.array([0., 10.e2]), np.array([0., 6.01e0]), np.array([0., 20.e0]), np.array([0., 6.01e0]), np.array([0., 6.e0])]]
# from dynamic_fea.io.import_loads import import_loads
# dynamic_loads, t_eval = import_loads(mesh=mesh, file_path='quadrotor_input_profile.json')
input_loads = [[np.array([0., 30.])],
                [np.array([0., 50.])]]

nt_loads = len(input_loads[0])
# t_loads = [0., tf/4, tf/2, tf*3/4]
t_loads = np.linspace(t0, tf*(nt_loads-1)/nt_loads, nt_loads)
t_eval = np.linspace(t0, tf, nt_eval)


dynamic_loads = []
for i, t in enumerate(t_loads):
    # t = time[i]
    dynamic_loads_per_time = []
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
    load = input_loads[0][i]
    load_per_node = load/load_node_indices.shape[0]
    for node in load_node_indices[:,0]:
        dynamic_loads_per_time.append((node, load_per_node))

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
    load = input_loads[1][i]
    load_per_node = load/load_node_indices.shape[0]
    for node in load_node_indices[:,0]:
        dynamic_loads_per_time.append((node, load_per_node))

    dynamic_loads.append([t, dynamic_loads_per_time])


''' Run dynamics'''
quadrotor_2d_prob = FEA(mesh=mesh, boundary_conditions=quadrotor_2d_prob_bc)
quadrotor_2d_prob.setup_dynamics()
# quadrotor_2d_prob.apply_self_weight(g=9.81)
quadrotor_2d_prob.first_time = True
# topology_densities = np.ones((quadrotor_2d_prob.num_elements,))*0.5
# # quadrotor_2d_prob.evaluate_topology_dynamic(x=topology_densities, simp_penalization_factor=3, ramp_penalization_factor=None, filter_radius=.1)
# quadrotor_2d_prob.evaluate_dynamics(loads=dynamic_loads, t_eval=t_eval)
# # U_reshaped = quadrotor_2d_prob.U_per_dim_per_time
# stresses = quadrotor_2d_prob.evaluate_stresses()
# # quadrotor_2d_prob.plot(stress_type='xx', time_step=-1)
# # quadrotor_2d_prob.plot(stress_type='yy', time_step=-1)
# # quadrotor_2d_prob.plot(stress_type='xy', time_step=-1)
# # quadrotor_2d_prob.plot(stress_type='yy', dof=-1)
# quadrotor_2d_prob.plot(dof=[-1, -3], show_dislpacements=True)
# # quadrotor_2d_prob.plot(dof=[-1, -3], show_dislpacements=True, stress_type='yy')
# # quadrotor_2d_prob.plot(stress_type='vm', time_step=-1)
# quadrotor_2d_prob.plot(stress_type='avm', time_step=-1)
# # quadrotor_2d_prob.plot(stress_type='vm', dof=-1)
# # quadrotor_2d_prob.plot(stress_type='avm', dof=-1)
# quadrotor_2d_prob.plot(show_dislpacements=True, show_connections=False, stress_type='xx', video_file_name='displacement_animation.avi', video_fps=10, show=False)
# # quadrotor_2d_prob.plot_rigid_body_displacement()
# # quadrotor_2d_prob.plot_rigid_body_displacement(x_axis='t', y_axis='y')
# # quadrotor_2d_prob.plot_rigid_body_displacement(x_axis='t', y_axis='x')
# # quadrotor_2d_prob.plot_rigid_body_displacement(x_axis='t', y_axis='rot_z')

prob = om.Problem()

independent_variable_component = om.IndepVarComp()
# topology_densities = np.ones((quadrotor_2d_prob.num_elements,))*0.9
topology_densities = np.ones((quadrotor_2d_prob.num_elements,))*9.e-1
# topology_densities = np.random.rand(quadrotor_2d_prob.num_elements)

independent_variable_component.add_output('topology_densities', topology_densities)
prob.model.add_subsystem('independent_variables', independent_variable_component, promotes=['*'])

fea_comp = EvaluateDynamicFea(fea_object=quadrotor_2d_prob, simp_penalization_factor=3., filter_radius=height*3/num_pts_00_01, loads=dynamic_loads, t_eval=t_eval)
prob.model.add_subsystem('fea', fea_comp, promotes=['*'])

prob.model.add_objective('weight')
prob.model.add_design_var('topology_densities', lower=9.e-2, upper=1.)
# prob.model.add_constraint('stress_constraint', upper=0.)
# prob.model.add_constraint('strain_energy_constraint', upper=0.)
prob.model.add_constraint('strain_energy_constraint', equals=0.)

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
quadrotor_2d_prob.U = np.zeros((quadrotor_2d_prob.num_total_dof, quadrotor_2d_prob.nt+1))
quadrotor_2d_prob.U[quadrotor_2d_prob.free_dof,:] = quadrotor_2d_prob.U_discrete[:quadrotor_2d_prob.num_free_dof,:]
quadrotor_2d_prob.U_per_dim_per_time = quadrotor_2d_prob.U.reshape((quadrotor_2d_prob.num_nodes, quadrotor_2d_prob.num_dimensions, quadrotor_2d_prob.nt+1))
quadrotor_2d_prob.U_per_dim_per_time = np.moveaxis(quadrotor_2d_prob.U_per_dim_per_time, -1, 0)
# plot_densities = np.array([1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,0.4 ,0.1 ,0.62,1.  ,0.1 ,0.1 ,0.1 ,0.1 ,1.  ,0.1 ,0.1 ,
#  0.1 ,0.1 ,1.  ,0.1 ,0.1 ,0.1 ,0.1 ,1.  ,0.1 ,0.1 ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,0.1 ,0.1 ,0.34,1.  ,0.1 ,0.1 ,0.1 ,1.  ,1.  ,1.  ,0.1 ,0.1 ,0.1 ,
#  0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,
#  0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,
#  0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,1.  ,1.  ,1.  ,
#  0.1 ,0.1 ,0.34,1.  ,0.1 ,0.1 ,0.1 ,1.  ,1.  ,1.  ,0.1 ,0.1 ,1.  ,0.1 ,
#  0.1 ,1.  ,1.  ,1.  ,0.1 ,0.1 ,0.1 ,0.1 ,1.  ,0.1 ,0.1 ,0.1 ,0.1 ,1.  ,
#  0.1 ,0.1 ,0.1 ,0.1 ,1.  ,1.  ,0.4 ,0.1 ,0.62,1.  ,1.  ,1.  ,1.  ,1.  ,
#  1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ,1.  ])
# plot_densities = np.array([1.,         1.,         1.,         1.,         1.,         0.09,
#  1.,         0.09,       1.,         0.09,       0.09,       1.,
#  1.,         0.79040001, 0.79755981, 1.,         1.,         0.09,
#  0.09,       1.,         1.,         0.80762359, 0.8092114,  1.,
#  1.,         0.09,       0.09,       1.,         1.,         0.09,
#  0.09,       1.,         1.,         0.60667533, 0.59208721, 1.,
#  1.,         0.09,       0.09,       1.,         1.,         1.,
#  1.,         1.,         1.,         1.,         1.,         1.,
#  1.,         0.8171343,  0.93722681, 1.,         1.,         0.09,
#  0.09,       1.,         1.,         0.09,       0.93450011, 1.,
#  1.,         1.,         0.09,       1.,         1.,         0.09,
#  1.,         1.,         1.,         1.,         0.09,       1.,
#  1.,         1.,         1.,         1.,         1.,         1.,
#  1.,         1.        ])

# quadrotor_2d_prob.plot_topology(plot_densities)

# prob.check_totals('dummy_out', 'dummy_in')
# prob.check_totals()
# prob.check_totals('strain_energy_constraint', 'topology_densities')
# prob.check_totals()
prob.run_driver()

print(prob['topology_densities'])
solution_densities = prob['topology_densities']

quadrotor_2d_prob.evaluate_topology_dynamic(x=solution_densities)
quadrotor_2d_prob.setup_dynamics()
quadrotor_2d_prob.evaluate_dynamics(dynamic_loads, t_eval)

quadrotor_2d_prob.plot_topology(solution_densities)

np.set_printoptions(threshold=sys.maxsize)
save_file_solution = np.array2string(solution_densities, precision=2, separator=',', suppress_small=True)
with open('solution_densities.txt', 'w') as f:
    f.write('Densities')
    f.write('\n')
    f.write(save_file_solution)