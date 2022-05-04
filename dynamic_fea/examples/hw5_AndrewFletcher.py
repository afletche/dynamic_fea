'''
Script for AE498 HW5

Author: Andrew Fletcher
'''

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
# from vedo import Points, Plotter, LegendBox

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

'''PROBLEM 1'''
''' Problem Definition '''
hw5_mat = IsotropicMaterial(name='hw5_mat', E=75.e6, nu = 0.3)
thickness = 0.1    # m, so 1 cm


# USING COARSE MESH SO FD IS FEASIBLE

# num_pts_00_10 = 120+1
# num_pts_00_10 = 12+1
num_pts_00_10 = 48+1
# num_pts_00_01 = 20+1
# num_pts_00_01 = 2+1
num_pts_00_01 = 8+1
point_00 = np.array([0., 0., 0.])
point_01 = np.array([0., 2., 0.])
point_10 = np.array([12., 0., 0.])
point_11 = np.array([12., 2., 0.])
left_edge = np.linspace(point_00, point_01, num_pts_00_01)
right_edge = np.linspace(point_10, point_11, num_pts_00_01)
nodes_struct = np.linspace(left_edge, right_edge, num_pts_00_10)
num_nodes = num_pts_00_10 * num_pts_00_01

# # Plot mesh
# vp = Plotter(axes=1)
# nodes1 = Points(nodes_struct.reshape((-1,3)), r = 7, c = 'plum').legend('mesh')
# vp.show(nodes1, 'Mesh', at=0, viewup="z", interactive = True)

# For left panel
plane_stress_or_plane_strain = 'stress'
elements = []
for u in range(nodes_struct.shape[0]-1):
    for v in range(nodes_struct.shape[1]-1):
        node_00 = nodes_struct[u,v,:2]
        node_01 = nodes_struct[u,v+1,:2]
        node_10 = nodes_struct[u+1,v,:2]
        node_11 = nodes_struct[u+1,v+1,:2]
        nodes = np.array([node_00, node_01, node_10, node_11])

        node_00_index = u*nodes_struct.shape[1] + v
        node_01_index = u*nodes_struct.shape[1] + (v+1)
        node_10_index = (u+1)*nodes_struct.shape[1] + v
        node_11_index = (u+1)*nodes_struct.shape[1] + (v+1)
        node_map = np.array([node_00_index, node_01_index, node_10_index, node_11_index])

        elements.append(QuadElement(plane_stress_or_plane_strain, nodes, node_map, hw5_mat, thickness))

mesh = UnstructuredMesh(name='hw5_mesh', nodes=nodes_struct[:,:,:2].reshape((-1, 2)), elements=elements)
mesh.set_material(material=hw5_mat)

hw5_prob_bc = []

# left wall x-bc: Doing these first to make the x-bc on the left wall the first boundary conditions
x_bounds = np.array([-0.001, 0.001])
y_bounds = np.array([-0.001, 2.001])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)
hw5_bc_node_indices = np.argwhere(mask)
for node in hw5_bc_node_indices[:,0]:
    hw5_prob_bc.append((node, 0, 0))
num_fixed_nodes_at_left = len(hw5_bc_node_indices[:,0])

# left wall y-bc
x_bounds = np.array([-0.001, 0.001])
y_bounds = np.array([-0.001, 2.001])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)
hw5_bc_node_indices = np.argwhere(mask)
for node in hw5_bc_node_indices[:,0]:
    hw5_prob_bc.append((node, 1, 0))

# right wall bc
x_bounds = np.array([11.999, 12.001])
y_bounds = np.array([-0.001, 2.001])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)
hw5_bc_node_indices = np.argwhere(mask)
for node in hw5_bc_node_indices[:,0]:
    hw5_prob_bc.append((node, 0, 0))
    hw5_prob_bc.append((node, 1, 0))


static_loads = []
x_bounds = np.array([6.-0.251, 6.+0.251])
y_bounds = np.array([1.999, 2.001])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)

load_node_indices = np.argwhere(mask)
load = np.array([0., -1.e3*0.5])
load_per_node = load/load_node_indices.shape[0]
for node in load_node_indices[:,0]:
    static_loads.append((node, load_per_node))


x_bounds = np.array([6.-0.001, 6.+0.001])
y_bounds = np.array([1.999, 2.001])
mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
mask_x = np.logical_and(mask_x1, mask_x2)
mask_y = np.logical_and(mask_y1, mask_y2)
mask = np.logical_and(mask_x, mask_y)
top_center_node_index = np.argwhere(mask)


# Setup FEA model
hw5_prob = FEA(mesh=mesh, boundary_conditions=hw5_prob_bc)
hw5_prob.apply_loads(static_loads)
hw5_prob.setup()


'''
Function for evaluating u_c and its gradient
'''
def evaluate_model_a(x, rho=0.):
    densities_too_high = x > 1
    x[densities_too_high] = 1.
    densities_too_low = x < 1.e-3
    x[densities_too_low] = 1.e-3

    SIMP_parameter = 3
    RAMP_parameter = 3

    hw5_prob.evaluate_topology(x, simp_penalization_factor=SIMP_parameter, ramp_penalization_factor=None, filter_radius=0.01)
    hw5_prob.evaluate_static()

    # Evaluate objective
    L = np.zeros((hw5_prob.num_total_dof))
    top_center_dof = top_center_node_index*2 + 1
    L[top_center_dof] = 1.
    f = -L.dot(hw5_prob.U)

    # Evaluate gradient
    adjoint_term = sps.lil_matrix((1, hw5_prob.num_total_dof))
    L = sps.lil_matrix((1, hw5_prob.num_total_dof))
    L[0, top_center_dof] = 1.
    L_f = L[0, np.ix_(hw5_prob.free_dof)]
    L_f = L_f.tocsc()
    adjoint_term_f_transpose = spsolve(hw5_prob.K_ff, -L_f.T)
    # adjoint_term_p = 0    # default is 0
    adjoint_term[0,np.ix_(hw5_prob.free_dof)] = adjoint_term_f_transpose.T
    # adjoint_term[0,np.ix_(hw5_prob.prescribed_dof)] = adjoint_term_p.T

    pR_px = sps.lil_matrix((hw5_prob.num_total_dof, hw5_prob.num_elements))
    for i, element in enumerate(hw5_prob.mesh.elements):
        K = sps.lil_matrix((hw5_prob.num_total_dof, hw5_prob.num_total_dof))
        element_sensitivity = element.K0*SIMP_parameter*x[i]**(SIMP_parameter-1)
        element_dofs = hw5_prob.mesh.nodes_to_dof_indices(element.node_map)
        K[np.ix_(element_dofs, element_dofs)] = element_sensitivity
        K = K.tocsc()
        pR_px[:, i] = K.dot(hw5_prob.U)

    pR_px = pR_px.tocsc()
    gradient = adjoint_term.dot(pR_px).todense()

    if hw5_prob.mesh.density_filter is not None:
        gradient = gradient.dot(hw5_prob.mesh.density_filter)

    # Annoying thing to make gradient 1D (because scipy.sparse bug)
    df_dx = np.zeros((len(x),))
    for i in range(len(x)):
        df_dx[i] = -gradient[0,i]

    # set boundary densities to 0 gradient
    df_dx[x == 1.] = 0.
    df_dx[x == 1.e-3] = 0.


    # Volume constraint (all elements equal volume so turning into mass constraint)
    total_mass_possible = hw5_prob.num_elements
    total_mass = np.sum(x)
    volume_constraint = total_mass/total_mass_possible - 0.4
    penalty = 1/2*rho*volume_constraint**2

    dpenalty_dx = rho*volume_constraint


    f = f + penalty
    c = np.array([volume_constraint])
    df_dx = df_dx + dpenalty_dx
    # df_dx = None
    dc_dx = dc_dx = np.array([])
    d2f_dx2 = None
    dl_dx = None
    kkt = None

    model_outputs = [f, c, df_dx, dc_dx, d2f_dx2, dl_dx, kkt]
    return model_outputs


'''
Function for evaluating P_bar and its gradient
'''
def evaluate_model_b(x):
    densities_too_high = x > 1
    x[densities_too_high] = 1.
    densities_too_low = x < 1.e-3
    x[densities_too_low] = 1.e-3

    SIMP_parameter = 3

    hw5_prob.evaluate_topology(x, simp_penalization_factor=SIMP_parameter)
    hw5_prob.evaluate_static()

    # Evaluate objective
    L = np.zeros((hw5_prob.num_total_dof))
    L[:num_fixed_nodes_at_left] = 1./num_fixed_nodes_at_left
    f = L.dot(hw5_prob.F.todense())

    # Evaluate gradient
    adjoint_term = sps.lil_matrix((1, hw5_prob.num_total_dof))
    L = sps.lil_matrix((1, hw5_prob.num_total_dof))
    L[0, :num_fixed_nodes_at_left] = 1./num_fixed_nodes_at_left
    L_p = L[0, np.ix_(hw5_prob.prescribed_dof)]
    L_p = L_p.tocsc()
    adjoint_term_f_transpose = spsolve(hw5_prob.K_ff, -(hw5_prob.K_fp).dot(L_p.T))
    adjoint_term_p = L_p
    adjoint_term[0,np.ix_(hw5_prob.free_dof)] = adjoint_term_f_transpose.T
    adjoint_term[0,np.ix_(hw5_prob.prescribed_dof)] = adjoint_term_p

    pR_px = sps.lil_matrix((hw5_prob.num_total_dof, hw5_prob.num_elements))
    for i, element in enumerate(hw5_prob.mesh.elements):
        K = sps.lil_matrix((hw5_prob.num_total_dof, hw5_prob.num_total_dof))
        element_sensitivity = element.K0*SIMP_parameter*x[i]**(SIMP_parameter-1)
        element_dofs = hw5_prob.mesh.nodes_to_dof_indices(element.node_map)
        K[np.ix_(element_dofs, element_dofs)] = element_sensitivity
        K = K.tocsc()
        pR_px[:, i] = K.dot(hw5_prob.U)

    pR_px = pR_px.tocsc()
    gradient = adjoint_term.dot(pR_px).todense()

    # Annoying thing to make gradient 1D (because scipy.sparse bug)
    df_dx = np.zeros((len(x),))
    for i in range(len(x)):
        df_dx[i] = gradient[0,i]

    # set boundary densities to 0 gradient
    df_dx[x == 1.] = 0.
    df_dx[x == 1.e-3] = 0.


    f = f
    c = np.array([])
    df_dx = df_dx
    # df_dx = None
    dc_dx = dc_dx = np.array([])
    d2f_dx2 = None
    dl_dx = None
    kkt = None

    model_outputs = [f, c, df_dx, dc_dx, d2f_dx2, dl_dx, kkt]
    return model_outputs


x0 = np.ones(len(elements),)*0.4
x0 = np.linspace(0.1, 0.9, len(elements))
model_outputs = evaluate_model_a(x=x0)

# np.set_printoptions(threshold=sys.maxsizek)
# print('FD', finite_difference(evaluate_model_a, x0, h=1.e-5))
# # np.set_printoptions(threshold=sys.maxsize)
# print('MODEL A ANALYTIC', evaluate_model_a(x0)[2])

# print('FD', finite_difference(evaluate_model_b, x0, h=1.e-5))
# # np.set_printoptions(threshold=sys.maxsize)
# print('MODEL B ANALYTIC', evaluate_model_b(x0)[2])



''' PROBLEM 2 '''

'''
Function for evaluating u_c and its gradient
'''
def evaluate_model_2a(x, rho=0.):
    densities_too_high = x > 1
    x[densities_too_high] = 1.
    densities_too_low = x < 1.e-3
    x[densities_too_low] = 1.e-3

    SIMP_parameter = 3
    RAMP_parameter = 3

    hw5_prob.evaluate_topology(x, simp_penalization_factor=SIMP_parameter, ramp_penalization_factor=None, filter_radius=0.15)
    hw5_prob.evaluate_static()

    # Evaluate objective
    L = np.zeros((hw5_prob.num_total_dof))
    top_center_dof = top_center_node_index*2 + 1
    L[top_center_dof] = 1.
    f = -L.dot(hw5_prob.U)

    # Evaluate gradient
    adjoint_term = sps.lil_matrix((1, hw5_prob.num_total_dof))
    L = sps.lil_matrix((1, hw5_prob.num_total_dof))
    L[0, top_center_dof] = 1.
    L_f = L[0, np.ix_(hw5_prob.free_dof)]
    L_f = L_f.tocsc()
    adjoint_term_f_transpose = spsolve(hw5_prob.K_ff, -L_f.T)
    # adjoint_term_p = 0    # default is 0
    adjoint_term[0,np.ix_(hw5_prob.free_dof)] = adjoint_term_f_transpose.T
    # adjoint_term[0,np.ix_(hw5_prob.prescribed_dof)] = adjoint_term_p.T

    pR_px = sps.lil_matrix((hw5_prob.num_total_dof, hw5_prob.num_elements))
    for i, element in enumerate(hw5_prob.mesh.elements):
        K = sps.lil_matrix((hw5_prob.num_total_dof, hw5_prob.num_total_dof))
        element_sensitivity = element.K0*SIMP_parameter*x[i]**(SIMP_parameter-1)
        element_dofs = hw5_prob.mesh.nodes_to_dof_indices(element.node_map)
        K[np.ix_(element_dofs, element_dofs)] = element_sensitivity
        K = K.tocsc()
        pR_px[:, i] = K.dot(hw5_prob.U)

    pR_px = pR_px.tocsc()
    gradient = adjoint_term.dot(pR_px).todense()

    if hw5_prob.mesh.density_filter is not None:
        gradient = gradient.dot(hw5_prob.mesh.density_filter)

    # Annoying thing to make gradient 1D (because scipy.sparse bug)
    df_dx = np.zeros((len(x),))
    for i in range(len(x)):
        df_dx[i] = -gradient[0,i]

    # set boundary densities to 0 gradient
    df_dx[x == 1.] = 0.
    df_dx[x == 1.e-3] = 0.


    # Volume constraint (all elements equal volume so turning into mass constraint)
    total_mass_possible = hw5_prob.num_elements
    total_mass = np.sum(x)
    volume_constraint = total_mass/total_mass_possible - 0.4
    penalty = 1/2*rho*volume_constraint**2

    dpenalty_dx = rho*volume_constraint

    f = f + penalty
    c = np.array([volume_constraint])
    df_dx = df_dx + dpenalty_dx
    # df_dx = None
    dc_dx = dc_dx = np.array([])
    d2f_dx2 = None
    dl_dx = None
    kkt = None

    model_outputs = [f, c, df_dx, dc_dx, d2f_dx2, dl_dx, kkt]
    return model_outputs



# Run optimization
hw5_optimization = OptimizationProblem()
steepest_descent_optimizer = GradientDescentOptimizer(alpha=1e-2)
hw5_optimization.set_model(model=evaluate_model_2a)
hw5_optimization.set_optimizer(steepest_descent_optimizer)
hw5_optimization.setup()
x0 = np.ones(len(elements),)*0.4
steepest_descent_optimizer.set_initial_guess(x0)
hw5_optimization.run(line_search='GFD', grad_norm_abs_tol=1.e-6, delta_x_abs_tol=1e-7, objective_penalty=1.e-3, updating_penalty=True, max_iter=500)
solution = hw5_optimization.report(history=True)
hw5_optimization.plot()

hw5_prob.plot_topology(solution[0])


def evaluate_model_2b(x, rho=0.):
    densities_too_high = x > 1
    x[densities_too_high] = 1.
    densities_too_low = x < 1.e-3
    x[densities_too_low] = 1.e-3

    SIMP_parameter = 3
    RAMP_parameter = 3

    hw5_prob.evaluate_topology(x, simp_penalization_factor=None, ramp_penalization_factor=RAMP_parameter, filter_radius=0.15)
    hw5_prob.evaluate_static()

    # Evaluate objective
    L = np.zeros((hw5_prob.num_total_dof))
    top_center_dof = top_center_node_index*2 + 1
    L[top_center_dof] = 1.
    f = -L.dot(hw5_prob.U)

    # Evaluate gradient
    adjoint_term = sps.lil_matrix((1, hw5_prob.num_total_dof))
    L = sps.lil_matrix((1, hw5_prob.num_total_dof))
    L[0, top_center_dof] = 1.
    L_f = L[0, np.ix_(hw5_prob.free_dof)]
    L_f = L_f.tocsc()
    adjoint_term_f_transpose = spsolve(hw5_prob.K_ff, -L_f.T)
    # adjoint_term_p = 0    # default is 0
    adjoint_term[0,np.ix_(hw5_prob.free_dof)] = adjoint_term_f_transpose.T
    # adjoint_term[0,np.ix_(hw5_prob.prescribed_dof)] = adjoint_term_p.T

    pR_px = sps.lil_matrix((hw5_prob.num_total_dof, hw5_prob.num_elements))
    for i, element in enumerate(hw5_prob.mesh.elements):
        K = sps.lil_matrix((hw5_prob.num_total_dof, hw5_prob.num_total_dof))
        element_sensitivity = element.K0*SIMP_parameter*x[i]**(SIMP_parameter-1)
        element_dofs = hw5_prob.mesh.nodes_to_dof_indices(element.node_map)
        K[np.ix_(element_dofs, element_dofs)] = element_sensitivity
        K = K.tocsc()
        pR_px[:, i] = K.dot(hw5_prob.U)

    pR_px = pR_px.tocsc()
    gradient = adjoint_term.dot(pR_px).todense()

    if hw5_prob.mesh.density_filter is not None:
        gradient = gradient.dot(hw5_prob.mesh.density_filter)

    # Annoying thing to make gradient 1D (because scipy.sparse bug)
    df_dx = np.zeros((len(x),))
    for i in range(len(x)):
        df_dx[i] = -gradient[0,i]

    # set boundary densities to 0 gradient
    df_dx[x == 1.] = 0.
    df_dx[x == 1.e-3] = 0.


    # Volume constraint (all elements equal volume so turning into mass constraint)
    total_mass_possible = hw5_prob.num_elements
    total_mass = np.sum(x)
    volume_constraint = total_mass/total_mass_possible - 0.4
    penalty = 1/2*rho*volume_constraint**2

    dpenalty_dx = rho*volume_constraint

    f = f + penalty
    c = np.array([volume_constraint])
    df_dx = df_dx + dpenalty_dx
    # df_dx = None
    dc_dx = dc_dx = np.array([])
    d2f_dx2 = None
    dl_dx = None
    kkt = None

    model_outputs = [f, c, df_dx, dc_dx, d2f_dx2, dl_dx, kkt]
    return model_outputs


'''
# Run optimization
hw5_optimization = OptimizationProblem()
steepest_descent_optimizer = GradientDescentOptimizer(alpha=1e-2)
hw5_optimization.set_model(model=hw5_prob)
hw5_optimization.set_optimizer(steepest_descent_optimizer)
hw5_optimization.setup()
x0 = np.ones(len(elements),)*0.9
steepest_descent_optimizer.set_initial_guess(x0)
# t_start = time.time()
# print(hw5_optimization.evaluate_model(x0))
# print('time:', time.time() - t_start)
# # hw5_prob.plot_topology(x0)
# hw5_prob.evaluate_gradient(x0)
# exit()
hw5_optimization.run(line_search='GFD', grad_norm_abs_tol=1.e-4, delta_x_abs_tol=1e-7, objective_penalty=1e-3, max_iter=15)
solution = hw5_optimization.report(history=True)
hw5_optimization.plot()

hw5_prob.plot_topology(solution[0])
'''