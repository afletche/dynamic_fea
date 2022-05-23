'''
Author: Andrew Fletcher
Date: 11/24

This class is for performing 2D structural FEA problems.
'''


from functools import total_ordering
from multiprocessing.dummy import connection
from pprint import pprint
from tkinter import Y
import numpy as np
import numpy.matlib as npmatlib
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time
import cv2
import os

from dynamic_fea.element import QuadElement, TrussElement

class FEA:

    '''
    Sets up the FEA problem.

    Inputs:
    -
    '''
    def __init__(self, name=None, mesh=None, loads=[], boundary_conditions=[]):
        self.name = name
        # self.set_material(self.components[0].properties['material'])
        self.set_mesh(mesh)
        
        # self.initialize_ode() Moved into set_mesh method
        self.applied_loads = []
        self.apply_loads(loads)
        self.apply_boundary_conditions(boundary_conditions)

        self.visualization_scaling_factor = 1.
        self.use_self_weight = False
        self.g = 9.81



    '''
    Specifies the material properties for the structure.

    Inputs:
    - material : Material object : the material object containing the material props
    '''
    def set_material(self, material):
        self.material = material
        self.E = self.material.E
        self.nu = self.material.nu


    # '''
    # Specifies the mesh for the FEA problem.
    # NOTE: DOESN'T WORK FOR AN UNSTRUCTURED MESH!!

    # Inputs:
    # - mesh : np.ndarray (nx,ny,nz,3),(nx,ny,3),(nx,3), or (nx,ny,2), etc. : the FEA mesh
    # '''
    # def set_mesh(self, mesh):
    #     self.mesh = mesh
    #     self.num_dimensions = len(mesh.shape())-1   #shape is a vector
        
    #     if self.num_dimensions >= 1:
    #         self.num_nodes_x = mesh.shape[0]
    #         self.num_elements_x = self.num_nodes_x - 1
    #     if self.num_dimensions >= 2:
    #         self.num_nodes_y = mesh.shape[1]
    #         self.num_elements_y = self.num_nodes_y - 1
    #     if self.num_dimensions >= 3:
    #         self.num_nodes_z = mesh.shape[2]
    #         self.num_elements_z = self.num_nodes_z - 1
    #     self.dimensions_spanned = mesh.shape[-1]    # sees if it's in R^1, R^2, or R^3
    #     # Could potentially add a check to see if all the z coordinates are the same value, etc.

    #     self.num_nodes = np.cumprod(mesh.shape[:-1])[-1] # takes product of dimensions
    #     self.num_elements = self.num_elements_x * self.num_elements_y * self.num_elements_z
        
    #     self.num_total_dof = self.num_nodes*self.dimensions_spanned
    #     self.total_dof = np.arange(self.num_total_dof)

    #     # Initializing to all degrees free
    #     self.prescribed_dof = np.array([])
    #     self.prescribed_displacement = np.array([])



    '''
    Specifies the mesh for the FEA problem.

    Inputs:
    - mesh : Mesh : the FEA mesh
    -   nodes : np.ndarray (num_nodes,3) : the mesh nodes
    -   element_nodes: np.ndarray (num_elements, num_nodes_per_element) : the node-element map
    '''
    def set_mesh(self, mesh):
        self.mesh = mesh
        self.num_dimensions = 2 # TODO hardcoded for now!

        self.num_nodes = self.mesh.nodes.shape[0]
        # self.num_elements = self.mesh.element_nodes.shape[0]  # Commented out because switching to element object method
        self.num_elements = len(self.mesh.elements)
        self.dimensions_spanned = mesh.nodes.shape[-1]    # sees if it's in R^1, R^2, or R^3

        self.num_total_dof = self.num_nodes*self.dimensions_spanned
        self.total_dof = np.arange(self.num_total_dof)

        # Initializing to all degrees free
        self.prescribed_dof = np.array([])
        self.prescribed_displacement = np.array([])

        self.initialize_ode()



    '''
    Adds loads onto the structure.

    Inputs:
    - loads : List : [(node1, force1), (node2, force2), ...]
    - nodes  : numpy.ndarray (nf,): The nodes where the forces are applied
    - forces : numpy.ndarray (nf,3): The forces applied to the corresponding nodes
    '''
    def apply_loads(self, loads):        
        for load in loads:
            if len(load) != 2:
                Exception("Please input a 'load' of the form (nodes, forces) into the loads List")
            node = load[0]
            force = load[1]
            for i in range(len(force)):
                if force[i] == 0:
                    continue
                load_component_dof = node*self.dimensions_spanned+i
                if self.F[load_component_dof] != 0:
                    self.F[load_component_dof] += np.array([force[i]])
                    self.F_body_fixed[load_component_dof] += np.array([force[i]])
                else:
                    self.F[load_component_dof] = force[i]
                    self.F_body_fixed[load_component_dof] = force[i]

        if loads != []:
            self.applied_loads = loads


    '''
    Applies self-weight loads onto the structure.

    Inputs:
    - g : float : gravitational constant (9.81 for Earth)
    '''
    def apply_self_weight(self, g, rotation=0):
        mass_matrix = self.M
        odd_indices = np.arange(1, self.num_total_dof, 2)
        mass_dof_array = mass_matrix[np.ix_(odd_indices, odd_indices)].diagonal()

        self.F[odd_indices] -= (mass_dof_array*g).reshape((-1, 1))
        self.F_earth_fixed[odd_indices] -= (mass_dof_array*g).reshape((-1, 1))

        self.use_self_weight = True
        self.g = g


    '''
    Adds a point mass to the structure.

    Inputs:
    - g : float : gravitational constant (9.81 for Earth)
    - masses : List : list of Lists of locations_node and masses: [[node1, mass1],[node2, mass2],...]
    '''
    def add_point_masses(self, masses):
        self.mesh.add_point_masses(masses)



    def reset_loads(self):
        self.F = sps.lil_matrix((self.num_total_dof, 1))
        self.F_earth_fixed = sps.lil_matrix((self.num_total_dof, 1))
        self.F_body_fixed = sps.lil_matrix((self.num_total_dof, 1))



    '''
    Applies the boundary conditions to the FEA problem.
    
    Inputs:
    - boundary_conditions : List : [(node1, axis1, displacement1), (node2, ...), ...]
    - nodes : np.ndarray (num_bc,): the nodes with the applied BCs
    - axis  : np.ndarray (num_bc,): the corresponding axis that is prescribed at each node
    - displacements : np.ndarray (num_bc,): the prescribed displacement for each dof
    '''
    def apply_boundary_conditions(self, boundary_conditions):
        if (len(boundary_conditions) != 2) and (len(boundary_conditions) != 3):
            Exception("Please input a 'load' of the form [nodes, axes, displacements]")
        
        for boundary_condition in boundary_conditions:
            node = boundary_condition[0]
            axis = boundary_condition[1]
            prescribed_dof = node*self.dimensions_spanned+axis

            if len(boundary_condition) == 3:
                displacement = boundary_condition[2]
            else:
                displacement = 0

            self.prescribed_dof = np.append(self.prescribed_dof, prescribed_dof).astype(int)
            # self.prescribed_displacement = np.append(self.prescribed_displacement, displacement)
            self.U[prescribed_dof] = displacement


    '''
    Sets up the data structures for the ODE (preallocates).
    '''
    def initialize_ode(self):
        self.K = sps.lil_matrix((self.num_total_dof, self.num_total_dof))
        self.F = sps.lil_matrix((self.num_total_dof, 1))
        self.F_earth_fixed = sps.lil_matrix((self.num_total_dof, 1))
        self.F_body_fixed = sps.lil_matrix((self.num_total_dof, 1))
        # self.U = sps.lil_matrix((self.num_total_dof, 1))
        self.U = np.zeros((self.num_total_dof, 1))

        self.stress_maps = {}
        self.integration_coordinates = {}
        self.num_quad_elements = 0
        self.num_truss_elements = 0
        self.num_stress_evals = 0
        self.num_stress_eval_points = 0
        for element in self.mesh.elements:
            if type(element) is QuadElement:
                self.num_quad_elements += 1
                num_stress_types = 3
                self.num_stress_evals += element.num_integration_points*num_stress_types
                self.num_stress_eval_points += element.num_integration_points
            elif type(element) is TrussElement:
                self.num_truss_elements += 1
                self.num_stress_evals += 1      # just an axial load
                self.num_stress_eval_points += 1    # only one value for the whole truss


        self.stress_map = sps.lil_matrix((self.num_stress_evals, self.num_total_dof))   # TODO Generalize for 3D non-quads
        self.stress_eval_points_map = sps.lil_matrix((self.num_stress_eval_points, self.num_nodes))   # TODO Generalize for 3D non-quads
        self.stress_eval_points = np.zeros((self.num_stress_eval_points, self.num_dimensions))



    '''
    Assembles the global stiffness matrix based on the local stiffness matrices (using element objects)
    '''
    def assemble_k(self):
        # In the future, this could potentially loop over meshes, or perhaps do it's own assembly of something.
        self.K = self.mesh.K

    def assemble_mass_matrix(self):
        self.M = self.mesh.M

    '''
    Gets the DOF indices for the corresponding nodes.
    '''
    def nodes_to_dof_indices(self, nodes):
        dof_indices = np.zeros(2*len(nodes))
        for i in range(len(nodes)):
            dof_indices[2*i] = self.dimensions_spanned*nodes[i]
            dof_indices[2*i+1] = self.dimensions_spanned*nodes[i]+1
        return dof_indices.astype(int)

    '''
    Assembles the connectivity (mostly for plotting).
    '''
    def assemble_connectivity(self):
        self.node_connectivity = self.mesh.node_connectivity

    '''
    Calculates the local load vector for an element.
    '''    
    def calc_f_local(self):
        # TODO Come up with a scheme to do this. (Rather difficult)
        #   Hardest part is deciding what the user should even supply.
        Exception("This has not been implemented yet!")
        pass


    '''
    Assembles the global stress map for the A matrix.
    '''
    def assemble_stress_map(self):
        # For each element, add its stress map
        self.stress_index_map = {'xx' : np.array([]).astype(int), 'yy' : np.array([]).astype(int), 'xy' : np.array([]).astype(int), 'axial' : np.array([]).astype(int)}
        self.stress_eval_points_indices = {'xx' : np.array([]).astype(int), 'yy' : np.array([]).astype(int), 'xy' : np.array([]).astype(int), 'axial' : np.array([]).astype(int)}

        start_index_stresses = 0
        end_index_stresses = 0
        start_index_eval_points = 0
        end_index_eval_points = 0
        num_quad_stress_points = 0
        for element in self.mesh.elements:
            local_stress_map = element.stress_map
            element_dofs = self.nodes_to_dof_indices(element.node_map)

            if type(element) is QuadElement:
                num_integration_points = element.num_integration_points # same as num eval points (these are gauss-quadrature int points)
                num_stress_types = 3    # xx, yy, xy so 3 calculated (maybe 4 if von mises is added)
                num_stress_evals = num_integration_points*num_stress_types
                end_index_stresses = start_index_stresses + num_stress_evals
                indices = np.arange(start_index_stresses, end_index_stresses)
                indices_for_one_stress = np.arange(start_index_stresses, end_index_stresses, num_stress_types).astype(int)
                self.stress_index_map['xx'] = np.append(self.stress_index_map['xx'], indices_for_one_stress)
                self.stress_index_map['yy'] = np.append(self.stress_index_map['yy'], indices_for_one_stress+1)
                self.stress_index_map['xy'] = np.append(self.stress_index_map['xy'], indices_for_one_stress+2)
                self.stress_map[np.ix_(indices, element_dofs)] = local_stress_map

                end_index_eval_points = start_index_eval_points + num_integration_points                
                indices = np.arange(start_index_eval_points, end_index_eval_points)
                self.stress_eval_points_indices['xx'] = np.append(self.stress_eval_points_indices['xx'], indices)
                self.stress_eval_points_indices['yy'] = np.append(self.stress_eval_points_indices['yy'], indices)
                self.stress_eval_points_indices['xy'] = np.append(self.stress_eval_points_indices['xy'], indices)
                self.stress_eval_points_map[np.ix_(indices, element.node_map)] = element.integration_coordinates_map

                num_quad_stress_points += num_integration_points
            elif type(element) is TrussElement:
                # num_eval_points = 1 # the one element between the 2 nodes.
                num_stress_types = 1

                self.stress_index_map['axial'] = np.append(self.stress_index_map['axial'], start_index_stresses)
                self.stress_map[np.ix_(np.array([start_index_stresses]), element_dofs)] = local_stress_map
                end_index_stresses = start_index_stresses + 1

                self.stress_eval_points_indices['axial'] = np.append(self.stress_eval_points_indices['axial'], start_index_eval_points)
                self.stress_eval_points_map[np.ix_(np.array([start_index_stresses]), element.node_map)] = element.midpoint_map
                end_index_eval_points = start_index_eval_points + 1

            start_index_stresses = end_index_stresses
            start_index_eval_points = end_index_eval_points

        # self.stress_averaging_map = np.zeros((int(num_quad_stress_points/4), num_quad_stress_points))
        # for i in range(int(num_quad_stress_points/4)):
        #     for j in np.arange(4*i, 4*(i+1)):
        #         self.stress_averaging_map[i,j] = 1/4


    # '''
    # Assembles a map that will individually calculate the strain energy in each element.
    # This has been commented out for now becomes the map is insanely large and may not even provide any real benefit.
    # '''
    # def assemble_strain_energy_per_element_map(self):
    #     self.strain_energy_per_element_map = self.mesh.strain_energy_per_element_map
    #     self.strain_energy_density_per_element_map = self.mesh.strain_energy_density_per_element_map


    '''
    Runs the preliminary setup to solve the problem.
    '''
    def setup(self):
        self.mesh.setup()

        self.assemble_k()
        self.assemble_mass_matrix()
        self.assemble_connectivity()
        self.assemble_stress_map()
        # self.assemble_strain_energy_per_element_map()
        self.K = self.K.tocsc()
        self.F = self.F.tocsc()
        self.F_earth_fixed = self.F_earth_fixed.tocsc()
        self.F_body_fixed = self.F_body_fixed.tocsc()
        self.stress_map = self.stress_map.tocsc()      

        self.free_dof = np.setdiff1d(self.total_dof, self.prescribed_dof)   # all dof that are not prescribed are free
        self.num_free_dof = self.free_dof.size
        self.num_prescribed_dof = self.prescribed_dof.size

        self.K_ff = self.K[np.ix_(self.free_dof, self.free_dof)]
        self.K_fp = self.K[np.ix_(self.free_dof, self.prescribed_dof)]
        self.K_pf = self.K_fp.T
        self.K_pp = self.K[np.ix_(self.prescribed_dof, self.prescribed_dof)]

        self.F_f = self.F[self.free_dof]
        self.F_p = self.F[self.prescribed_dof]    # don't know yet

        self.U_f = self.U[self.free_dof]          # don't know yet
        self.U_p = self.U[self.prescribed_dof]


    '''
    Solves the static FEA problem.
    '''
    def evaluate_static(self):
        self.nt = 0
        self.t_eval = [0]

        # self.U = np.linalg.solve(self.K[np.ix_(self.free_dof, self.free_dof)], self.F[self.free_dof])
        a = self.K_ff
        b = self.F_f - self.K_fp.dot(self.U_p)
        self.U_f = spsolve(a, b)
        self.F_p = self.K_pf.dot(self.U_f).reshape((self.K_pf.shape[0], 1)) + self.K_pp.dot(self.U_p)

        self.U[self.free_dof] = self.U_f.reshape((-1, 1))
        # self.U[self.prescribed_dof] = self.U_p    # known already
        # self.U = self.U.todense()                 # changed self.U to just be a numpy array initially 
        self.U = self.U.reshape((-1, self.nt+1))

        # self.F[self.free_dof] = self.F_f          # known already
        self.F[self.prescribed_dof] = self.F_p

        self.U_per_dim_per_time = self.U.reshape((self.nt+1, self.num_nodes, self.num_dimensions))


    '''
    Apply topology densities
    '''
    def evaluate_topology(self, x, simp_penalization_factor=3, ramp_penalization_factor=None, filter_radius=None):
        self.mesh.evaluate_topology(x, simp_penalization_factor=simp_penalization_factor, ramp_penalization_factor=ramp_penalization_factor, filter_radius=filter_radius)
        self.assemble_k()
        self.K_ff = self.K[np.ix_(self.free_dof, self.free_dof)]
        self.K_fp = self.K[np.ix_(self.free_dof, self.prescribed_dof)]
        self.K_pf = self.K_fp.T
        self.K_pp = self.K[np.ix_(self.prescribed_dof, self.prescribed_dof)]
        self.assemble_mass_matrix()
        self.assemble_stress_map()

        self.reset_loads()
        self.apply_loads(self.applied_loads)
        self.apply_self_weight(self.g)
        self.F_f = self.F[self.free_dof]
        self.F_p = self.F[self.prescribed_dof]    # don't know yet


    '''
    NOTE: This is hardcoded to take in density inputs for topology optimization.
    Objective: Weight
    Constraints:
        - Stress
        - Strain energy
    '''
    # def evaluate(self, x, rho=0.):
    #     # min_density = 1.e-2
    #     # max_density = 1.
    #     # densities_too_high = x > max_density
    #     # x[densities_too_high] = max_density
    #     # densities_too_low = x < min_density
    #     # x[densities_too_low] = min_density

    #     weight = x.dot(x)

    #     self.evaluate_topology(x, simp_penalization_factor=4, ramp_penalization_factor=None, filter_radius=None)
    #     # self.apply_self_weight(g=9.81)        This is currently reapplying self weight every time, so it becomes insanely heavy.

    #     self.evaluate_static()
    #     # self.evaluate_stresses()
    #     # self.evaluate_strain_energy()
    #     stress_constraint = self.evaluate_stress_constraint(x, eta=1., p=10., epsilon=1e-1)
    #     strain_energy_constraint = self.evaluate_strain_energy_constraint()

    #     # avm_stress_penalty_scaling_factor = 1.
    #     avm_stress_penalty_scaling_factor=0.
    #     strain_energy_penalty_scaling_factor = 1.
    #     # strain_energy_penalty_scaling_factor = 0.

    #     constraint_vector = np.array([])
    #     constraint_vector = np.append(constraint_vector, stress_constraint)
    #     constraint_vector = np.append(constraint_vector, strain_energy_constraint)
    #     penalty_scaling_factors_vector = np.array([])
    #     penalty_scaling_factors_vector = np.append(penalty_scaling_factors_vector, avm_stress_penalty_scaling_factor)
    #     penalty_scaling_factors_vector = np.append(penalty_scaling_factors_vector, strain_energy_penalty_scaling_factor)

    #     active_contraints = constraint_vector[constraint_vector > 0]
    #     active_penalty_scaling_factors = penalty_scaling_factors_vector[constraint_vector > 0]
    #     active_penalty_scaling_matrix = np.diag(active_penalty_scaling_factors)

    #     if active_contraints.size == 0:
    #         f = weight
    #     else:
    #         f = weight + 1/2*(active_contraints.T).dot(active_penalty_scaling_matrix).dot(active_contraints)
    #         # print('weight', weight)
    #         # print('penalty', 1/2*(active_contraints.T).dot(active_penalty_scaling_matrix).dot(active_contraints))
    #     f = strain_energy_constraint
    #     # f = self.F[-1,0]
    #     # print(self.strain_energy_constraint)
    #     c = np.array([])
    #     # df_dx = self.evaluate_gradient(x=x, rho=rho)
    #     df_dx = None
    #     self.h = 1e-8
    #     dc_dx = dc_dx = np.array([])
    #     d2f_dx2 = None
    #     dl_dx = None
    #     kkt = None

    #     model_outputs = [f, c, df_dx, dc_dx, d2f_dx2, dl_dx, kkt]
    #     return model_outputs

    def evaluate_stress_constraint(self, x, eta=1., p=10., epsilon=1.e-2):
        self.evaluate_stresses()

        SF = 2.
        sigma_yield = self.mesh.material.sigma_y/SF
        avm_stress_constraint_vector = ((x**eta)*self.averaged_von_mises_stresses/sigma_yield).reshape((-1,))
        aggregated_stress_constraint = (((avm_stress_constraint_vector**(p/2)).dot(avm_stress_constraint_vector**(p/2))))**(1/p) - (1 + epsilon)
        return aggregated_stress_constraint

    def evaluate_stress_constraint_gradient(self, x, eta=1., p=10., epsilon=1.e-2):
        stresses_vec = np.zeros((self.nt+1, 3, self.stresses_dict['xx'].shape[1]))
        stresses_vec[:,0,:] = self.stresses_dict['xx']
        stresses_vec[:,1,:] = self.stresses_dict['yy']
        stresses_vec[:,2,:] = self.stresses_dict['xy']

        von_mises_map = np.array([
            [1., -1/2, 0],
            [-1/2, 1., 0],
            [0., 0., 3.]
        ])
        m_dot_stresses = von_mises_map.dot(stresses_vec)
        
        von_mises_stresses = np.zeros((self.nt+1, self.stresses_dict['xx'].shape[1]))
        averaged_von_mises_stresses = np.zeros((self.nt+1, int(self.stresses_dict['xx'].shape[1]/4)))
        counter = 0
        averaged_stress_index = 0
        for i in range(self.stresses_dict['xx'].shape[1]):
            for j in range(self.nt+1):
                von_mises_stresses[j,i] = np.sqrt(stresses_vec[j,:,i].dot(m_dot_stresses[:,j,i]))
                averaged_von_mises_stresses[j,averaged_stress_index] += von_mises_stresses[j,i]/4

            counter += 1
            if counter == 4:
                averaged_stress_index += 1
                counter = 0

        self.von_mises_stresses = von_mises_stresses
        self.averaged_von_mises_stresses = averaged_von_mises_stresses

        if self.mesh.density_filter is not None:
            dc_dx = dc_dx.dot(self.mesh.density_filter)
        
        # return dc_dx

    
    def evaluate_strain_energy_constraint(self):
        self.evaluate_strain_energy()

        # print(self.strain_energy)

        odd_indices = np.arange(1, self.num_total_dof, 2)
        self.total_mass = self.M[np.ix_(odd_indices, odd_indices)].sum()

        self.max_strain_energy = 0.8418*self.total_mass**4 - 1.644*self.total_mass**3 + 1.244*self.total_mass**2 - 0.313*self.total_mass + 0.02661
        strain_energy_constraint = self.strain_energy - self.max_strain_energy
        # strain_energy_constraint = self.strain_energy - 0.11275
        strain_energy_constraint = np.max(strain_energy_constraint)

        return strain_energy_constraint

    def evaluate_strain_energy_gradient(self, x, simp_penalization_factor=3., ramp_penalization_factor=None, filter_radius=filter_radius):
        self.evaluate_topology(x=x, simp_penalization_factor=simp_penalization_factor, ramp_penalization_factor=ramp_penalization_factor, filter_radius=filter_radius)
        self.evaluate_static()

        pm_px = np.ones(len(x),)*self.mesh.material.density*self.mesh.elements[0].thickness*self.mesh.elements[0].area * simp_penalization_factor*x**(simp_penalization_factor-1)
        pc_px = (4*0.8418*self.total_mass**3 - 3*1.644*self.total_mass**2 + 2*1.244*self.total_mass - 0.313)*pm_px

        pR_px = sps.lil_matrix((self.num_free_dof, self.num_elements))
        for i, element in enumerate(self.mesh.elements):
            pK_pRho = sps.lil_matrix((self.num_total_dof, self.num_total_dof))
            element_sensitivity = element.K0*(simp_penalization_factor)*x[i]**(simp_penalization_factor-1)
            element_dofs = self.mesh.nodes_to_dof_indices(element.node_map)
            pK_pRho[np.ix_(element_dofs, element_dofs)] = element_sensitivity
            pKf_pRho = pK_pRho[np.ix_(self.free_dof, self.free_dof)]
            # pK_pRho = pK_pRho.tocsc()
            pKf_pRho = pKf_pRho.tocsc()
            pR_px[:, i] = pKf_pRho.dot(self.U_f)


        pF_px = np.zeros((self.num_total_dof, len(x)))
        for i, element in enumerate(self.mesh.elements):
            element_sensitivity = -element.density0*element.volume*self.g/4 * (simp_penalization_factor)*x[i]**(simp_penalization_factor-1)
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            vertical_dofs = element_dofs[np.array([1, 3, 5, 7])]
            pF_px[vertical_dofs, i] = element_sensitivity
        pFf_px = pF_px[np.ix_(self.free_dof), :].reshape((self.num_free_dof, len(x)))

        dCompliance_dx = self.U_f.dot(-pR_px/2 + pFf_px)
        # dCompliance_dx = np.dot(-self.U_f.T, pR_px.todense())
        dStrainEnergy_dx = dCompliance_dx
        dc_dx = dStrainEnergy_dx - pc_px
        # dc_dx = dStrainEnergy_dx

        if self.mesh.density_filter is not None:
            dc_dx = dc_dx.dot(self.mesh.density_filter)

        return dc_dx


    def evaluate_dynamic_strain_energy_gradient(self, x, simp_penalization_factor=3., ramp_penalization_factor=None, filter_radius=None, loads=[], t_eval=np.array([])):
        self.evaluate_topology_dynamic(x=x, simp_penalization_factor=simp_penalization_factor, ramp_penalization_factor=ramp_penalization_factor, filter_radius=filter_radius)
        self.evaluate_dynamics(loads, t_eval, t0=0, x0=None)

        pm_px = np.ones(len(x),)*self.mesh.material.density*self.mesh.elements[0].thickness*self.mesh.elements[0].area * simp_penalization_factor*x**(simp_penalization_factor-1)
        pc_px = (4*0.8418*self.total_mass**3 - 3*1.644*self.total_mass**2 + 2*1.244*self.total_mass - 0.313)*pm_px

        pR_px = sps.lil_matrix((self.num_free_dof, self.num_elements))
        for i, element in enumerate(self.mesh.elements):
            pK_pRho = sps.lil_matrix((self.num_total_dof, self.num_total_dof))
            element_sensitivity = element.K0*(simp_penalization_factor)*x[i]**(simp_penalization_factor-1)
            element_dofs = self.mesh.nodes_to_dof_indices(element.node_map)
            pK_pRho[np.ix_(element_dofs, element_dofs)] = element_sensitivity
            pKf_pRho = pK_pRho[np.ix_(self.free_dof, self.free_dof)]
            # pK_pRho = pK_pRho.tocsc()
            pKf_pRho = pKf_pRho.tocsc()
            pR_px[:, i] = pKf_pRho.dot(self.U_f)

        pF_px = np.zeros((self.num_total_dof, len(x)))
        for i, element in enumerate(self.mesh.elements):
            element_sensitivity = -element.density0*element.volume*self.g/4 * (simp_penalization_factor)*x[i]**(simp_penalization_factor-1)
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            vertical_dofs = element_dofs[np.array([1, 3, 5, 7])]
            pF_px[vertical_dofs, i] = element_sensitivity
        pFf_px = pF_px[np.ix_(self.free_dof), :].reshape((self.num_free_dof, len(x)))

        dCompliance_dx = self.U_f.dot(-pR_px/2 + pFf_px)
        # dCompliance_dx = np.dot(-self.U_f.T, pR_px.todense())
        dStrainEnergy_dx = dCompliance_dx
        dc_dx = dStrainEnergy_dx - pc_px
        # dc_dx = dStrainEnergy_dx

        if self.mesh.density_filter is not None:
            dc_dx = dc_dx.dot(self.mesh.density_filter)

        return dc_dx



    # def evaluate_gradient(self, x, rho=0.):
    #     gradient = 2*x

    #     # Annoying thing to make gradient 1D
    #     df_dx = np.zeros((len(x),))
    #     for i in range(len(x)):
    #         df_dx[i] = gradient[0,i]

    #     # set boundary densities to 0 gradient
    #     df_dx[x == 1.] = 0.
    #     df_dx[x == 1.e-3] = 0.


    #     # penalty portion
    #     SF = 2.
    #     sigma_yield = self.mesh.material.sigma_y/SF
    #     avm_stress_constraint = (self.averaged_von_mises_stresses/sigma_yield - 1).reshape((-1,))
    #     avm_stress_penalty_scaling_factors = np.ones_like(avm_stress_constraint)*1e-3
    #     # print('von mises', np.max(self.averaged_von_mises_stresses))

    #     constraint_vector = np.array([])
    #     constraint_vector = np.append(constraint_vector, avm_stress_constraint)
    #     penalty_scaling_factors_vector = np.array([])
    #     penalty_scaling_factors_vector = np.append(penalty_scaling_factors_vector, avm_stress_penalty_scaling_factors)

    #     active_contraints = constraint_vector[constraint_vector > 0]
    #     active_penalty_scaling_factors = penalty_scaling_factors_vector[constraint_vector > 0]
    #     gradient_penalty = np.zeros((len(x),))
    #     gradient_penalty[constraint_vector > 0] = active_penalty_scaling_factors*active_contraints

    #     if active_contraints.size != 0:
    #         df_dx += gradient_penalty

    #     return df_dx


    def evaluate_analytic_test(self, x, rho=0.):
        self.evaluate_topology(x=x)
        self.evaluate_static()
        
        gradient = 2*x

        # Annoying thing to make gradient 1D
        df_dx = np.zeros((len(x),))
        # for i in range(len(x)):
        #     df_dx[i] = gradient[0,i]
        df_dx = gradient

        # set boundary densities to 0 gradient
        df_dx[x == 1.] = 0.
        df_dx[x == 1.e-3] = 0.


        # penalty portion
        stress_constraint = self.evaluate_stress_constraint(x)      # TODO Probably need to feed in eta, p, etc.!!!!
        stress_constraint_gradient = self.evaluate_stress_constraint_gradient(x)
        strain_energy_constraint = self.evaluate_strain_energy_constraint()
        strain_energy_constraint_gradient = self.evaluate_strain_energy_gradient(x, simp_penalization_factor=4.)

        simp_penalization_factor=3.

        # pR_px = sps.lil_matrix((self.num_total_dof, self.num_elements))
        pR_px = sps.lil_matrix((self.num_free_dof, self.num_elements))
        for i, element in enumerate(self.mesh.elements):
            pK_pRho = sps.lil_matrix((self.num_total_dof, self.num_total_dof))
            element_sensitivity = element.K0*(simp_penalization_factor)*x[i]**(simp_penalization_factor-1)
            element_dofs = self.mesh.nodes_to_dof_indices(element.node_map)
            pK_pRho[np.ix_(element_dofs, element_dofs)] = element_sensitivity
            pKf_pRho = pK_pRho[np.ix_(self.free_dof, self.free_dof)]
            pK_pRho = pK_pRho.tocsc()
            pKf_pRho = pKf_pRho.tocsc()
            pR_px[:, i] = pKf_pRho.dot(self.U_f)

        # print(pR_px.todense())

        # du_dx = -spsolve(self.K_ff, pR_px).todense()
        # du_dx = -((sps.linalg.inv(self.K_ff).dot(self.F_f)).T.dot(pR_px)).todense()
        du_dx = -(self.U_f.dot(pR_px.todense()))/2
        du_dx_large = du_dx
        # du_dx_large = np.zeros((self.num_total_dof, self.num_elements))
        # du_dx_large[np.ix_(self.free_dof), :] = du_dx

        df_dx = strain_energy_constraint_gradient

        pF_px = np.zeros((self.num_total_dof, len(x)))
        # odd_indices = np.arange(1, self.num_total_dof, 2)
        # pF_px[odd_indices,:] = -self.mesh.material.density*self.mesh.elements[0].thickness*self.mesh.elements[0].area*self.g
        for i, element in enumerate(self.mesh.elements):
            element_sensitivity = -element.density0*element.volume*self.g/4 * (simp_penalization_factor)*x[i]**(simp_penalization_factor-1)
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            vertical_dofs = element_dofs[np.array([1, 3, 5, 7])]
            pF_px[vertical_dofs, i] = element_sensitivity

        pm_px = np.ones(len(x),)*self.mesh.material.density*self.mesh.elements[0].thickness*self.mesh.elements[0].area * simp_penalization_factor*x**(simp_penalization_factor-1)
        pc_px = (4*0.8418*self.total_mass**3 - 3*1.644*self.total_mass**2 + 2*1.244*self.total_mass - 0.313)*pm_px

        return pc_px
    

    
    def setup_dynamics(self):
        self.setup()
        self.M = self.M.tocsc()
        self.M_ff = self.M[np.ix_(self.free_dof, self.free_dof)]

        size_K_ff = self.K_ff.shape[0]
        self.num_dislpacement_dofs = size_K_ff
        self.num_dislpacement_states = 2*self.num_dislpacement_dofs
        self.num_rigid_body_dofs = 3     # 2D
        num_rigid_body_states = 2*self.num_rigid_body_dofs

        # TODO Move this block out of here. Make it a problem input.
        M_ff_inv = sps.linalg.inv(self.M_ff)
        self.dampening_per_node = 0.03      # TODO implement Proportional damping
        displacement_dampening = sps.eye(self.num_dislpacement_dofs, format='csc')*self.dampening_per_node
        

        self.num_states = self.num_dislpacement_states
        num_inputs = self.num_dislpacement_dofs     # each node is a location for a potential input

        # Constructing displacement portion of A matrix
        A_10 = -M_ff_inv.dot(self.K_ff)
        A_11 = -M_ff_inv.dot(displacement_dampening)
        A = sps.lil_matrix((self.num_states, self.num_states))
        A[:self.num_dislpacement_dofs, self.num_dislpacement_dofs:] = sps.eye(self.num_dislpacement_dofs, format='csc')
        A[self.num_dislpacement_dofs:, :self.num_dislpacement_dofs] = A_10
        A[self.num_dislpacement_dofs:, self.num_dislpacement_dofs:] = A_11
        A = A.tocsc()

        # print('Eigenvalues of A: ', sps.linalg.eigs(A))
        self.eigs = sps.linalg.eigs(A)
        self.A_inv = sps.linalg.inv(A)
        # print('inv A: ', sps.linalg.inv(A))   # made sure not singular.
        # print('det: ', np.linalg.det(A.todense()))
        # print('NORM: ', np.linalg.norm((A.dot(A_inv)).todense() - np.eye(A.shape[0])))

        B = sps.lil_matrix((self.num_states, num_inputs))
        B[self.num_dislpacement_dofs:,:] = M_ff_inv
        B = B.tocsc()

        C_dislacements = sps.lil_matrix((self.num_dislpacement_dofs, self.num_states))
        C_dislacements[:, :self.num_dislpacement_dofs] = sps.eye(self.num_dislpacement_dofs)
        C_dislacements = C_dislacements.tocsc()

        self.A = A
        self.B = B
        self.C_dislacements = C_dislacements

        # Maps from self.F*delta_t**2 to rigid body displacement. 
        self.rigid_body_translation_map = np.zeros((2, self.num_total_dof)) # 2 is the number of rigid body translation
        even_indices = np.arange(0, self.num_total_dof, 2)
        odd_indices = np.arange(1, self.num_total_dof, 2)

        self.total_mass = self.M[np.ix_(odd_indices, odd_indices)].sum()
        self.moment_of_inertia_zz = 0.012495943554666666   # TODO update this.

        self.rigid_body_translation_map[0, np.ix_(even_indices)] = 1/(self.total_mass)
        self.rigid_body_translation_map[1, np.ix_(odd_indices)] = 1/(self.total_mass)


    def evaluate_topology_dynamic(self, x, simp_penalization_factor=3, ramp_penalization_factor=None, filter_radius=None):
        self.evaluate_topology(x, simp_penalization_factor=simp_penalization_factor, ramp_penalization_factor=ramp_penalization_factor, filter_radius=filter_radius)
        self.M_ff = self.M[np.ix_(self.free_dof, self.free_dof)]

        size_K_ff = self.K_ff.shape[0]
        self.num_dislpacement_dofs = size_K_ff
        self.num_dislpacement_states = 2*self.num_dislpacement_dofs
        self.num_rigid_body_dofs = 3     # 2D
        num_rigid_body_states = 2*self.num_rigid_body_dofs

        # TODO Move this block out of here. Make it a problem input.
        M_ff_inv = sps.linalg.inv(self.M_ff)
        self.dampening_per_node = 0.03      # TODO implement Proportional damping
        displacement_dampening = sps.eye(self.num_dislpacement_dofs, format='csc')*self.dampening_per_node
        

        self.num_states = self.num_dislpacement_states
        num_inputs = self.num_dislpacement_dofs     # each node is a location for a potential input

        # Constructing displacement portion of A matrix
        A_10 = -M_ff_inv.dot(self.K_ff)
        A_11 = -M_ff_inv.dot(displacement_dampening)
        A = sps.lil_matrix((self.num_states, self.num_states))
        A[:self.num_dislpacement_dofs, self.num_dislpacement_dofs:] = sps.eye(self.num_dislpacement_dofs, format='csc')
        A[self.num_dislpacement_dofs:, :self.num_dislpacement_dofs] = A_10
        A[self.num_dislpacement_dofs:, self.num_dislpacement_dofs:] = A_11
        A = A.tocsc()

        B = sps.lil_matrix((self.num_states, num_inputs))
        B[self.num_dislpacement_dofs:,:] = M_ff_inv
        B = B.tocsc()

        C_dislacements = sps.lil_matrix((self.num_dislpacement_dofs, self.num_states))
        C_dislacements[:, :self.num_dislpacement_dofs] = sps.eye(self.num_dislpacement_dofs)
        C_dislacements = C_dislacements.tocsc()

        self.A = A
        self.B = B
        self.C_dislacements = C_dislacements

        # Maps from self.F*delta_t**2 to rigid body displacement. 
        self.rigid_body_translation_map = np.zeros((2, self.num_total_dof)) # 2 is the number of rigid body translation
        even_indices = np.arange(0, self.num_total_dof, 2)
        odd_indices = np.arange(1, self.num_total_dof, 2)

        self.total_mass = self.M[np.ix_(odd_indices, odd_indices)].sum()
        self.moment_of_inertia_zz = 0.012495943554666666   # TODO update this.

        self.rigid_body_translation_map[0, np.ix_(even_indices)] = 1/(self.total_mass)
        self.rigid_body_translation_map[1, np.ix_(odd_indices)] = 1/(self.total_mass)


    '''
    Evaluates dynamimcs for a fixed structural frame.

    @param loads [[t_1, [[ind1, F1], [ind2, F2], ...]_1], [t_2, [[]]_2], ...] t is the vector of times, ind is the node index where the load is applied F is the of load vector (3,)
    '''
    def evaluate_dynamics(self, loads, t_eval, t0=0, x0=None):
        if x0 is None:
            x0 = np.zeros((2*self.K_ff.shape[0],1))
            # x0 = np.zeros((1,2*self.K_ff.shape[0]))
        x = x0

        self.t_eval = t_eval
        if t0 in t_eval:
            t_eval = np.delete(t_eval, 0)
        self.nt = len(t_eval)
        
        self.x = np.zeros((self.nt+1, self.num_dislpacement_states))
        self.x[0,:] = x.reshape((-1,))
        self.rigid_body_displacement = np.zeros((self.nt+1, self.num_rigid_body_dofs))
        self.rigid_body_velocity = np.zeros((self.nt+1, self.num_rigid_body_dofs))
        self.rigid_body_acceleration = np.zeros((self.nt+1, self.num_rigid_body_dofs))
        current_rigid_body_displacement = np.zeros((3,))
        current_rigid_body_velocity = np.zeros((3,))
        self.rigid_body_origin = np.array([0.25, 0.05, 0.])

        u = np.zeros_like(self.F_f.todense())
        u = sps.csc_matrix(u)

        identity_mat = sps.eye(self.A.shape[0])

        t_eval = np.sort(t_eval)
        t_step_inputs = np.array([i[0] for i in loads])
        input_counter = 0

        if len(t_step_inputs) == 1:
            t_next_interval = np.Inf
        else:
            t_next_interval = t_step_inputs[input_counter+1]
        self.reset_loads()
        self.apply_loads(loads[input_counter][1])
        if self.use_self_weight:
            self.apply_self_weight(g=self.g)
        self.F = self.F.tocsc()
        self.F_f = self.F[self.free_dof]
        self.F_p = self.F[self.prescribed_dof]    # don't know yet
        # self.reshaped_Ff = self.F_f.reshape((-1, self.num_dimensions))
        self.reshaped_F = self.F.reshape((self.num_nodes, self.num_dimensions))
        evaluated_dynamics = {t0: x0} # {t: x}
        evaluated_exponentials = {} # {delta_t, e^(A*(delta_t))}
        for i, t in enumerate(t_eval):
            print(t)
            while t >= t_next_interval:
                
                last_eval_t = list(evaluated_dynamics.keys())[-1]
                last_eval_x = list(evaluated_dynamics.values())[-1]

                # evaulate to end of step input (from last evaluated dynamic) and add to evaluated_dynamics list
                delta_t = t_next_interval-last_eval_t
                evaluated_delta_t_list = np.array(list(evaluated_exponentials.keys()))
                tol = 1e-6
                if evaluated_delta_t_list.size != 0 and np.min(abs(evaluated_delta_t_list - delta_t)) <= tol:
                    index = np.argmin(abs(evaluated_delta_t_list - delta_t))
                    delta_t = evaluated_delta_t_list[index]
                    e_At = evaluated_exponentials[delta_t]
                else:
                    e_At = sps.linalg.expm(self.A*(delta_t))
                    evaluated_exponentials[delta_t] = e_At
                x_next_interval = e_At.dot(last_eval_x) + self.A_inv.dot((e_At - identity_mat).dot(self.B.dot(self.F_f)))
                evaluated_dynamics[t_next_interval] = x_next_interval

                
                average_displcements = self.C_dislacements.dot((last_eval_x + x_next_interval)/2)
                delta_rigid_body_dynamics = self.integrate_rigid_body_dynamics(current_rigid_body_displacement, current_rigid_body_velocity, delta_t, average_displcements)
    
                delta_rigid_body_translation = delta_rigid_body_dynamics[0]
                delta_rigid_body_translational_velocity = delta_rigid_body_dynamics[1]
                rigid_body_translational_acceleration = delta_rigid_body_dynamics[2]
                delta_rigid_body_rotation = delta_rigid_body_dynamics[3]
                delta_rigid_body_rotational_velocity = delta_rigid_body_dynamics[4]
                rigid_body_rotational_acceleration = delta_rigid_body_dynamics[5]
                
                current_rigid_body_displacement[:2] = current_rigid_body_displacement[:2] + delta_rigid_body_translation.reshape((-1,))
                current_rigid_body_displacement[2] = current_rigid_body_displacement[2] + delta_rigid_body_rotation
                current_rigid_body_velocity[:2] = current_rigid_body_velocity[:2] + delta_rigid_body_translation.reshape((-1,))
                current_rigid_body_velocity[2] = current_rigid_body_velocity[2] + delta_rigid_body_rotational_velocity


                # update self.F and self.F_f
                self.reset_loads()
                self.apply_loads(loads[input_counter][1])
                if self.use_self_weight:
                    self.apply_self_weight(g=self.g, rotation=current_rigid_body_displacement[2])
                self.F = self.F.tocsc()
                self.F_f = self.F[self.free_dof]
                self.F_p = self.F[self.prescribed_dof]    # don't know yet
                # self.reshaped_Ff = self.F_f.reshape((-1, self.num_dimensions))
                self.reshaped_F = self.F.reshape((self.num_nodes, self.num_dimensions))

                input_counter += 1
                if input_counter == len(t_step_inputs):
                    t_next_interval = np.Inf
                else:
                    t_next_interval = t_step_inputs[input_counter]

            # evaluate from last last evaluated dynamic and add to evaluated_dynamics list.
            last_eval_t = list(evaluated_dynamics.keys())[-1]
            last_eval_x = list(evaluated_dynamics.values())[-1]

            # evaulate to end of step input (from last evaluated dynamic) and add to evaluated_dynamics list
            delta_t = t-last_eval_t
            evaluated_delta_t_list = np.array(list(evaluated_exponentials.keys()))
            tol = 1e-6
            if evaluated_delta_t_list.size != 0 and np.min(abs(evaluated_delta_t_list - delta_t)) <= tol:
                index = np.argmin(abs(evaluated_delta_t_list - delta_t))
                delta_t = evaluated_delta_t_list[index]
                e_At = evaluated_exponentials[delta_t]
            else:
                e_At = sps.linalg.expm(self.A*(delta_t))
                evaluated_exponentials[delta_t] = e_At

            x_t = e_At.dot(last_eval_x) + (self.A_inv.dot((e_At - identity_mat).dot(self.B.dot(self.F_f)))).todense()
            # x_t = e_At.dot(last_eval_x) + self.A_inv.dot((e_At - identity_mat).dot((self.B.dot(self.F_f)).todense()))
            # x_t = x_t.reshape((-1,))
            evaluated_dynamics[t] = x_t
            self.x[i+1,:] = x_t.reshape((-1,))
            # self.x = np.hstack((self.x, x_t))

            # evaluate rigid body dynamics
            # taking the average displacement to update the nodes (not analytic!!)
            average_displcements = self.C_dislacements.dot((last_eval_x + x_t)/2)
            delta_rigid_body_dynamics = self.integrate_rigid_body_dynamics(current_rigid_body_displacement, current_rigid_body_velocity, delta_t, average_displcements)
    
            delta_rigid_body_translation = delta_rigid_body_dynamics[0]
            delta_rigid_body_translational_velocity = delta_rigid_body_dynamics[1]
            rigid_body_translational_acceleration = delta_rigid_body_dynamics[2]
            delta_rigid_body_rotation = delta_rigid_body_dynamics[3]
            delta_rigid_body_rotational_velocity = delta_rigid_body_dynamics[4]
            rigid_body_rotational_acceleration = delta_rigid_body_dynamics[5]

            self.rigid_body_displacement[i+1,:2] = current_rigid_body_displacement[:2] + delta_rigid_body_translation.reshape((-1,))
            self.rigid_body_displacement[i+1,2]  = current_rigid_body_displacement[2] +  delta_rigid_body_rotation
            current_rigid_body_displacement = self.rigid_body_displacement[i+1,:]
            self.rigid_body_velocity[i+1,:2] = current_rigid_body_velocity[:2] + delta_rigid_body_translational_velocity.reshape((-1,))
            self.rigid_body_velocity[i+1,2]  = current_rigid_body_velocity[2] +  delta_rigid_body_rotational_velocity
            current_rigid_body_velocity = self.rigid_body_velocity[i+1,:]
            self.rigid_body_acceleration[i+1,:2] = rigid_body_translational_acceleration
            self.rigid_body_acceleration[i+1,2]  = rigid_body_rotational_acceleration

        self.U = np.zeros((self.num_total_dof, self.nt+1))
        self.U[self.free_dof,:] = self.C_dislacements.dot(self.x.T)
        self.U_per_time_step = self.U.reshape((-1, self.nt+1))
        self.U_per_time_step = np.moveaxis(self.U_per_time_step, -1, 0)
        self.U_per_dim_per_time = self.U.reshape((self.num_nodes, self.num_dimensions, self.nt+1))
        self.U_per_dim_per_time = np.moveaxis(self.U_per_dim_per_time, -1, 0)

    
    '''
    Explicitely integrates the rigid body dynamics over a time interval.
    '''
    def integrate_rigid_body_dynamics(self, current_rigid_body_displacement, current_rigid_body_velocity, delta_t, average_displcements):
        current_rotation = current_rigid_body_displacement[2]
        # taking the average displacement to update the nodes (not analytic!!)
        # calculate rotation
        total_displacement_vec = np.zeros((self.num_nodes*self.num_dimensions,))
        total_displacement_vec[self.free_dof] = average_displcements.reshape((-1,))
        undeformed_nodes = self.mesh.nodes.copy()
        deformed_nodes = np.zeros((self.num_nodes, 3))  # need 3D to get z direction moment
        deformed_nodes[:,:2] = undeformed_nodes + total_displacement_vec.reshape((self.num_nodes, self.num_dimensions))
        deformed_nodes -= self.rigid_body_origin
        zz_basis_vector = np.array([0., 0., 1.])
        force_3d = np.zeros((self.num_nodes, 3))  # Need 3D force to get z direction moment
        force_3d[:,:2] = self.reshaped_F.todense()
        net_moment_zz = np.sum(np.cross(deformed_nodes, force_3d).dot(zz_basis_vector))
        delta_rigid_body_rotational_velocity = 1/(self.moment_of_inertia_zz) * net_moment_zz * delta_t
        delta_rigid_body_rotation = 1/(2*self.moment_of_inertia_zz)*net_moment_zz*(delta_t**2) + (current_rigid_body_velocity[2] + delta_rigid_body_rotational_velocity/2)*delta_t

        # calculate translation
        averate_rotation = current_rotation + delta_rigid_body_rotation/2
        c = np.cos(averate_rotation)
        s = np.sin(averate_rotation)
        rotation_matrix = np.array([[c, -s], [s, c]])

        net_translation_acceleration = rotation_matrix.dot(self.rigid_body_translation_map.dot(self.F_body_fixed.todense())) + self.rigid_body_translation_map.dot(self.F_earth_fixed.todense())
        rigid_body_translational_acceleration = np.zeros((self.num_dimensions,))    # made because of scipy sparse reshape bug
        rigid_body_translational_acceleration[0] = net_translation_acceleration[0]
        rigid_body_translational_acceleration[1] = net_translation_acceleration[1]
        delta_rigid_body_translational_velocity = rigid_body_translational_acceleration * delta_t
        delta_rigid_body_translation = rigid_body_translational_acceleration/2*delta_t**2 + (current_rigid_body_velocity[:2] + delta_rigid_body_translational_velocity/2)*delta_t

        delta_rigid_body_dynamics = [delta_rigid_body_translation, delta_rigid_body_translational_velocity, rigid_body_translational_acceleration, 
                                    delta_rigid_body_rotation, delta_rigid_body_rotational_velocity, net_moment_zz]

        return delta_rigid_body_dynamics


    '''
    Calculates the stresses at the integration points.
    '''
    def evaluate_stresses(self):
        self.stresses = (self.stress_map.dot(self.U)).T
        self.stresses_dict = {}
        self.stresses_dict['xx'] = self.stresses[:, np.ix_(self.stress_index_map['xx'])].reshape((self.nt+1, -1))   # reshape for weird indexing thing (to get rid of extra axis)
        self.stresses_dict['yy'] = self.stresses[:, np.ix_(self.stress_index_map['yy'])].reshape((self.nt+1, -1))
        self.stresses_dict['xy'] = self.stresses[:, np.ix_(self.stress_index_map['xy'])].reshape((self.nt+1, -1))
        self.stresses_dict['axial'] = self.stresses[:, np.ix_(self.stress_index_map['axial'])].reshape((self.nt+1, -1))

        stresses_vec = np.zeros((self.nt+1, 3, self.stresses_dict['xx'].shape[1]))
        stresses_vec[:,0,:] = self.stresses_dict['xx']
        stresses_vec[:,1,:] = self.stresses_dict['yy']
        stresses_vec[:,2,:] = self.stresses_dict['xy']

        # test = np.zeros((self.nt+1, self.stresses_dict['xx'].shape[1], 3))
        # test[:,:,0] = self.stresses_dict['xx']
        # test[:,:,1] = self.stresses_dict['yy']
        # test[:,:,2] = self.stresses_dict['xy']

        von_mises_map = np.array([
            [1., -1/2, 0],
            [-1/2, 1., 0],
            [0., 0., 3.]
        ])
        m_dot_stresses = von_mises_map.dot(stresses_vec)
        
        von_mises_stresses = np.zeros((self.nt+1, self.stresses_dict['xx'].shape[1]))
        averaged_von_mises_stresses = np.zeros((self.nt+1, int(self.stresses_dict['xx'].shape[1]/4)))
        counter = 0
        averaged_stress_index = 0
        for i in range(self.stresses_dict['xx'].shape[1]):
            for j in range(self.nt+1):
                von_mises_stresses[j,i] = np.sqrt(stresses_vec[j,:,i].dot(m_dot_stresses[:,j,i]))
                averaged_von_mises_stresses[j,averaged_stress_index] += von_mises_stresses[j,i]/4

            counter += 1
            if counter == 4:
                averaged_stress_index += 1
                counter = 0

        self.von_mises_stresses = von_mises_stresses
        self.averaged_von_mises_stresses = averaged_von_mises_stresses

        return self.stresses


    def evaluate_stress_points(self, visualization_scaling_factor=1):
        nodes = self.mesh.nodes.copy()
        nodes = np.broadcast_to(nodes, (self.nt+1, *nodes.shape))
        scaled_displacements = self.U_per_dim_per_time*visualization_scaling_factor
        if len(scaled_displacements.shape) == 2:    # unfortunately added because static won't reshape to 3D for some reason.
            nodes_static = nodes.reshape((self.num_nodes, self.num_dimensions))
            deformed_nodes = nodes_static + scaled_displacements.reshape(nodes.shape)
            # deformed_nodes = deformed_nodes.reshape((self.nt+1, self.num_nodes, self.num_dimensions))   ## caaaaaan't do this!!!!!
            deformed_nodes_new = np.zeros((self.nt+1, self.num_nodes, self.num_dimensions))
            deformed_nodes_new[0,:,:] = deformed_nodes
            deformed_nodes = deformed_nodes_new
        else:
            deformed_nodes = nodes + scaled_displacements.reshape(nodes.shape)            
            

        # sps doesn't have tensordot...
        # self.stress_eval_points = self.stress_eval_points_map.dot(deformed_nodes) # so this can't be done
        self.stress_eval_points = np.zeros((self.nt+1, self.stress_eval_points_map.shape[0], self.num_dimensions))
        for i in range(self.num_dimensions):    # loop over dimensions instead of time bcecause fewer iterations
            deformed_nodes_slice = deformed_nodes[:,:,i]
            self.stress_eval_points[:,:,i] = self.stress_eval_points_map.dot(deformed_nodes_slice.T).T


        self.stress_eval_points_dict = {}
        self.stress_eval_points_dict['xx'] = self.stress_eval_points[:, np.ix_(self.stress_eval_points_indices['xx']), :].reshape((self.nt+1, -1, self.num_dimensions))     #reshape because of weird axis was added.
        self.stress_eval_points_dict['yy'] = self.stress_eval_points[:, np.ix_(self.stress_eval_points_indices['yy']), :].reshape((self.nt+1, -1, self.num_dimensions))
        self.stress_eval_points_dict['xy'] = self.stress_eval_points[:, np.ix_(self.stress_eval_points_indices['xy']), :].reshape((self.nt+1, -1, self.num_dimensions))
        self.stress_eval_points_dict['axial'] = self.stress_eval_points[:, np.ix_(self.stress_eval_points_indices['axial']), :].reshape((self.nt+1, -1, self.num_dimensions))


    '''
    Calculates the total strain energy.
    '''
    def evaluate_strain_energy(self):
        # self.strain_energy = self.U.T.dot(self.K.dot(self.U))/2
        self.strain_energy = self.F.T.dot(self.U)/2
        return self.strain_energy

    '''
    Calculates the strain energy per element.
    '''
    def evaluate_strain_energy_per_element(self):
        self.strain_energy_per_element = np.zeros(self.num_elements)
        self.strain_energy_density_per_element = np.zeros(self.num_elements)
        self.element_midpoints = np.zeros((self.num_elements, self.num_dimensions))
        self.element_midpoints_plot = np.zeros((self.num_elements, self.num_dimensions))

        nodes = self.mesh.nodes
        U_reshaped = self.U.reshape((self.num_nodes, -1))
        max_x_dist = np.linalg.norm(max(nodes[:,0]) - min(nodes[:,0]))
        max_y_dist = np.linalg.norm(max(nodes[:,1]) - min(nodes[:,1]))
        scale_dist = np.linalg.norm(np.array([max_x_dist, max_y_dist]))
        if np.linalg.norm(self.U) != 0:
            self.visualization_scaling_factor = scale_dist*0.1/max(np.linalg.norm(U_reshaped, axis=1))
        else:
            self.visualization_scaling_factor = 0

        for i, element in enumerate(self.mesh.elements):
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            element_U = self.U[np.ix_(element_dofs)]
            strain_energy, strain_energy_density = element.calc_strain_energy(element_U)
            self.strain_energy_per_element[i] = strain_energy
            self.strain_energy_density_per_element[i] = strain_energy_density

            self.element_midpoints[i,:] = element.calc_midpoint(element_U)
            self.element_midpoints_plot[i,:] = element.calc_midpoint(element_U*self.visualization_scaling_factor)

        return self.strain_energy_per_element, self.strain_energy_density_per_element


    # '''
    # ABANDONED: because map is insanely large, and it's just looping over dot products instead of a for loop. there are also bug(s).
    # Calculates the strain energy per element.
    # strain_energy = 1/2*U.T*K*U per element
    # '''
    # def calc_strain_energy_density_per_element(self):
    #     reshaped_U = np.zeros(self.U.shape[0])
    #     for i in range(self.U.shape[0]):
    #         reshaped_U[i] = self.U[i]

    #     print(self.strain_energy_density_per_element_map)
    #     stacked_U = sps.kron(sps.eye(self.num_elements), reshaped_U)   # would work normally if it weren't for reshae issue with scipy
    #     stiffness_force = self.strain_energy_density_per_element_map.dot(self.U)
    #     self.strain_energy_density_per_element = stacked_U.dot(stiffness_force)/2
    #     return self.strain_energy_density_per_element



    def plot(self, stress_type=None, time_step=None, dof=None, show_dislpacements=False, show_nodes=False, show_connections=False, show_undeformed=False,
                save_plots=False, video_file_name=None, video_fps=1, show=True):
        nodes = self.mesh.nodes
        U_reshaped = self.U.reshape((-1, self.num_dimensions))
        max_x_dist = np.linalg.norm(max(nodes[:,0]) - min(nodes[:,0]))
        max_y_dist = np.linalg.norm(max(nodes[:,1]) - min(nodes[:,1]))
        scale_dist = np.linalg.norm(np.array([max_x_dist, max_y_dist]))
        if np.linalg.norm(self.U) != 0:
            visualization_scaling_factor = scale_dist*0.1/max(np.linalg.norm(U_reshaped, axis=1))
        else:
            visualization_scaling_factor = 0
        self.visualization_scaling_factor = visualization_scaling_factor

        self.element_midpoints = np.zeros((self.nt+1, self.num_elements, self.num_dimensions))
        self.element_midpoints_plot = np.zeros((self.nt+1, self.num_elements, self.num_dimensions))
        self.element_midpoints_undeformed = np.zeros((self.nt+1, self.num_elements, self.num_dimensions))
        for i, element in enumerate(self.mesh.elements):
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            element_U = self.U[np.ix_(element_dofs)]

            self.element_midpoints[:,i,:] = element.calc_midpoint(element_U)
            self.element_midpoints_plot[:,i,:] = element.calc_midpoint(element_U*self.visualization_scaling_factor)
            self.element_midpoints_undeformed[:,i,:] = element.calc_midpoint()

        if time_step is None:
            time_step = range(len(self.t_eval))
        elif type(time_step) == int:
            time_step = [time_step]
        if dof is not None and type(dof) == int:
            dof = [dof]

        if stress_type is not None:
            self.evaluate_stress_points(visualization_scaling_factor)

            if stress_type == 'x' or stress_type == 'xx':
                stresses = self.stresses_dict['xx']
                stress_eval_points = self.stress_eval_points_dict['xx']
            elif stress_type == 'y' or stress_type == 'yy':
                stresses = self.stresses_dict['yy']
                stress_eval_points = self.stress_eval_points_dict['yy']
            elif stress_type == 'tao' or stress_type == 'xy':
                stresses = self.stresses_dict['xy']
                stress_eval_points = self.stress_eval_points_dict['xy']
            elif stress_type == 'von_mises' or stress_type == 'vm':
                stresses = self.von_mises_stresses
                stress_eval_points = self.stress_eval_points_dict['xx']
            elif stress_type == 'averaged_von_mises' or stress_type == 'avm':
                stresses = self.averaged_von_mises_stresses
                stress_eval_points = self.element_midpoints_plot
            elif stress_type == 'axial' or stress_type == 'tension' or stress_type == 'compression' or stress_type == '11':
                stresses = self.stresses_dict['axial']
                stress_eval_points = self.stress_eval_points_dict['axial']

        if dof is None:
            print('Plotting...')
            for t_step in time_step:
                t = self.t_eval[t_step]
                plt.figure()
                if show_dislpacements:
                    self.plot_displacements(show_nodes=show_nodes, show_connections=show_connections, show_undeformed=show_undeformed,
                                             time_step=t_step, visualization_scaling_factor=visualization_scaling_factor, show=False)
                if stress_type is not None:
                    self.plot_stresses(stresses=stresses, stress_eval_points=stress_eval_points, time_step=t_step, show=False)

                if stress_type is None:
                    # plt.title(f'Structure at t ={t: 9.5f}')
                    plt.title(f'Structure at t ={t: 1.2e}')
                else:
                    plt.title(f'Stress (sigma_{stress_type}) Colorplot of Structure at t ={t:1.2e}')
                plt.xlabel(f'x (m*{visualization_scaling_factor:3.0e})')
                plt.ylabel(f'y (m*{visualization_scaling_factor:3.0e})')
                plt.gca().set_aspect('equal')
                if save_plots or video_file_name is not None:
                    plt.savefig(f'plots/video_plot_at_t_{t:9.9f}.png', bbox_inches='tight')
                if show:
                    plt.show()
                plt.close()

            if video_file_name is not None:
                self.generate_video(video_file_name=video_file_name, video_fps=video_fps)

        elif dof is not None:
            if stress_type is not None:
                visualization_scaling_factor = max(np.linalg.norm(stresses, axis=1))*0.01/max(np.linalg.norm(U_reshaped, axis=1))
            else:
                visualization_scaling_factor = 1

            plt.figure()
            for index in dof:
                if show_dislpacements:
                    plt.plot(self.t_eval, self.U[index, :]*visualization_scaling_factor*50, '-', label=f'Displacement of node {index}')
                if stress_type is not None:
                    plot_stresses = stresses[:, index]   # (time_step, dof)
                    plt.plot(self.t_eval, plot_stresses, '-o', label=f'Stress of node {index}')
            
            if show_dislpacements and stress_type is not None:
                plt.title(f'Stress (sigma_{stress_type}) and Scaled Displacement vs. Time')
                plt.ylabel(f'Stress (Pa) and Displacement (m /{visualization_scaling_factor:3.0e})')
            elif show_dislpacements:
                plt.title(f'Y-Displacement of Node(s)')
                plt.ylabel('Displacement (m)')
            elif stress_type is not None:
                plt.title(f'Stress (sigma_{stress_type}) vs. Time')
                plt.ylabel('Stress (Pa)')

            plt.xlabel('Time (s)')
            plt.legend()

            if show:
                plt.show()


    '''
    Plots the solution.
    '''
    def plot_displacements(self, show_nodes=True, show_connections=True, show_undeformed=True, time_step=0, visualization_scaling_factor=None, show=True):
        nodes = self.mesh.nodes.copy()
        nodes = np.broadcast_to(nodes, (self.nt+1, *nodes.shape))
        if visualization_scaling_factor is None:
            visualization_scaling_factor = self.visualization_scaling_factor
        
        scaled_displacements = self.U_per_dim_per_time*visualization_scaling_factor
        if len(scaled_displacements.shape) == 2:    # unfortunately added because static won't reshape to 3D for some reason.
            nodes_static = nodes.reshape((self.num_nodes, self.num_dimensions))
            deformed_nodes = nodes_static + scaled_displacements.reshape(nodes.shape)
            # deformed_nodes = deformed_nodes.reshape((self.nt+1, self.num_nodes, self.num_dimensions))   ## caaaaaan't do this!!!!!
            deformed_nodes_new = np.zeros((self.nt+1, self.num_nodes, self.num_dimensions))
            deformed_nodes_new[0,:,:] = deformed_nodes
            deformed_nodes = deformed_nodes_new
        else:
            deformed_nodes = nodes + scaled_displacements.reshape(nodes.shape)
            

        
        if nodes.shape[-1] == 2:
            if show_nodes:
                if show_undeformed:
                    plt.plot(nodes[time_step,:,0], nodes[time_step, :,1], 'ko')
                else:
                    plt.plot(deformed_nodes[time_step,:,0], deformed_nodes[time_step,:,1], 'r*')
            if show_connections:
                for start_node_index in self.node_connectivity.keys():
                    for end_node_index in self.node_connectivity[start_node_index]:
                        if end_node_index <= start_node_index:
                            continue
                        
                        if show_undeformed:
                            start_node = nodes[time_step,start_node_index]
                            end_node = nodes[time_step,end_node_index]
                            plt.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], '--k')
                        else:
                            start_deformed_node_x = deformed_nodes[time_step,start_node_index,0]
                            start_deformed_node_y = deformed_nodes[time_step,start_node_index,1]
                            end_deformed_node_x = deformed_nodes[time_step,end_node_index,0]
                            end_deformed_node_y = deformed_nodes[time_step,end_node_index,1]
                            plt.plot([start_deformed_node_x, end_deformed_node_x], [start_deformed_node_y, end_deformed_node_y], '--r')


            plt.title(f'Scaled Discplacements (x{visualization_scaling_factor}) with {self.num_elements} Elements')
            plt.xlabel('x1 (m)')
            plt.ylabel('x2 (m)')
            # plt.legend(('Initial Structure', 'Deformed Structure'))
            if show:
                plt.show()
        else:
            print("WARNING: Plotting not set up for anything other than 2d right now.")



    def plot_stresses(self, stresses, stress_eval_points, time_step=-1, show=False):
        # Added unnecessary section because can't get 1D vector from sps output.
        stresses_plot = np.zeros(stresses.shape[1])
        stress_eval_points_plot = np.zeros(stress_eval_points.shape[1:])
        for i in range(len(stresses_plot)):
            stresses_plot[i] = stresses[time_step, i]
            stress_eval_points_plot[i,:] = stress_eval_points[time_step, i,:]
        # end unnecessary section

        plt.scatter(stress_eval_points_plot[:,0], stress_eval_points_plot[:,1], c=stresses_plot, cmap='bwr')

        plt.title(f'Stress (Pa) Color Plot with {self.num_elements} Elements')
        plt.xlabel('x1 (m)')
        plt.ylabel('x2 (m)')
        plt.colorbar(orientation='horizontal')
        if show:
            plt.show()


    def plot_rigid_body_displacement(self, x_axis='x', y_axis='y', show=True):
        t_data = self.t_eval
        x_data = self.rigid_body_displacement[:,0]
        y_data = self.rigid_body_displacement[:,1]
        rot_z_data = self.rigid_body_displacement[:,2]
        plot_points = np.zeros((self.nt+1, 2))  # 2D plot

        if x_axis == 't':
            x_coords = t_data
        elif x_axis == 'x':
            x_coords = x_data
        elif x_axis == 'y':
            x_coords = y_data
        elif x_axis == 'rot_z':
            x_coords = rot_z_data

        if y_axis == 't':
            y_coords = t_data
        elif y_axis == 'x':
            y_coords = x_data
        elif y_axis == 'y':
            y_coords = y_data
        elif y_axis == 'rot_z':
            y_coords = rot_z_data

        # plt.plot(plot_points[:,0], plot_points[:,1], '-bo')
        plt.plot(x_coords, y_coords, '-bo')
        plt.title(f'Rigid Body Dynamics: {y_axis} vs. {x_axis}')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        if show:
            plt.show()


    '''
    Generates a video.
    '''
    def generate_video(self, video_file_name, video_fps):
        print('Creating Video...')
        image_folder = 'plots'
        images = [f'video_plot_at_t_{t:9.9f}.png' for t in self.t_eval]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_file_name, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (width,height))

        for image in images:
            image_frame = cv2.imread(os.path.join(image_folder, image))
            frame_resized = cv2.resize(image_frame, (width, height)) 
            video.write(frame_resized)

        cv2.destroyAllWindows()
        video.release()


    def plot_strain_energy(self, normalized_by_volume=False, show_dislpacements=False, show_nodes=False, show_connections=False, show_undeformed=False):
        plt.figure()
        if show_dislpacements:
            self.plot_displacements(show_nodes=show_nodes, show_connections=show_connections, show_undeformed=show_undeformed, show=False)
        if normalized_by_volume:
            plt.scatter(self.element_midpoints_plot[:,0], self.element_midpoints_plot[:,1], c=self.strain_energy_density_per_element, cmap='Oranges')
            plt.title(f'Strain Energy Density (N/m^2) Color Plot with {self.num_elements} Elements')
        else:    
            plt.scatter(self.element_midpoints_plot[:,0], self.element_midpoints_plot[:,1], c=self.strain_energy_per_element, cmap='Oranges')
            plt.title(f'Strain Energy (N*m) Color Plot with {self.num_elements} Elements')

        plt.xlabel('x1 (m)')
        plt.ylabel('x2 (m)')
        plt.colorbar()
        plt.show()


    '''
    Plot the topology of the structure given the densities.
    '''
    def plot_topology(self, densities):
        self.plot(show=False)
        self.plot_displacements(show_nodes=False, show_connections=True, show_undeformed=True, show=False)
        densities[densities > 1] = 1.
        densities[densities < 1.e-3] = 1.e-3
        plt.scatter(self.element_midpoints_undeformed[0,:,0], self.element_midpoints_undeformed[0,:,1], c=densities, cmap='gray_r')
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.show()