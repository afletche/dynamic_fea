'''
Author: Andrew Fletcher
Date: 11/24

This class is for performing 2D structural FEA problems.
'''


from functools import total_ordering
from pprint import pprint
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

        self.apply_loads(loads)
        self.apply_boundary_conditions(boundary_conditions)

        self.visualization_scaling_factor = 1.



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
        if len(loads) != 2:
            Exception("Please input a 'load' of the form [nodes, forces]")
        
        for load in loads:
            node = load[0]
            force = load[1]
            for i in range(len(force)):
                if force[i] == 0:
                    continue
                load_component_dof = node*self.dimensions_spanned+i
                if self.F[load_component_dof] != 0:
                    self.F[load_component_dof] = self.F[load_component_dof].data + force[i]
                else:
                    self.F[load_component_dof] = force[i]


    '''
    Applies self-weight loads onto the structure.

    Inputs:
    - g : float : gravitational constant (9.81 for Earth)
    '''
    def apply_self_weight(self, g):
        # For each element, add its weight
        for i in range(self.num_elements):
            element = self.mesh.elements[i]
            element.calc_l()
            local_F_g = element.calc_self_weight(g)
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            self.F[np.ix_(element_dofs)] += local_F_g


    def reset_loads(self):
        self.F = sps.lil_matrix((self.num_total_dof, 1))


    '''
    Applies the boundary conditions to the FEA problem.
    
    Inputs:
    - boundary_conditions : List : [(node1, axis1, displacement1), (node2, ...), ...]
    - nodes : np.ndarray (num_bc,): the nodes with the applied BCs
    - axes  : np.ndarray (num_bc,): the corresponding axis that is prescribed at each node
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
        # self.K = np.zeros((self.num_total_dof, self.num_total_dof))
        # self.F = np.zeros((self.num_total_dof, 1))
        # self.U = np.zeros((self.num_total_dof, 1))

        self.K = sps.lil_matrix((self.num_total_dof, self.num_total_dof))
        self.F = sps.lil_matrix((self.num_total_dof, 1))
        self.U = sps.lil_matrix((self.num_total_dof, 1))

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
        self.assemble_connectivity()
        self.assemble_stress_map()
        # self.assemble_strain_energy_per_element_map()
        self.K = self.K.tocsc()
        self.F = self.F.tocsc()
        self.stress_map = self.stress_map.tocsc()      

        self.free_dof = np.setdiff1d(self.total_dof, self.prescribed_dof)   # all dof that are not prescribed are free
        
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

        self.U[self.free_dof] = self.U_f
        # self.U[self.prescribed_dof] = self.U_p    # known already
        self.U = self.U.todense()

        # self.F[self.free_dof] = self.F_f          # known already
        self.F[self.prescribed_dof] = self.F_p

        self.U_per_dim_per_time = self.U.reshape((self.nt+1, self.num_nodes, self.num_dimensions))

    
    def setup_dynamics(self):
        self.setup()
        self.size_K_ff = self.K_ff.shape[0]
        num_dislpacement_dofs = self.size_K_ff
        num_dislpacement_states = 2*num_dislpacement_dofs
        num_rigid_body_dofs = 3     # 2D
        num_rigid_body_states = 2*num_rigid_body_dofs

        # TODO Move this block out of here. Make it a problem input.
        mass_per_node = 1
        M = sps.eye(self.size_K_ff, format='csc')*mass_per_node
        M_inv = sps.linalg.inv(M)
        self.dampening_per_node = 0.03      # TODO implement Proportional damping
        displacement_dampening = sps.eye(num_dislpacement_dofs, format='csc')*self.dampening_per_node
        total_mass = M.sum()
        moment_of_inertia_zz = total_mass   # TODO update this.

        num_states = num_dislpacement_states + num_rigid_body_states
        num_inputs = self.size_K_ff

        # Constructing displacement portion of A matrix
        A_10 = -M_inv.dot(self.K_ff)
        A_11 = -M_inv.dot(displacement_dampening)
        A = sps.lil_matrix((num_states, num_states))
        A[:num_dislpacement_dofs, num_dislpacement_dofs:num_dislpacement_states] = sps.eye(self.size_K_ff, format='csc')
        A[num_dislpacement_dofs:num_dislpacement_states, :num_dislpacement_dofs] = A_10
        A[num_dislpacement_dofs:num_dislpacement_states, num_dislpacement_dofs:num_dislpacement_states] = A_11

        test = A.todense()
        print(test)
        print(test[-8:,-8:])

        # Constructing rigid body portion of A matrix. No RBD stiffness or damping for now. (drag could contribute damping)
        # X direction state space
        A_10 = -M_inv.dot(self.K_ff)
        A_11 = -M_inv.dot(displacement_dampening)
        A[num_dislpacement_states:num_dislpacement_states+1, num_dislpacement_states+1:num_dislpacement_states+2] = 1    # Identity of size one.
        # A[num_dislpacement_dofs:num_dislpacement_states, :num_dislpacement_dofs] = A_10
        # A[num_dislpacement_dofs:num_dislpacement_states, num_dislpacement_dofs:num_dislpacement_states] = A_11

        test = A.todense()
        print(test)
        print(test[-8:,-8:])


        # Y direction state space
        A_10 = -M_inv.dot(self.K_ff)
        A_11 = -M_inv.dot(displacement_dampening)
        A[num_dislpacement_states+2:num_dislpacement_states+3, num_dislpacement_states+3:num_dislpacement_states+4] = 1    # Identity of size one.
        # A[num_dislpacement_dofs:num_dislpacement_states, :num_dislpacement_dofs] = A_10
        # A[num_dislpacement_dofs:num_dislpacement_states, num_dislpacement_dofs:num_dislpacement_states] = A_11

        test = A.todense()
        print(test)
        print(test[-8:,-8:])


        # Rotational (around Z) state space
        A_10 = -M_inv.dot(self.K_ff)
        A_11 = -M_inv.dot(displacement_dampening)
        A[num_dislpacement_states+4:num_dislpacement_states+5, num_dislpacement_states+5:num_dislpacement_states+6] = 1    # Identity of size one.
        # A[num_dislpacement_dofs:num_dislpacement_states, :num_dislpacement_dofs] = A_10
        # A[num_dislpacement_dofs:num_dislpacement_states, num_dislpacement_dofs:num_dislpacement_states] = A_11

        test = A.todense()
        print(test)
        print(test[-8:,-8:])


        A = A.tocsc()
        test = A.todense()
        print(test)
        print(test[-4:,-4:])
        # print('Eigenvalues of A: ', sps.linalg.eigs(A))
        self.eigs = sps.linalg.eigs(A)
        self.A_inv = sps.linalg.inv(A)
        # print('inv A: ', sps.linalg.inv(A))   # made sure not singular.
        # print('det: ', np.linalg.det(A.todense()))
        A_inv = sps.linalg.inv(A)
        print('NORM: ', np.linalg.norm((A.dot(A_inv)).todense() - np.eye(A.shape[0])))

        B = sps.lil_matrix((num_states, num_inputs))
        B[num_dislpacement_dofs:num_dislpacement_states,:] = M_inv
        B = B.tocsc()

        C_dislacements = sps.lil_matrix((num_dislpacement_dofs, num_states))
        C_dislacements[:, :num_dislpacement_states] = sps.eye(num_dislpacement_dofs)
        C_dislacements = C_dislacements.tocsc()

        C_rigid_body_dofs = sps.lil_matrix((num_rigid_body_dofs, num_states))
        C_rigid_body_dofs[:, num_dislpacement_states:(num_dislpacement_states+num_rigid_body_dofs)] = sps.eye(num_rigid_body_dofs)
        C_rigid_body_dofs = C_rigid_body_dofs.tocsc()

        self.A = A
        self.B = B
        self.C_dislacements = C_dislacements
        self.C_rigid_body_dofs = C_rigid_body_dofs


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
        
        self.x = x
        self.rigid_body_disp = np.zeros((self.num_dimensions,1))
        self.rigid_body_origin = np.array([0.25, 0.05])

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
        self.F = self.F.tocsc()
        self.F_f = self.F[self.free_dof]
        self.F_p = self.F[self.prescribed_dof]    # don't know yet
        self.reshaped_Ff = self.F_f.reshape((-1, self.num_dimensions))
        evaluated_dynamics = {t0: x0} # {t: x}
        evaluated_exponentials = {} # {delta_t, e^(A*(delta_t))}
        for t in t_eval:
            print(t)
            while t >= t_next_interval:
                
                last_eval_t = list(evaluated_dynamics.keys())[-1]
                last_eval_x = list(evaluated_dynamics.values())[-1]

                # evaulate to end of step input (from last evaluated dynamic) and add to evaluated_dynamics list
                delta_t = t_next_interval-last_eval_t
                if delta_t in evaluated_exponentials.keys():
                    e_At = evaluated_exponentials[delta_t]
                else:
                    e_At = sps.linalg.expm(self.A*(delta_t))
                    evaluated_exponentials[delta_t] = e_At
                x_next_interval = e_At.dot(last_eval_x) + self.A_inv.dot((e_At - identity_mat).dot(self.B.dot(self.F_f)))
                evaluated_dynamics[t_next_interval] = x_next_interval

                rigid_body_displacement = self.reshaped_Ff.sum(0).T * (delta_t**2)/2
                self.rigid_body_disp = np.hstack((self.rigid_body_disp, rigid_body_displacement))

                # update self.F and self.F_f
                self.reset_loads()
                self.apply_loads(loads[input_counter][1])
                self.F = self.F.tocsc()
                self.F_f = self.F[self.free_dof]
                self.F_p = self.F[self.prescribed_dof]    # don't know yet
                self.reshaped_Ff = self.F_f.reshape((-1, self.num_dimensions))


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
            if delta_t in evaluated_exponentials.keys():
                e_At = evaluated_exponentials[delta_t]
            else:
                e_At = sps.linalg.expm(self.A*(delta_t))
                evaluated_exponentials[delta_t] = e_At

            x_t = e_At.dot(last_eval_x) + (self.A_inv.dot((e_At - identity_mat).dot(self.B.dot(self.F_f)))).todense()
            # x_t = e_At.dot(last_eval_x) + self.A_inv.dot((e_At - identity_mat).dot((self.B.dot(self.F_f)).todense()))
            # x_t = x_t.reshape((-1,))
            evaluated_dynamics[t] = x_t
            self.x = np.hstack((self.x, x_t))

            rigid_body_displacement = self.reshaped_Ff.sum(0).T * (delta_t**2)/2        # to account for initial velocity, add term here
            
            for i, load in enumerate(loads):
                pass

            # get displalcement vector

            # cross displacement vector with force vector
            # integrate across time
            # rigid_body_rotation = self.
            self.rigid_body_disp = np.hstack((self.rigid_body_disp, rigid_body_displacement))

            # all of the time steps can be built, and then all of the matrix exponentials can be calculated in parallel
            # These matrix exponentials can then be plugged into equations to get all states in parallel.

        self.U = np.zeros((self.num_total_dof, self.nt+1))
        self.U[self.free_dof,:] = self.C_dislacements.dot(self.x)
        self.U_per_time_step = self.U.reshape((-1, self.nt+1))
        self.U_per_time_step = np.moveaxis(self.U_per_time_step, -1, 0)
        self.U_per_dim_per_time = self.U.reshape((self.num_nodes, self.num_dimensions, self.nt+1))
        self.U_per_dim_per_time = np.moveaxis(self.U_per_dim_per_time, -1, 0)


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


    def evaluate_dynamic_stresses(self):
        self.stresses = (self.stress_map.dot(self.U)).T
        self.stresses_dict = {}
        self.stresses_dict['xx'] = self.stresses[:, np.ix_(self.stress_index_map['xx'])].reshape((self.nt+1, -1))   # reshape for weird indexing thing (to get rid of extra axis)
        self.stresses_dict['yy'] = self.stresses[:, np.ix_(self.stress_index_map['yy'])].reshape((self.nt+1, -1))
        self.stresses_dict['xy'] = self.stresses[:, np.ix_(self.stress_index_map['xy'])].reshape((self.nt+1, -1))
        self.stresses_dict['axial'] = self.stresses[:, np.ix_(self.stress_index_map['axial'])].reshape((self.nt+1, -1))
        return self.stresses


    '''
    Calculates the total strain energy.
    '''
    def calc_strain_energy(self):
        self.strain_energy = self.U.T.dot(self.K.dot(self.U))/2
        return self.strain_energy

    '''
    Calculates the strain energy per element.
    '''
    def calc_strain_energy_per_element(self):
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
        plt.colorbar()
        if show:
            plt.show()


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
                    plt.title(f'Structure at t ={t: 9.5f}')
                else:
                    plt.title(f'Stress Colorplot of Structure at t ={t:9.5f}')
                plt.xlabel(f'x (m*{visualization_scaling_factor})')
                plt.ylabel(f'y (m*{visualization_scaling_factor})')
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
                    plt.plot(self.t_eval, self.U[index, :]*visualization_scaling_factor, '-o', label=f'Displacement of node {index}')
                if stress_type is not None:
                    plot_stresses = stresses[:, index]   # (time_step, dof)
                    plt.plot(self.t_eval, plot_stresses, '-o', label=f'Stress of node {index}')
            
            if show_dislpacements and stress_type is not None:
                plt.title(f'Stress (sigma_{stress_type}) and Scaled Displacement vs. Time')
                plt.ylabel(f'Stress (Pa) and Displacement (m /{visualization_scaling_factor})')
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


if __name__ == "__main__":
    from material import IsotropicMaterial
    from component import Component
    from mesh import UnstructuredMesh
    from design_representation import FERepresentation


    nodes = np.array([
        [0., 0.],
        [0., 0.0025],
        [0., 0.005],
        [0.00333, 0.],
        [0.00333, 0.0025],
        [0.00333, 0.005],
        [0.00666, 0.],
        [0.00666, 0.0025],
        [0.01, 0.],
        [0.01, 0.005]
    ])

    element_nodes = np.array([
        [0, 1, 3, 4],
        [1, 2, 4, 5],
        [3, 4, 6, 7],
        [4, 5, 7, 9],
        [6, 7, 8, 9]
    ])

    thickness = 0.001   # m

    hw5_mat = IsotropicMaterial(name='hw5_mat', E=100e9, nu=0.3)
    
    hw5_component = Component(name='hw5_structure')
    hw5_component_properties_dict = {
        'material' : hw5_mat,
        'thickness': thickness
    }
    hw5_component.add_properties(hw5_component_properties_dict)

    hw5_mesh = UnstructuredMesh(name='hw_mesh', nodes=nodes, element_nodes=element_nodes)
    # hw5_component.add_design_representations(hw5_mesh)

    hw5_fe_representation = FERepresentation(name='hw5_fe_representation', components=[hw5_component], meshes=[hw5_mesh])

    load_nodes = np.array([2, 5, 9, 8, 9])
    load_forces = np.array([
        [0., 30./4e3],
        [0., 11.25e3],
        [0., 11.25e3],
        [0.4e3, 0.],
        [1.6e3, 0.]
        ])
    
    hw5_loads = [(2, np.array([0., 5.e3])),
        (5, np.array([0., 15.e3])),
        (9, np.array([0., 10.e3])),
        (8, np.array([0.4e3, 0.])),
        (9, np.array([1.6e3, 0.]))]

    
    hw5_boundary_conditions = [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (0, 1, 0),
        (3, 1, 0),
        (6, 1, 0),
        (8, 1, 0)
    ]

    hw5 = FEA(design_representation=hw5_fe_representation, loads=hw5_loads, boundary_conditions=hw5_boundary_conditions)
    hw5.setup()
    hw5.evaluate()
    print(hw5.U)
    hw5.plot()
    
    hw5.calc_stresses()
    print(hw5.stresses[4,:,:2])     # normal stressses of element A
