'''
Author: Andrew Fletcher
Date: 11/24

This class is for performing 2D structural FEA problems.
'''


from pprint import pprint
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time
import cv2
import os


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
        
        self.initialize_ode()

        self.apply_loads(loads)
        self.apply_boundary_conditions(boundary_conditions)



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

        self.stress_map = sps.lil_matrix((self.num_elements*4*3, self.num_total_dof))   # TODO Generalize for 3D non-quads
        self.integration_coordinates = np.zeros((self.num_elements*4, self.num_dimensions))     # TODO Generalize for 3D non-quads


    '''
    Assembles the global stiffness matrix based on the local stiffness matrices (using element objects)
    '''
    def assemble_k(self):
        # For each element, add its local stiffness contribution
        for i in range(self.num_elements):
            element = self.mesh.elements[i]
            # local_K = element.calc_k_local()
            local_K = element.K_local
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            self.K[np.ix_(element_dofs, element_dofs)] += local_K

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
        for i in range(self.num_elements):
            element = self.mesh.elements[i]
            local_stress_map = element.stress_map
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            start_index = i*element.num_integration_points*3    # 3 types of stresses for 2D TODO Generalize!!
            end_index = (i+1)*element.num_integration_points*3
            indices = np.arange(start_index, end_index)
            self.stress_map[np.ix_(indices, element_dofs)] += local_stress_map
            start_index = i*element.num_integration_points
            end_index = (i+1)*element.num_integration_points
            indices = np.arange(start_index, end_index)
            self.integration_coordinates[np.ix_(indices, np.array([0,1]))] = element.integration_coordinates


    '''
    Runs the preliminary setup to solve the problem.
    '''
    def setup(self):
        for element in self.mesh.elements:
            element.assemble()

        self.assemble_k()
        self.assemble_stress_map()
        self.K = self.K.tocsc()
        self.F = self.F.tocsc()
        self.stress_map.tocsc()      

        self.free_dof = np.setdiff1d(self.total_dof, self.prescribed_dof)   # all dof that are not prescribed are free
        
        self.K_ff = self.K[np.ix_(self.free_dof, self.free_dof)]
        self.K_fp = self.K[np.ix_(self.free_dof, self.prescribed_dof)]
        self.K_pf = self.K_fp.T
        self.K_pp = self.K[np.ix_(self.prescribed_dof, self.prescribed_dof)]

        self.F_f = self.F[self.free_dof]
        self.F_p = self.F[self.prescribed_dof]    # don't know yet

        self.U_f = self.U[self.free_dof]          # don't know yet
        self.U_p = self.U[self.prescribed_dof]

        # self.K = self.K.tocsc()   # TODO figure out when is more efficient
        # self.F = self.F.tocsc() 


    '''
    Solves the static FEA problem.
    '''
    def evaluate(self):
        # self.U = np.linalg.solve(self.K[np.ix_(self.free_dof, self.free_dof)], self.F[self.free_dof])

        a = self.K_ff
        b = self.F_f - self.K_fp.dot(self.U_p)
        # b = self.F_f - np.dot(self.K_fp, self.U_p)
        # self.U_f = np.linalg.solve(a, b)
        self.U_f = spsolve(a, b)
        # self.F_p = np.dot(self.K_pf, self.U_f) + np.dot(self.K_pp, self.U_p)
        self.F_p = self.K_pf.dot(self.U_f).reshape((self.K_pf.shape[0], 1)) + self.K_pp.dot(self.U_p)

        self.U[self.free_dof] = self.U_f
        # self.U[self.prescribed_dof] = self.U_p    # known already
        self.U = self.U.todense()

        # self.F[self.free_dof] = self.F_f          # known already
        self.F[self.prescribed_dof] = self.F_p

    
    def setup_dynamics(self):
        self.setup()
        self.size_K_ff = self.K_ff.shape[0]
        mass_per_node = 1
        M = sps.eye(self.size_K_ff, format='csc')*mass_per_node
        M_inv = sps.linalg.inv(M)
        self.dampening_per_node = 50.
        dampening = sps.eye(self.size_K_ff, format='csc')*self.dampening_per_node
        # A_10 = -spsolve(M, self.K_ff)
        A_10 = -M_inv.dot(self.K_ff)
        # A_11 = -spsolve(M, dampening)
        A_11 = -M_inv.dot(dampening)
        A = sps.lil_matrix((2*self.size_K_ff, 2*self.size_K_ff))
        # A[:size_K_ff, :size_K_ff] = sps.zeros     # not necessary for sparse
        A[:self.size_K_ff, self.size_K_ff:] = sps.eye(self.size_K_ff, format='csc')
        A[self.size_K_ff:, :self.size_K_ff] = A_10
        A[self.size_K_ff:, self.size_K_ff:] = A_11
        A = A.tocsc()
        # print('Eigenvalues of A: ', sps.linalg.eigs(A))
        self.eigs = sps.linalg.eigs(A)
        self.A_inv = sps.linalg.inv(A)
        # print('inv A: ', sps.linalg.inv(A))   # made sure not singular.
        # print('det: ', np.linalg.det(A.todense()))
        # A_inv = sps.linalg.inv(A)
        # print('NORM: ', np.linalg.norm((A.dot(A_inv)).todense() - np.eye(A.shape[0])))

        B = sps.lil_matrix((2*self.size_K_ff, self.size_K_ff))
        # B[size_K_ff:,:] = spsolve(M, sps.eye(size_K_ff))
        B[self.size_K_ff:,:] = M_inv
        B = B.tocsc()

        C = sps.lil_matrix((self.size_K_ff, 2*self.size_K_ff))
        C[:, :self.size_K_ff] = sps.eye(self.size_K_ff)
        C = C.tocsc()

        self.A = A
        self.B = B
        self.C = C


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

            rigid_body_displacement = self.reshaped_Ff.sum(0).T * (delta_t**2)/2
            self.rigid_body_disp = np.hstack((self.rigid_body_disp, rigid_body_displacement))

            # all of the time steps can be built, and then all of the matrix exponentials can be calculated in parallel
            # These matrix exponentials can then be plugged into equations to get all states in parallel.

        self.U = np.zeros((self.num_total_dof, self.nt+1))
        self.U[self.free_dof,:] = self.C.dot(self.x)
        self.U_per_time_step = self.U.reshape((-1, self.nt+1))
        self.U_per_dim_per_time = self.U.reshape((self.num_nodes, -1, self.nt+1))



    '''
    Plots the solution.
    '''
    def plot_displacements(self):
        nodes = self.mesh.nodes
        U_reshaped = self.U.reshape((self.num_nodes, -1))
        max_x_dist = np.linalg.norm(max(nodes[:,0]) - min(nodes[:,0]))
        max_y_dist = np.linalg.norm(max(nodes[:,1]) - min(nodes[:,1]))
        scale_dist = np.linalg.norm(np.array([max_x_dist, max_y_dist]))
        if np.linalg.norm(self.U) != 0:
            visualization_scaling_factor = scale_dist*0.1/max(np.linalg.norm(U_reshaped, axis=1))
        else:
            visualization_scaling_factor = 0
        deformed_nodes = nodes + U_reshaped*visualization_scaling_factor
        
        
        if len(nodes.shape) == 2:
            plt.plot(nodes[:,0], nodes[:,1], 'ko')
            plt.plot(deformed_nodes[:,0], deformed_nodes[:,1], 'r*')
            # for i in range(self.num_nodes):
            #     connections_indices = self.mesh.connections[i]
            #     j = 0
            #     for index in connections_indices:
            #         this_node = nodes[i]
            #         other_node = nodes[index]
            #         plt.plot([this_node[0], other_node[0]], [this_node[1], other_node[1]], '--k')
            #         this_deformed_node = deformed_nodes[i]  # I don't understand why the deformed come out as 2d arrays
            #         other_deformed_node = deformed_nodes[index].reshape(deformed_nodes.shape[1])
            #         plt.plot([this_deformed_node[0, 0], other_deformed_node[0, 0]] , [this_deformed_node[0, 1], other_deformed_node[0, 1]], '--r')

            plt.title(f'Scaled Discplacements (x{visualization_scaling_factor}) with {self.num_elements} Elements')
            plt.xlabel('x1 (m)')
            plt.ylabel('x2 (m)')
            plt.legend(('Initial Structure', 'Deformed Structure'))
            plt.show()
        else:
            print("WARNING: Plotting not set up for anything other than 2d right now.")


    '''
    Plots dynamics
    '''
    def plot_dynamics(self, show_plots=False, save_plots=False, video_file_name=None, fps=10):
        nodes = self.mesh.nodes
        max_x_dist = np.linalg.norm(max(nodes[:,0]) - min(nodes[:,0]))
        max_y_dist = np.linalg.norm(max(nodes[:,1]) - min(nodes[:,1]))
        scale_dist = np.linalg.norm(np.array([max_x_dist, max_y_dist]))
        scaling_norms = np.zeros(self.nt)
        for i in range(self.nt):
            scaling_norms[i] = np.max(np.linalg.norm(self.U_per_dim_per_time[:,:,i], axis=1))
        visualization_scaling_factor = scale_dist*0.1/np.max(scaling_norms)

        print('Plotting...')
        for i, t in enumerate(self.t_eval):
            plt.figure()
            plt.plot(self.mesh.nodes[:,0], self.mesh.nodes[:,1], 'bo')

            U_reshaped_plot = self.U_per_dim_per_time[:,:,i]
            deformed_nodes = nodes + U_reshaped_plot*visualization_scaling_factor
            plt.plot(deformed_nodes[:,0], deformed_nodes[:,1], 'r*')
            plt.title(f'Displacement at t={t: 9.5f}')
            plt.xlabel(f'x (m*{visualization_scaling_factor})')
            plt.ylabel(f'y (m*{visualization_scaling_factor})')
            if save_plots or video_file_name is not None:
                plt.savefig(f'plots/displacement_plot_at_t_{t:9.6f}.png', bbox_inches='tight')
            if show_plots:
                plt.show()
            plt.close()

        if video_file_name is not None:
            print('Creating Video...')
            image_folder = 'plots'
            images = [f'displacement_plot_at_t_{t:9.6f}.png' for t in self.t_eval]
            frame = cv2.imread(os.path.join(image_folder, images[0]))
            height, width, layers = frame.shape

            video = cv2.VideoWriter(video_file_name, cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))

            for image in images:
                video.write(cv2.imread(os.path.join(image_folder, image)))

            cv2.destroyAllWindows()
            video.release()


    '''
    Calculates the stresses at the integration points.
    '''
    def calc_stresses(self):
        self.stresses = self.stress_map.dot(self.U)
        self.stresses_per_point = self.stresses.reshape((-1,3))

    
    def calc_dynamic_stresses(self):
        print(self.stress_map.shape)
        print(self.U_per_time_step.shape)
        self.stresses = self.stress_map.dot(self.U_per_time_step)
        print(self.stresses.shape)
        self.stresses_per_point = self.stresses.reshape((-1,3,self.nt+1))
        return self.stresses_per_point


    def plot_stresses(self, stress_type):
        if stress_type == 'x' or stress_type == 'xx':
            stress_index = 0
        elif  stress_type == 'y' or stress_type == 'yy':
            stress_index = 1
        elif  stress_type == 'tao' or stress_type == 'xy':
            stress_index = 2

        plt.figure()
        stresses_per_integration_point = self.stresses.reshape((-1,3))
        plot_stresses = []
        for i in range(stresses_per_integration_point.shape[0]):
            plot_stresses.append(stresses_per_integration_point[i,stress_index])
        plt.scatter(self.integration_coordinates[:,0], self.integration_coordinates[:,1], c=plot_stresses)

        plt.title(f'Stress (sigma_{stress_type}) Color Plot with {self.num_elements} Elements')
        plt.xlabel('x1 (m)')
        plt.ylabel('x2 (m)')
        plt.colorbar()
        plt.show()

    def plot_dynamic_stresses(self, stress_type, time_step=None, dof=None):
        if stress_type == 'x' or stress_type == 'xx':
            stress_index = 0
        elif  stress_type == 'y' or stress_type == 'yy':
            stress_index = 1
        elif  stress_type == 'tao' or stress_type == 'xy':
            stress_index = 2
        
        if time_step is None and dof is None:
            print("Please supply either a time steo or a dof.")
        elif time_step is not None and dof is None:
            # plot_stresses = self.stresses_per_point[:,:,time_step]
            plot_stresses = []
            for i in range(self.stresses_per_point.shape[0]):
                plot_stresses.append(self.stresses_per_point[i,stress_index,time_step])
            plt.figure()
            plt.scatter(self.integration_coordinates[:,0], self.integration_coordinates[:,1], c=plot_stresses)

            plt.title(f'Stress (sigma_{stress_type}) Color Plot with {self.num_elements} Elements')
            plt.xlabel('x1 (m)')
            plt.ylabel('x2 (m)')
            plt.colorbar()
            plt.show()
        elif dof is not None and time_step is None:
            t = self.t_eval
            plot_stresses = self.stresses_per_point[dof, stress_index, :]
            plt.figure()
            plt.plot(t, plot_stresses, '-bo')
            plt.title(f'Stress (sigma_{stress_type}) Color Plot with {self.num_elements} Elements')
            plt.xlabel('t (s)')
            plt.ylabel('stress (Pa)')
            plt.show()
        else:
            print(self.stresses_per_point[dof, stress_index, time_step])

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
