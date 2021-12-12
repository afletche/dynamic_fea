'''
Author: Andrew Fletcher
Date: 11/24

This class is for performing 2D structural FEA problems.
'''

from os import name
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


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
        self.F = self.F.tocsc()        


    '''
    Apples the boundary conditions to the FEA problem.
    
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


    # '''
    # Assembles the global stiffness matrix based on the local stiffness matrices
    # '''
    # def assemble_k(self):
    #     # For each element, add its local stiffness contribution
    #     for i in range(self.num_elements):
    #         element_nodes = self.mesh.nodes[self.mesh.element_nodes[i,:],:]
    #         local_K = self.calc_k_local(element_nodes[:,0], element_nodes[:,1], self.material, self.design_representation.components[0].properties['thickness'])
    #         element_dofs = self.nodes_to_dof_indices(self.mesh.element_nodes[i,:])
    #         self.K[np.ix_(element_dofs, element_dofs)] += local_K

    
    '''
    Assembles the global stiffness matrix based on the local stiffness matrices (using element objects)
    '''
    def assemble_k(self):
        # For each element, add its local stiffness contribution
        for i in range(self.num_elements):
            element = self.mesh.elements[i]
            local_K = element.calc_k_local()
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            self.K[np.ix_(element_dofs, element_dofs)] += local_K
        self.K = self.K.tocsc()

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
    Runs the preliminary setup to solve the problem.
    '''
    def setup(self):
        self.assemble_k()

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
    Solves the FEA problem.
    '''
    def evaluate(self):
        # self.U = np.linalg.solve(self.K[np.ix_(self.free_dof, self.free_dof)], self.F[self.free_dof])

        a = self.K_ff
        print(a)
        b = self.F_f - self.K_fp.dot(self.U_p)
        # b = self.F_f - np.dot(self.K_fp, self.U_p)
        # self.U_f = np.linalg.solve(a, b)
        self.U_f = spsolve(a, b)
        print('line252, U: ', self.U_f)
        # self.F_p = np.dot(self.K_pf, self.U_f) + np.dot(self.K_pp, self.U_p)
        self.F_p = self.K_pf.dot(self.U_f).reshape((self.K_pf.shape[0], 1)) + self.K_pp.dot(self.U_p)

        self.U[self.free_dof] = self.U_f
        # self.U[self.prescribed_dof] = self.U_p    # known already
        self.U = self.U.todense()

        # self.F[self.free_dof] = self.F_f          # known already
        self.F[self.prescribed_dof] = self.F_p


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
    Calculates the stresses at the integration points.
    '''
    def calc_stresses(self):
        num_int_points = 4   # TODO generalize for non-4-node quads!!
        num_stresses = 3     # TODO generalize for non-2D!!
        stresses = np.zeros((self.num_elements, num_int_points, num_stresses))
        for i in range(self.num_elements):
            element = self.mesh.elements[i]
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            element_U = self.U[element_dofs]
            stresses[i,:,:] = element.calc_stresses(element_U)
        self.stresses = stresses


    def plot_stresses(self, stress_type):
        if stress_type == 'x' or stress_type == 'xx':
            index = 0
        elif  stress_type == 'y' or stress_type == 'yy':
            index = 1
        elif  stress_type == 'tao' or stress_type == 'xy':
            index = 2

        plt.figure()
        j = 0
        integration_points = np.zeros((1,2))
        stress_values = np.zeros((1,3))
        for element in self.mesh.elements:
            element.evaluate_integration_coordinates()
            j += 1
            for i in range(len(element.integration_coordinates)):
                integration_point = element.integration_coordinates[i,:]
                integration_points = np.vstack((integration_points, integration_point))
                stress_values = np.vstack((stress_values, element.stresses[i, :]))
        plt.scatter(integration_points[1:,0], integration_points[1:,1], c=stress_values[1:,index])

        plt.title(f'Stress (sigma_{stress_type}) Color Plot with {self.num_elements} Elements')
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
