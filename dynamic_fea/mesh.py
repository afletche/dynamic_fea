from typing import List

import numpy as np
import scipy.sparse as sps

class Mesh:
    '''
    Class for representing meshes.
    '''
    def __init__(self, name) -> None:
        self.name = name
        # self.nodes = nodes  # I don't think I actually want this here.





class StructuredMesh(Mesh):
    '''
    Class for representing meshes.
    '''
    def __init__(self, name, nodes) -> None:
        self.name = name
        self.nodes = nodes


class UnstructuredMesh(Mesh):
    '''
    Class for representing meshes.
    '''
    def __init__(self, name, nodes, elements) -> None:
        super().__init__(name)
        self.name = name
        self.nodes = nodes
        self.elements = elements
        self.num_nodes = nodes.shape[0]
        self.num_elements = len(elements)
        self.dimensions_spanned = nodes.shape[-1]
        self.num_total_dof = self.num_nodes*self.dimensions_spanned

        self.initialize_ode()



    '''
    Gets the DOF indices for the corresponding nodes.
    '''
    def nodes_to_dof_indices(self, node_map):
        dof_indices = np.zeros(2*len(node_map))
        for i in range(len(node_map)):
            dof_indices[2*i] = self.dimensions_spanned*node_map[i]
            dof_indices[2*i+1] = self.dimensions_spanned*node_map[i]+1
        return dof_indices.astype(int)


    '''
    Assembles the global stiffness matrix based on the local stiffness matrices (using element objects)
    '''
    def assemble_k(self):
        # For each element, add its local stiffness contribution
        for element in self.elements:
            K_element = element.K
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            self.K[np.ix_(element_dofs, element_dofs)] += K_element

    '''
    Assembles the connectivity (mostly for plotting).
    '''
    def assemble_connectivity(self):
        self.node_connectivity = {}
        for element in self.elements:
            for node_index in element.node_connectivity.keys():
                if node_index in self.node_connectivity.keys():
                    self.node_connectivity[node_index] += element.node_connectivity[node_index]
                else:
                    self.node_connectivity[node_index] = element.node_connectivity[node_index]

    '''
    Sets up the data structures for the ODE (preallocates).
    '''
    def initialize_ode(self):
        self.K = sps.lil_matrix((self.num_total_dof, self.num_total_dof))


    def setup(self):
        for element in self.elements:
            element.assemble()

        self.assemble_k()
        self.assemble_connectivity()
