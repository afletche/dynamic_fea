from turtle import pen
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

        self.point_masses = []

        # self.initialize_ode()


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


    def assemble_mass_matrix(self):
        for element in self.elements:
            element_density = element.density
            element_mass = element_density*element.thickness*element.area
            element_mass_per_node = element_mass/element.num_nodes
            element_dofs = self.nodes_to_dof_indices(element.node_map)
            for dof in element_dofs:
                self.M[dof, dof] += element_mass_per_node

        for point_mass in self.point_masses:
            node = point_mass[0]
            mass = point_mass[1]
            element_dof = self.nodes_to_dof_indices([node])
            self.M[np.ix_(element_dof, element_dof)] += np.array([mass])


    '''
    Adds a point mass to the structure.

    Inputs:
    - g : float : gravitational constant (9.81 for Earth)
    - masses : List : list of Lists of locations_node and masses: [[node1, mass1],[node2, mass2],...]
    '''
    def add_point_masses(self, masses):
        for point_mass in masses:
            self.point_masses.append(point_mass)


    '''
    Sets a default material for each element in the mesh

    Inputs:
    - material : Material : object containing material properties
    '''
    def set_material(self, material):
        self.material = material


    # '''
    # Assembles a map that will individually calculate the strain energy in each element.
    # '''
    # def assemble_strain_energy_per_element_map(self):
    #     # for each element, stack the element stiffness matrix
    #     # in order to calculate a vector of strain energies (one for each element)/
    #     element_number = 0
    #     for element in self.elements:
    #         K_element = element.K
    #         element_dofs = self.nodes_to_dof_indices(element.node_map)
    #         row_indices = element_dofs + element_number*self.num_total_dof
    #         self.strain_energy_per_element_map[np.ix_(row_indices, element_dofs)] = K_element
    #         self.strain_energy_per_element_map[np.ix_(row_indices, element_dofs)] = K_element/element.volume
    #         element_number += 1
    #     self.strain_energy_per_element_map.tocsc()
    #     self.strain_energy_density_per_element_map.tocsc()

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
    Apply toppology optimization densities
    NOTE: Need to add density smoothing.
    '''
    def evaluate_topology(self, densities, simp_penalization_factor=3, ramp_penalization_factor=None, filter_radius=None):

        filtered_densities = densities
        if filter_radius is not None:
            W = self.apply_density_aggregation_filter(filter_radius)
            filtered_densities = W.dot(filtered_densities)
        else:
            self.density_filter = None

        penalized_densities = filtered_densities
        if simp_penalization_factor is not None:
            penalized_densities = densities**simp_penalization_factor

        if ramp_penalization_factor is not None:
            penalized_densities = penalized_densities/(1 + ramp_penalization_factor*(1-penalized_densities))
        
        for i, element in enumerate(self.elements):
            element.evaluate_topology(penalized_densities[i])

        self.initialize_ode()
        self.assemble_k()
        self.assemble_mass_matrix()

        self.K = self.K.tocsc()
        self.M = self.M.tocsc()


    def apply_density_aggregation_filter(self, radius):
        element_midpoints = np.zeros((self.num_elements, 2))
        for i, element in enumerate(self.elements):
            element_midpoints[i,:] = element.calc_midpoint()

        W = np.zeros((self.num_elements, self.num_elements))

        for i, element in enumerate(self.elements):
            weight_sum = 0
            for j in range(len(self.elements)):
                distance = np.linalg.norm(element_midpoints[i,:] - element_midpoints[j,:])
                coeff = radius - distance
                
                if coeff > 0:
                    W[i,j] = coeff
                    weight_sum += coeff
            
            W[i,:] = W[i,:]/weight_sum

        self.density_filter = W

        return W


    '''
    Sets up the data structures for the ODE (preallocates).
    '''
    def initialize_ode(self):
        self.K = sps.lil_matrix((self.num_total_dof, self.num_total_dof))
        self.M = sps.lil_matrix((self.num_total_dof, self.num_total_dof))
        # self.strain_energy_per_element_map = sps.lil_matrix((self.num_total_dof*self.num_elements, self.num_total_dof))
        # self.strain_energy_density_per_element_map = sps.lil_matrix((self.num_total_dof*self.num_elements, self.num_total_dof))


    def setup(self):
        for element in self.elements:
            element.assemble()

        self.initialize_ode()
        self.assemble_k()
        self.assemble_mass_matrix()
        self.assemble_connectivity()
        # self.assemble_strain_energy_per_element_map()


class StructuredMesh(UnstructuredMesh):
    '''
    Class for representing meshes.
    '''
    def __init__(self, name, nodes) -> None:
        self.name = name
        self.nodes = nodes


class UniformStructredMesh(StructuredMesh):
    '''
    Class for representing meshes.
    '''
    def __init__(self, name, nodes) -> None:
        self.name = name
        self.nodes = nodes
