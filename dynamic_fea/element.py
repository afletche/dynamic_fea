import numpy as np

"""Class for defining an element type in FEA."""
class Element:

    def __init__(self, nodes, node_map, material) -> None:
        self.nodes = nodes
        self.node_map = node_map
        self.material = material

        
        
    def center(self):
        sumx = 0
        sumy = 0
        for i in range(4):
            sumx += self.nodes[i, 0]
            sumy += self.nodes[i, 1]
        return np.array([sumx / 4, sumy / 4])


    '''
    Calculates the stiffness matrix of the element.
    '''
    def calc_k(self):
        return

    '''
    Takes input information and sets up information in usable matrices/structures.
    '''
    def setup(self):
        self.calc_k()



'''
The class for a quad element. Currently only set up for 4 node quads.
'''
class QuadElement(Element):
    
    
    def __init__(self, plane_strain_or_stress, nodes, node_map, material, thickness) -> None:
        super().__init__(nodes, node_map, material)
        self.plane_strain_or_stress = plane_strain_or_stress
        self.thickness = thickness

        """ list node connectivity"""
        self.node_connectivity = {}
        self.node_connectivity[node_map[0]] = [node_map[1], node_map[2]]
        self.node_connectivity[node_map[1]] = [node_map[0], node_map[3]]
        self.node_connectivity[node_map[2]] = [node_map[0], node_map[3]]
        self.node_connectivity[node_map[3]] = [node_map[1], node_map[2]]

        oosqrt3 = 1/np.sqrt(3)
        self.parametric_integration_coordinates = np.array([[-oosqrt3, -oosqrt3], [-oosqrt3, oosqrt3], [oosqrt3, -oosqrt3], [oosqrt3, oosqrt3]])
        self.num_integration_points = 4
        self.num_nodes = 4

        self.density0 = self.material.density
        self.density = self.material.density
        self.E0 = self.material.E
        self.E = self.material.E


    '''
    Calculates the local stiffness matrix of an element.
    '''
    def calc_k(self):
        def calc_k_integrand(zeta, eta):
            if self.plane_strain_or_stress == 'strain':
                self.E_local = self.calc_e_local_plane_strain()
            elif self.plane_strain_or_stress == 'stress':
                self.E_local = self.calc_e_local_plane_stress()
            else:
                Warning("Please input 'strain' or 'stress' for the plane_strain_or_stress")
            self.pn_ppara = self.calc_jacobian_parametric_to_shape(zeta, eta)
            self.J = self.calc_j(self.pn_ppara)
            self.B = self.calc_b(self.J, self.pn_ppara)
            self.det_J = np.linalg.det(self.J)
            integrand = np.dot(self.B.T, np.dot(self.E_local, np.dot(self.B, self.det_J)))
            return integrand

        self.K = self.thickness*self.second_order_double_integral_gauss_quadrature(calc_k_integrand)
        self.K0 = self.K.copy()
        return self.K


    '''
    A Practically hard-coded 2x2 Gauss-Quadrature integration with bounds [-1,1] along 2 dimensions.
    '''
    def second_order_double_integral_gauss_quadrature(self, integrand):
        oor3 = 1/np.sqrt(3)
        # TODO Can make this into an element (parent) method that uses the child element attribute for parametric coordinates and weights.
        # -- clarififcation: element (parent) as in the base element class.
        value = integrand(oor3, oor3) + integrand(-oor3, oor3) + integrand(oor3, -oor3) + integrand(-oor3, -oor3)
        return value

    
    '''
    Calculates the local E matrix of an element in plane strain.
    '''
    def calc_e_local_plane_strain(self):
        E = self.material.E
        nu = self.material.nu
        E_local = E/((1+nu)*(1-2*nu))*np.array([[1-nu, nu, 0], [nu, 1-nu, 0], [0, 0, (1-2*nu)/2]])
        self.E_local = E_local
        return E_local


    '''
    Calculates the local E matrix of an element in plane stress.
    '''
    def calc_e_local_plane_stress(self):
        E = self.material.E
        nu = self.material.nu
        E_local = E/(1 - nu**2)*np.array([[1, nu, 0],
                                          [nu, 1, 0],
                                          [0, 0, (1-nu)/2]])
        self.E_local = E_local
        return E_local


    '''
    Calculates the Jacobian matrix to map from parametric to cartesian coordinates.
    '''
    def calc_j(self, jacobian_parametric_to_shape):
        x = self.nodes[:,0]
        y = self.nodes[:,1]
        dx_dzeta = np.dot(x, jacobian_parametric_to_shape[0,:])
        dx_deta = np.dot(x, jacobian_parametric_to_shape[1,:])
        dy_dzeta = np.dot(y, jacobian_parametric_to_shape[0,:])
        dy_deta = np.dot(y, jacobian_parametric_to_shape[1,:])
        J = np.array([[dx_dzeta, dy_dzeta], [dx_deta, dy_deta]])
        return J


    '''
    Calculates the local B matrix of an element.
    '''
    def calc_b(self, J, jacobian_parametric_to_shape):
        pn_pcart = np.linalg.solve(J, jacobian_parametric_to_shape)
        # pn_pcart = np.dot(np.linalg.inv(J), jacobian_parametric_to_shape)
        B = np.zeros((3, 2*pn_pcart.shape[1]))    # 3 because xx, yy, and xy
        for i in range(pn_pcart.shape[1]):
            B[0,i*2] = pn_pcart[0,i]
            B[1,i*2+1] = pn_pcart[1,i]
            B[2,i*2] = pn_pcart[1,i]
            B[2,i*2+1] = pn_pcart[0,i]
        return B

    
    def calc_jacobian_parametric_to_shape(self, zeta, eta):
        dn1_dzeta = self.evaluate_dn_dzeta_1(eta)
        dn2_dzeta = self.evaluate_dn_dzeta_2(eta)
        dn3_dzeta = self.evaluate_dn_dzeta_3(eta)
        dn4_dzeta = self.evaluate_dn_dzeta_4(eta)
        dn1_deta = self.evaluate_dn_deta_1(zeta)
        dn2_deta = self.evaluate_dn_deta_2(zeta)
        dn3_deta = self.evaluate_dn_deta_3(zeta)
        dn4_deta = self.evaluate_dn_deta_4(zeta)

        pn_ppara = np.array([
            [dn1_dzeta, dn2_dzeta, dn3_dzeta, dn4_dzeta],
            [dn1_deta, dn2_deta, dn3_deta, dn4_deta]
        ])

        return pn_ppara

    # def evaluate_integration_coordinates(self):
    #     nodes_x = self.nodes[:,0]
    #     nodes_y = self.nodes[:,1]

    #     self.integration_coordinates = np.zeros((4,2))
    #     for i in range(4):
    #         zeta = self.parametric_integration_coordinates[i, 0]
    #         eta = self.parametric_integration_coordinates[i, 1]
    #         shape_funcs = np.array([
    #             self.evaluate_n_1(zeta, eta),
    #             self.evaluate_n_2(zeta, eta),
    #             self.evaluate_n_3(zeta, eta),
    #             self.evaluate_n_4(zeta, eta)
    #         ])

    #         # TODO Pretty sure x, y, and potentially z can be done in one operation
    #         x = np.dot(nodes_x, shape_funcs)
    #         y = np.dot(nodes_y, shape_funcs)
    #         self.integration_coordinates[i,:] = np.array([x, y])

    '''
    Assembles the map for evaluating the gauss-quadrature integration points.
    '''
    def assemble_integration_coordinates_map(self):
        self.integration_coordinates_map = np.array([
            self.evaluate_n_1(self.parametric_integration_coordinates),
            self.evaluate_n_2(self.parametric_integration_coordinates),
            self.evaluate_n_3(self.parametric_integration_coordinates),
            self.evaluate_n_4(self.parametric_integration_coordinates)
        ]).T

        return self.integration_coordinates_map

    '''
    Evaluates the gauss-quadrature integration points.
    '''
    def evaluate_integration_coordinates(self, U=None):
        if U is None:
            U = np.zeros_like(self.nodes, dtype=float)
        if U.shape[1] == 1:
            U = U.reshape(self.nodes.shape)
        deformed_nodes = self.nodes + U
        self.integration_coordinates = np.dot(self.integration_coordinates_map, deformed_nodes)
        return self.integration_coordinates

    
    ''' 
    Calculates the element midpoint.
    '''
    def calc_midpoint(self, U=None):
        if U is None:
            U = np.zeros_like(self.nodes, dtype=float)

        new_shape = (-1,) + self.nodes.shape    # (nt+1, num_dof, 2 or 3)
        U = U.reshape(new_shape)
        nt_plus_1 = U.shape[0]
        nodes = self.nodes.copy()

        nodes = np.broadcast_to(nodes, (nt_plus_1, *nodes.shape))
        deformed_nodes = self.nodes + U

        zeta_eta = np.array([[0., 0.]]) # midpoint
        shape_funcs = np.array([
            self.evaluate_n_1(zeta_eta),
            self.evaluate_n_2(zeta_eta),
            self.evaluate_n_3(zeta_eta),
            self.evaluate_n_4(zeta_eta)
        ]).T

        self.midpoint = shape_funcs.dot(deformed_nodes)
        return self.midpoint


    def evaluate_n_1(self, zeta_eta):
        zeta = zeta_eta[:,0]
        eta = zeta_eta[:,1]
        return (1-zeta)*(1-eta)/4
    
    def evaluate_n_2(self, zeta_eta):
        zeta = zeta_eta[:,0]
        eta = zeta_eta[:,1]
        return (1-zeta)*(1+eta)/4

    def evaluate_n_3(self, zeta_eta):
        zeta = zeta_eta[:,0]
        eta = zeta_eta[:,1]
        return (1+zeta)*(1-eta)/4

    def evaluate_n_4(self, zeta_eta):
        zeta = zeta_eta[:,0]
        eta = zeta_eta[:,1]
        return (1+zeta)*(1+eta)/4


    def evaluate_dn_dzeta_1(self, eta):
        dn1_dzeta = -(1 - eta)/4
        return dn1_dzeta

    def evaluate_dn_dzeta_2(self, eta):
        dn2_dzeta = -(1 + eta)/4
        return dn2_dzeta

    def evaluate_dn_dzeta_3(self, eta):
        dn3_dzeta = (1 - eta)/4
        return dn3_dzeta

    def evaluate_dn_dzeta_4(self, eta):
        dn4_dzeta = (1 + eta)/4
        return dn4_dzeta

    def evaluate_dn_deta_1(self, zeta):
        dn1_deta = -(1 - zeta)/4
        return dn1_deta

    def evaluate_dn_deta_2(self, zeta):
        dn2_deta = (1 - zeta)/4
        return dn2_deta

    def evaluate_dn_deta_3(self, zeta):
        dn3_deta = -(1 + zeta)/4
        return dn3_deta

    def evaluate_dn_deta_4(self, zeta):
        dn4_deta = (1 + zeta)/4
        return dn4_deta


    '''
    Calculates the element area.    
    '''
    def calc_area(self):
        u_vector = self.nodes[2] - self.nodes[0]
        v_vector = self.nodes[1] - self.nodes[0]
        element_area_00_bias = np.linalg.norm(np.cross(u_vector, v_vector))
        u_vector = self.nodes[1] - self.nodes[3]
        v_vector = self.nodes[2] - self.nodes[3]
        element_area_11_bias = np.linalg.norm(np.cross(u_vector, v_vector))
        self.area = (element_area_00_bias + element_area_11_bias)/2
        return self.area


    '''
    Calculates the element volume.
    '''
    def calc_volume(self):
        area = self.calc_area()
        self.volume = area*self.thickness
        return self.volume


    '''
    Calculates the local load vector for self-weight.
    '''
    def calc_self_weight(self, g, rotation=0):
        self.calc_volume()
        density = self.density
        c = np.cos(-rotation)
        s = np.sin(-rotation)
        rotation_matrix = np.array([[c, -s], [s, c]])
        # self.r_g_local = density*self.volume*g*1/4*np.array([0., -1., 0, -1., 0., -1., 0., -1.]).reshape((-1,1))
        self.r_g_local = density*self.volume*g*1/4*rotation_matrix.dot(np.array([[0., 0., 0., 0.],[-1., -1., -1., -1.]])).reshape((-1,1), order='F')
        return self.r_g_local


    '''
    Assembles the stress calculation map
    '''
    def assemble_stress_map(self):
        num_int_points = 4   # TODO generalize for non-4-node quads!!
        num_stresses = 3     # TODO generalize for non-2D!!
        # integration points of this element TODO generalize for non-4-node quads!!?
        zeta_eta = self.parametric_integration_coordinates

        B_stresses = self.calc_stresses_B_mat()
        E_stresses = np.kron(np.eye(num_int_points), self.E_local)

        self.stress_map = np.dot(E_stresses, B_stresses)
        self.stress_map0 = self.stress_map.copy()

        return self.stress_map


    '''
    Calculates the B matrix for the stress calculation map.
    '''
    def calc_stresses_B_mat(self):
        zeta_eta = self.parametric_integration_coordinates

        for j in range(len(zeta_eta)):
            pn_ppara = self.calc_jacobian_parametric_to_shape(zeta_eta[j][0], zeta_eta[j][1])
            J = self.calc_j(pn_ppara)
            if j == 0:
                B_local = self.calc_b(J, pn_ppara)
            else:
                B_local = np.vstack((B_local, self.calc_b(J, pn_ppara)))
                
        return B_local


    '''
    Calculates the stresses at the integration points.
    '''
    def calc_stresses(self, U):
        self.stresses = np.dot(self.stress_map, U).reshape((-1,3))
        return self.stresses


    '''
    Calculates the equivalent of F = K*x
    '''
    def calc_stiffness_force(self, U):
        self.stiffness_force = np.dot(self.K, U)
        return self.stiffness_force


    '''
    Calculates the strain energy of the element.
    strain_energy = 1/2*U*K*U
    '''
    def calc_strain_energy(self, U):
        self.strain_energy = U.T.dot(self.K.dot(U))/2
        self.strain_energy_density = self.strain_energy/self.volume
        return self.strain_energy, self.strain_energy_density


    '''
    Runs the setup for the element which consists of precomputing maps to be used for global maps.
    '''
    def assemble(self):
        super().setup()
        self.calc_k()
        self.assemble_stress_map()
        self.assemble_integration_coordinates_map()
        self.calc_area()
        self.calc_volume()
    

    def evaluate_topology(self, density):
        self.density = self.density0 * density
        self.K = self.K0 * density
        self.stress_map = self.stress_map0 * density

    
class TriangleElement(Element):

    def __init__(self, nodes, node_map) -> None:
        super().__init__(nodes, node_map)




'''
The class for a truss element.
'''
class TrussElement(Element):
    
    
    def __init__(self, nodes, node_map, material, area) -> None:
        super().__init__(nodes, node_map, material)
        self.area = area

        """ list node connectivity"""
        self.node_connectivity = {}
        self.node_connectivity[node_map[0]] = [node_map[1]]
        self.node_connectivity[node_map[1]] = [node_map[0]]

        self.midpoint = (nodes[0] + nodes[1])/2
        self.num_nodes = 2


    '''
    Calculates the element length.
    '''
    def calc_l(self):
        self.displacement_vec = self.nodes[1] - self.nodes[0]
        self.l = np.linalg.norm(self.displacement_vec)
        return self.l

    '''
    Calculates the local stiffness matrix of a truss element.
    '''
    def calc_k(self):
        self.displacement_vec = self.nodes[1] - self.nodes[0]
        self.calc_l()
        self.beta = np.arctan2(self.displacement_vec[1], self.displacement_vec[0])
        c = np.cos(self.beta)
        c2 = c**2
        s = np.sin(self.beta)
        s2 = s**2
        cs = c*s
        self.E = self.material.E

        self.K = self.E*self.area/self.l*np.array([
            [c2, cs, -c2, -cs],
            [cs, s2, -cs, -s2],
            [-c2, -cs, c2, cs],
            [-cs, -s2, cs, s2]])

        return self.K

    
    def set_A(self, area):
        self.area = area

    def set_nodes(self, nodes):
        self.nodes = nodes


    def calc_volume(self):
        self.volume = self.area*self.l
        return self.volume

    '''
    Calculates the local load vector for self-weight.
    '''
    def calc_self_weight(self, g, rotation=0):
        density = self.material.density
        self.calc_l()
        c = np.cos(-rotation)
        s = np.sin(-rotation)
        rotation_matrix = np.array([[c, -s], [s, c]])
        self.r_g_local = density*self.area*g*self.l*rotation_matrix.dot(1/2*np.array([[0., 0.], [-1., -1.]])).reshape((-1,1), order='F')
        return self.r_g_local


    '''
    Assembles a map to calculate the element midpoint from the nodes.    
    '''
    def assemble_midpoint_map(self):
        self.midpoint_map = np.array([0.5, 0.5])
        return self.midpoint_map

    '''
    Evaluates the midoint
    '''
    def evaluate_midpoint_map(self, U=None):
        if U is None:
            U = np.zeros_like(self.nodes)
        
        deformed_nodes = self.nodes + U
        self.midpoint = np.dot(self.midpoint_map, deformed_nodes)
        return self.midpoint

        

    '''
    Assembles the stress calculation map
    '''
    def assemble_stress_map(self):
        
        c = np.cos(self.beta)
        s = np.sin(self.beta)
        rotation_matrix = np.array([[c, s, 0, 0], [0, 0, c, s]])
        relative_disp_map = np.array([-1, 1])
        E = self.material.E
        self.stress_map = E/self.l*np.dot(relative_disp_map, rotation_matrix)

        return self.stress_map

    '''
    Calculates the stresses at the integration points.
    '''
    def calc_stresses(self, U):
        self.stresses = np.dot(self.stress_map, U).reshape((-1,3))
        return self.stresses



    def assemble(self):
        super().setup()
        self.calc_k()
        self.assemble_stress_map()
        self.assemble_midpoint_map()
        self.calc_l()
        self.calc_volume()
