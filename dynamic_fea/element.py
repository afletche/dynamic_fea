import numpy as np
from numpy.core.fromnumeric import shape


"""Class for defining an element type in FEA."""
class Element:

    def __init__(self, nodes, node_map, material, thickness=None) -> None:
        self.nodes = nodes
        self.node_map = node_map
        self.material = material
        self.thickness = thickness

        
        
    def center(self):
        sumx = 0
        sumy = 0
        for i in range(4):
            sumx += self.nodes[i, 0]
            sumy += self.nodes[i, 1]
        return np.array([sumx / 4, sumy / 4])


'''
The class for a quad element. Currently only set up for 4 node quads.
'''
class QuadElement(Element):
    
    
    def __init__(self, nodes, node_map, material, thickness) -> None:
        super().__init__(nodes, node_map, material, thickness)

        """ calculate node connectivity"""
        self.node_connectivity = {}
        self.node_connectivity[node_map[0]] = [node_map[1], node_map[2]]
        self.node_connectivity[node_map[1]] = [node_map[0], node_map[3]]
        self.node_connectivity[node_map[2]] = [node_map[0], node_map[3]]
        self.node_connectivity[node_map[3]] = [node_map[1], node_map[2]]

        oosqrt3 = 1/np.sqrt(3)
        self.parametric_integration_coordinates = np.array([[-oosqrt3, -oosqrt3], [-oosqrt3, oosqrt3], [oosqrt3, -oosqrt3], [oosqrt3, oosqrt3]])


    '''
    Calculates the local stiffness matrix of an element.
    '''
    def calc_k_local(self):
        def calc_k_local_integrand(zeta, eta):
            E_local = self.calc_e_local_plane_strain()
            pn_ppara = self.calc_jacobian_parametric_to_shape(zeta, eta)
            J = self.calc_j(pn_ppara)
            B = self.calc_b(J, pn_ppara)
            det_J = np.linalg.det(J)
            integrand = np.dot(B.T, np.dot(E_local, np.dot(B, det_J)))
            return integrand

        K_local = self.thickness*self.second_order_double_integral_gauss_quadrature(calc_k_local_integrand)
        self.K_local = K_local
        return K_local


    '''
    A Practically hard-coded 2x2 Gauss-Quadrature integration with bounds [-1,1] along 2 dimensions.
    '''
    def second_order_double_integral_gauss_quadrature(self, integrand):
        oor3 = 1/np.sqrt(3)
        # TODO Can make this into an element (parent) method that uses the child element attribute for parametric coordinates and weights.
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

    
    def calc_jacobian_parametric_to_shape(self, eta, zeta):
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

    def evaluate_integration_coordinates(self):
        nodes_x = self.nodes[:,0]
        nodes_y = self.nodes[:,1]

        self.integration_coordinates = np.zeros((4,2))
        for i in range(4):
            zeta = self.parametric_integration_coordinates[i, 0]
            eta = self.parametric_integration_coordinates[i, 1]
            shape_funcs = np.array([
                self.evaluate_n_1(zeta, eta),
                self.evaluate_n_2(zeta, eta),
                self.evaluate_n_3(zeta, eta),
                self.evaluate_n_4(zeta, eta)
            ])

            # TODO Pretty sure x, y, and potentially z can be done in one operation
            x = np.dot(nodes_x, shape_funcs)
            y = np.dot(nodes_y, shape_funcs)
            self.integration_coordinates[i,:] = np.array([x, y])


    def evaluate_n_1(self, zeta, eta):
        return (1-zeta)*(1-eta)/4
    
    def evaluate_n_2(self, zeta, eta):
        return (1-zeta)*(1+eta)/4

    def evaluate_n_3(self, zeta, eta):
        return (1+zeta)*(1-eta)/4

    def evaluate_n_4(self, zeta, eta):
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
    Calculates the stresses at the integration points.
    '''
    def calc_stresses(self, U):
        num_int_points = 4   # TODO generalize for non-4-node quads!!
        num_stresses = 3     # TODO generalize for non-2D!!
        stresses = np.zeros((num_int_points, num_stresses))
        # integration points of this element TODO generalize for non-4-node quads!!?
        oor3 = 1/np.sqrt(3)
        zeta_eta = self.parametric_integration_coordinates

        E_local = self.E_local
        self.U = U

        for j in range(len(zeta_eta)):
            pn_ppara = self.calc_jacobian_parametric_to_shape(zeta_eta[j][0], zeta_eta[j][1])
            J = self.calc_j(pn_ppara)
            B_local = self.calc_b(J, pn_ppara)
            stresses[j,:] = np.dot(E_local, np.dot(B_local, U)).reshape((num_stresses,))
        self.stresses = stresses
        return stresses

    
class TriangleElement(Element):

    def __init__(self, nodes, node_map) -> None:
        super().__init__(nodes, node_map)

    