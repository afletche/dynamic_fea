import numpy as np
import matplotlib.pyplot as plt
import shapely
from dynamic_fea.element import Element, QuadElement
from dynamic_fea.material import IsotropicMaterial
from shapely.geometry import Polygon

# calling this class UnstructuredMesh for lack of a better term
# it's just what holds the mesh and connection data as 2d matrices
class UnstructuredMesh:
    
    def __init__(self):
        self.nodes = np.empty((0, 2)) # n x 2 array of x,y coords of n nodes
        self.connections = []         # n x jagged list of all other nodes each node is connected to
        self.structs = []             # list of mesh structures to be stitched together
        self.elements = []            # list of elements in mesh
        self.bvh = []                 # k x k x jagged one-layer bounding volume hierarchy


    def searchDupe(self, loc, nodes, thresh):
        for i in range(len(nodes)):
            if np.linalg.norm(np.array(nodes[i]) - loc) < thresh:
                return i
        return -1


    # this assumes all structures only intersect at outer edges
    def stitch(self, mergeDist, materials, thickness):
        # add each structure to the mesh
        tempNodes = []
        numDupes = 0

        # plt.figure()

        for struct in self.structs:
            srows = struct.shape[0]
            scols = struct.shape[1]
            print(f"loading {srows} x {scols} structure")
            # make a temp list array for easier resizing
            tempNodeIndices = np.empty((srows, scols), int)
            # first iteration adds nodes
            for i in range(srows):
                for j in range(scols):
                    if i == 0 or j == 0 or i == srows-1 or j == scols-1:
                        dupe = self.searchDupe(struct[i, j, :], tempNodes, mergeDist)
                    else:
                        dupe = -1

                    if dupe == -1:
                        # not a duplicate, add node
                        tempNodes.append(struct[i, j, :])
                        tempNodeIndices[i, j] = len(tempNodes) - 1
                        self.connections.append([])
                    else:
                        numDupes += 1
                        tempNodeIndices[i, j] = dupe
            # nodes are all categorized
            dbgElementCounter = 0
            for i in range(srows):
                for j in range(scols):
                    node = tempNodeIndices[i, j]
                    # add edges (this method is cursed but it works so...)
                    for offset in [(0,1), (1,0), (-1,0), (0,-1)]:
                        try:
                            ii = i + offset[0]
                            jj = j + offset[1]
                            if ii < 0 or jj < 0:
                                continue
                            connected = tempNodeIndices[ii, jj]
                            if connected not in self.connections[node]:
                                self.connections[node].append(connected)
                        except IndexError:
                            pass
                    # piggy-back on these for loops to define elements
                    # there are (n-1) x (n-1) elements
                    if i != 0 and j != 0:
                        maps = np.array([tempNodeIndices[i-1][j-1],
                                         tempNodeIndices[i-1][j],
                                         tempNodeIndices[i][j-1],
                                         tempNodeIndices[i][j]])
                        corners = np.array(list(map(lambda ii: tempNodes[ii], maps)))
                        dbgElementCounter += 1
                        
                        element = QuadElement(corners, maps, materials[0], thickness)
                        self.elements.append(element)
                        element_center = np.sum(corners, axis=0)/len(corners)



        # convert temp list to array
        self.nodes = np.array(tempNodes)
        print(f"Completed stitching of {len(self.structs)} bodies, resulting in {len(tempNodes)} nodes, {len(self.elements)} elements after stitching {numDupes} nodes")

        
    def plotStructures(self):
        plt.figure()
        for struct in self.structs:
            plt.scatter(struct[:, :, 0].ravel(), struct[:, :, 1].ravel())
    
    
    def plotMesh(self):
        plt.figure(dpi=200)
        # plt.scatter(self.nodes[:,0], self.nodes[:,1], color='black', zorder=1, s=3)
        # for i in range(len(self.connections)):
        #     for j in self.connections[i]:
        #         if i <= j:
        #             m = np.array([self.nodes[i,:], self.nodes[j,:]])
        #             plt.plot(m[:,0], m[:,1], color='black', zorder=0)
        for element in self.elements:
            c = element.center()
            red = int((element.material.E - 40e9) / ((200e9*0.03/0.2 + 40e9*(1-0.03/0.2)) - 40e9) * 254)
            grn = 255 - red
            plt.scatter([c[0]], [c[1]], marker='s', color=f'#{red:02X}{grn:02X}00', s=4)
        # plt.xticks(np.linspace(0, 16 ,9))
        # plt.yticks(np.linspace(0, 16.5, 12))
        plt.title('Rebar vs Concrete Distribution')
        plt.xlabel('x1 (m)')
        plt.ylabel('x2 (m)')
        plt.show()
        
    
    def addRebar(self, p1, p2, w, matConc, matRebar):
        # get corners of rebar
        barDir = (p2 - p1) / np.linalg.norm(p2 - p1)
        offset = np.dot(np.array([[0, -1], [1, 0]]), barDir) * w
        p1 = p1 + offset / 2
        p2 = p2 + offset / 2
        p3 = p2 - offset
        p4 = p1 - offset
        rebar = Polygon([p1, p2, p3, p4])
        # TODO use bvh to get a subset of self.elements here instead of the full thing
        elements = self.elements
        for element in elements:
            ePoly = Polygon(element.nodes[[0, 1, 3, 2], :])
            fac = rebar.intersection(ePoly).area / ePoly.area
            mat = IsotropicMaterial(f"{fac*100:.0f}% STL", 
                                    E=fac*matRebar.E+(1-fac)*matConc.E,
                                    nu=fac*matRebar.nu+(1-fac)*matConc.nu)
            if (mat.E > element.material.E):
                element.material = mat


if __name__ == "__main__":
    m = UnstructuredMesh()
        
    # stuff the things with dummy stuff
    m.structs.append(np.empty((10, 12, 2)))
    m.structs.append(np.empty((8, 8, 2)))
    for i in range(10):
        for j in range(12):
            m.structs[0][i, j, 0] = 2 * i / 2
            m.structs[0][i, j, 1] = 3 * j / 2
    for i in range(8):
        for j in range(8):
            m.structs[1][i, j, 0] = 2 * i / 2 + 8
            m.structs[1][i, j, 1] = 3 * j / 2
    
    concrete = IsotropicMaterial(name='concrete', E=40e9, nu=0.3)  # replace with concrete material properties
    steel = IsotropicMaterial(name='steel', E=200e9, nu=0.3)  # replace with steel material properties
    materials = [concrete, steel]
    thickness = 1

    m.stitch(0.001, materials, thickness)
    m.addRebar(np.array([1,3]), np.array([12,6]), 3, concrete, steel)
    m.addRebar(np.array([1,10]), np.array([6,6]), 3, concrete, steel)
    m.plotMesh()
    # return m
