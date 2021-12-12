from typing import List

import numpy as np

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

