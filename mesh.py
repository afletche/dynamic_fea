from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class Mesh:
    name: str
    nodes: np.ndarray   # TODO I don't think I want this here. At least optional beacuse Bsplines don't have nodes.

@dataclass
class StructuredMesh(Mesh):
    shape: np.ndarray   # shape = (nx,ny,3) or (nx,ny,nz,3) or (nx,ny,2), etc.

@dataclass
class UnstructuredMesh(Mesh):
    # nodes: np.ndarray   # shape = (num_nodes,3)
    element_nodes: np.ndarray = None  # shape = (num_elements, num_nodes_per_ele)
    elements: List = None
    node_connectivity: List = None
