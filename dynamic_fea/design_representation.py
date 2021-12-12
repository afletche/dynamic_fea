# from dataclasses import dataclass
from typing import List

import numpy as np

from dynamic_fea.component import Component


class DesignRepresentation:
    def __init__(self, name=None, components=[]) -> None:
        self.name = name
        self.components = components    # list of Components


class FERepresentation(DesignRepresentation):
    def __init__(self, name=None, components=[], meshes=[]) -> None:
        super().__init__(name=name, components=components)
        self.meshes = meshes
    #     self.sub_representation = []

    
    # def assemble(self):
    #     for mesh, i in self.meshes:
    #         self.sub_representation.append((mesh, self.components[i]))

