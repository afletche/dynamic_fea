from material import Material
import numpy as np


"""Class for defining an engineering structure."""
class Component:

    def __init__(self, name=None, components=[]) -> None:
        self.name = name
        self.components = components
        self.properties_list = []
        self.properties = {}
        self.design_representations = {}
    

    def add_lifting_surface_properties(self, cl=None, cd=None, cd0=None, cdi=None, AR=None, planform_area=None):
        self.properties_list.append('lifting_surface')
        if cl is not None:
            self.cl = cl
        if cd is not None:
            self.cd = cd
        if cd0 is not None:
            self.cd0 = cd0
        if cdi is not None:
            self.cdi = cdi
        if AR is not None:
            self.AR = AR
        if planform_area is not None:
            self.planform_area = planform_area


    '''
    Alternative idea: have a general add_properties method
    '''
    def add_properties(self, properties_dict={}):
        for key in properties_dict.keys():
            self.properties[key] = properties_dict[key]

    def add_design_representations(self, design_representations_dict={}):
        for key in design_representations_dict.keys():
            self.design_representations[key] = design_representations_dict[key]


# @dataclass
# class PlateComponent(Component):
#     material: Material
#     thickness: float
#     dimensions: np.ndarray = None



# @dataclass
# class CompositeLayup(Component):
#     materials: List     # list of material objects
#     thetas: np.ndarray
