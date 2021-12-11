from dataclasses import dataclass

@dataclass
class Material:
    """Class for defining an engineering material."""
    name: str


@dataclass
class IsotropicMaterial(Material):
    """Class for defining an isotropic engineering material."""
    E: float = 69e9
    G: float = 25e9
    nu: float = 0.3
    cte: float = 24e-6
    density: float = 2.6989e3
    sigma_u: float = 310e6
    sigma_y: float = 276e6
    sigma_s: float = 207e6
    # SF_u: float = 1.5
    # SF_y: float = 1.2


@dataclass
class OrthotropicMaterial(Material):
    E11: float
    E22: float
    G12: float
    nu12: float
    nu21: float
    cte11: float
    cte22: float
    cte12: float
    # thickness: float  # Leave this to geometry
    density: float
    sigma11: float
    sigma22: float
    sigma12: float
    # SF: float = 1.5
