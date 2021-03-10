import json

class Material:
    def __init__(self):
        pass

def from_file(filename):
    material = Material()

    with open(filename) as file:
        material_dict = json.load(file)

    for attr in [
            "label",
            "description",
            "solidus",
            "liquidus",
            "melting_enthalpy",
            "knots",
            "heat_capacity",
            "density",
            "kappa_rad",
            "kappa_ax",
            ]:
        setattr(material, attr, material_dict[attr])

    return material
