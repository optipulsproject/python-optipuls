import meshio
import argparse

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='doublespot.msh')
parser.add_argument('-o', '--output', default='doublespot.xdmf')
parser.add_argument(
    '--dim', type=int, default=2, choices=[2, 3],
    help='dimension of the mesh'
    )
args = parser.parse_args()

mesh = meshio.read(args.input)

if args.dim == 2:
    meshio.write(
        args.output ,
        meshio.Mesh(
            points=mesh.points,
            cells={"triangle": mesh.get_cells_type("triangle")},
            )
        )

elif args.dim == 3:
    meshio.write(
        args.output,
        meshio.Mesh(
            points=mesh.points,
            cells={"tetra": mesh.get_cells_type("tetra")},
            )
        )

    meshio.write(
        "test" + "_boundaries.xdmf",
        meshio.Mesh(
            points=mesh.points,
            cells={"triangle": mesh.get_cells_type("triangle")},
            cell_data={"ids": [mesh.get_cell_data("gmsh:physical", "triangle")]},
            )
        )

    meshio.write(
        "test" + "_subdomains.xdmf",
        meshio.Mesh(
            points=dmeshpoints,
            cells={"tetra": dmeshget_cells_type("tetra")},
            cell_data={"ids": [dmeshget_cell_data("gmsh:physical", "tetra")]},
            )
        )
