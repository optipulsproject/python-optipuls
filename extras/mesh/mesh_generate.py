'''Helper script for generating 2d (XZ) or 3d meshes specific to the OptiPuls project.'''

import pygmsh
import gmsh
from numpy import sqrt

import argparse


def restricted_float(x, lower=0., upper=1.):
    '''Validator for a float value withing [lower, upper] range.'''

    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not a float")

    if x < lower or x > upper:
        raise argparse.ArgumentTypeError(f"{x} is not in range [lower, upper]")
    return x

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '-Z', type=float, default=0.0005,
    help='height of the problem domain',
    )
parser.add_argument(
    '-R', type=float, default=0.0050,
    help='radius of the problem domain',
    )
parser.add_argument(
    '-r', type=float, default=0.0002,
    help='radius of the laser beam',
    )
parser.add_argument(
    '--overlap', type=restricted_float, default=0.75,
    help='overlap of the welding spots for double-spot problem, float in [0, 1]',
    )
parser.add_argument(
    '--dim', type=int, default=3, choices=[2, 3],
    help='dimension of the mesh',
    )
parser.add_argument(
    '--lcar_min', type=float, default=1.5e-5,
    help='minimal resolution of the mesh',
    )
parser.add_argument(
    '--lcar_max', type=float, default=2e-3,
    help='maxinal resolution of the mesh',
    )
parser.add_argument('-o', '--output', default='doublespot.msh')
parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true')
parser.add_argument(
    '--singlespot', dest='overlap', action='store_const', const=1.0,
    help='use this option for single-spot problems (sets overlap to 1)',
    )
args = parser.parse_args()

d = 2 * args.r * (1 - args.overlap)

def dist(point_0, point_1):
    '''Euclidean distance between two points.'''

    return sqrt(
        sum(
            (coord_0 - coord_1)**2
            for (coord_0, coord_1) in zip (point_0, point_1)
        )
    )

def level(point, center_0, center_1):
    '''Distance from a point to the closest center.'''

    if point[0] >= center_0[0]:
        return dist(point, center_0)
    elif point[0] >= center_1[0]:
        return dist((0, point[1], point[2]), center_0)
    else:
        return dist(point, center_1)

def mesh_level(point, center_0, center_1, lcar_min, lcar_max, level_min, level_max):
    level_ = level(point, center_0, center_1)
    if level_ < level_min:
        return lcar_min
    else:
        return lcar_min + (lcar_max - lcar_min) * (level_ / level_max)**2

def print_elements_info():
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        for e in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim, e)
            print(dim, tag, name, elem_types, elem_tags)

with pygmsh.geo.Geometry() as geom:
    if args.dim == 3:
        if d == 0:
            # single-spot geometry
            points = [
                geom.add_point([0,            0, 0]),  # 0: center_0
                geom.add_point([args.R,       0, 0]),  # 1
                geom.add_point([0,       args.R, 0]),  # 2
                geom.add_point([- args.R - d, 0, 0]),  # 3
            ]

            center_0, center_1 = points[0], points[0]

            arcs = [
                geom.add_circle_arc(points[1], points[0], points[2]),
                geom.add_circle_arc(points[2], points[0], points[3]),
            ]

            lines = [
                geom.add_line(points[0], points[1]),
                geom.add_line(points[3], points[0]),
            ]

            curve_loop = geom.add_curve_loop(
                [
                    lines[0],
                    arcs[0],
                    arcs[1],
                    lines[1],
                ]
            )

        elif d > 0:
            # double-spot geometry
            points = [
                geom.add_point([0,            0, 0]),  # 0: center_0
                geom.add_point([args.R,       0, 0]),  # 1
                geom.add_point([0,       args.R, 0]),  # 2
                geom.add_point([0 - d,   args.R, 0]),  # 3
                geom.add_point([- args.R - d, 0, 0]),  # 4
                geom.add_point([- d,     0, 0]),       # 5: center_1
            ]

            center_0, center_1 = points[0], points[5]

            arcs = [
                geom.add_circle_arc(points[1], points[0], points[2]),
                geom.add_circle_arc(points[3], points[5], points[4]),
            ]

            lines = [
                geom.add_line(points[0], points[1]),
                geom.add_line(points[2], points[3]),
                geom.add_line(points[4], points[5]),
                geom.add_line(points[5], points[0]),
            ]

            curve_loop = geom.add_curve_loop(
                [
                    lines[0],
                    arcs[0],
                    lines[1],
                    arcs[1],
                    lines[2],
                    lines[3],
                ]
            )

        plane_surface = geom.add_plane_surface(curve_loop)
        extrusion = geom.extrude(plane_surface, [0., 0., -args.Z])

        geom.synchronize()
        geom.add_physical(plane_surface, label="top")
        geom.add_physical(extrusion[0], label="bottom")
        geom.add_physical(extrusion[1], label="domain")
        geom.add_physical(extrusion[2], label="side")

    elif args.dim == 2:
        if d == 0:
            # single-spot geometry
            # half section due to radial symmetry
            points = [
                geom.add_point([0,      0, 0]),        # 0: center
                geom.add_point([args.R, 0, 0]),        # 1: center
                geom.add_point([args.R, 0, -args.Z]),  # 2
                geom.add_point([0,      0, -args.Z]),  # 3
            ]

            center_0, center_1 = points[0], points[0]

            lines = [
                geom.add_line(points[0], points[1]),
                geom.add_line(points[1], points[2]),
                geom.add_line(points[2], points[3]),
                geom.add_line(points[3], points[0]),
            ]

            curve_loop = geom.add_curve_loop(
                [
                    lines[0],
                    lines[1],
                    lines[2],
                    lines[3],
                ]
            )

            plane_surface = geom.add_plane_surface(curve_loop)

            geom.synchronize()
            geom.add_physical(plane_surface, "RZ")

        if d > 0:
            # double-spot geometry
            # makes no sense since radial symmetry breaks
            raise NotImplementedError

    geom.set_mesh_size_callback(
        lambda dim, tag, x, y, z, _:
            mesh_level(
                point=(.5*x, .5*y, z), center_0=center_0.x, center_1=center_1.x,
                lcar_min=args.lcar_min, lcar_max=args.lcar_max,
                level_min=0.6*args.r, level_max=0.5*args.R,
            )
    )
    mesh = geom.generate_mesh(args.dim)
    pygmsh.write(args.output)

    if args.verbose:
        print_elements_info()
