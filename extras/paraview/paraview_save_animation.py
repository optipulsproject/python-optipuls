#!/bin/env pvbatch

'''Makes animation (as a set of .png files) from a ParaView .pvsm state file.'''

import argparse
from paraview.simple import *


parser = argparse.ArgumentParser()
parser.add_argument(
        '-s', '--state', default='state.pvsm',
        help='ParaView state file (*.pvsm *.py)')
parser.add_argument(
        '-o', '--output', default='/tmp/paraview/ani.png',
        help='output file name')
parser.add_argument(
        '-r', '--resolution', default='3840x2160',
        type=lambda res: list(map(int, res.split('x'))),
        help='rendering resolution, default 3840x2160')
args = parser.parse_args()


LoadState(args.state)
SaveAnimation(args.output, ImageResolution=args.resolution)
