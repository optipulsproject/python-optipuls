FEniCS model for the heat equation with phase transition
========================================================

[Simulation Based Optimization of the Time-Dependent Pulse Power for Laser Beam Welding of Aluminum Alloys in Order to Avoid Hot Cracks](https://www.tu-chemnitz.de/mathematik/part_dgl/projects/optipuls/index.en.php)

This repository contains a FEM numerical model for the laser beam welding of aluminium alloys.

Author: Dmytro Strelnikov <dmytro.strelnikov@math.tu-chemnitz.de>

![OptiPuls-preview](https://strelnikov.xyz/projects/OptiPuls/OptiPuls-preview.mp4)

## Requirements

A working [FEniCS](https://fenics.readthedocs.io/en/latest/installation.html) setup is required to run the code as well as `numpy` and `matplotlib` Python packages. ParaView is recommended to visualize the simulation output.


## Running

Ordinarily you may want to run
```
python3 run.py [--scratch /path/to/scratch_dir]
```

where `scratch_dir` is the desired directory for the simulation artifacts such as plots, ParaView files, dumped NumPy arrays, log files, etc. This variable is utilized by the script `calculate.sh`.

For the up-to-date list of command line arguments run `python3 run.py --help`.

In order to implement CI/CD the script `calculate.sh` is provided.


## Numerical experiments

Some commits are tagged as `experiments/*` and typically have neither children nor branches pointing at them. These commits are supposed to represent fully reproducible numerical experiments/simulations. The experiment setting is normally explained in the commit message.

The following command is used to tag a certain commit as an experiment:
```
git tag experiments/$(git rev-parse --short HEAD)
```

Use the following command to list all of such commits:
```
git show -s --tags=experiments --decorate --abbrev-commit --pretty=medium
```

Notice that some tools (like Sublime Merge) work incorrect with "hanging" commits under no branch even so they are still accessible by tags.


## Development

The development in this repo is carried using of a free variation of [git-flow](https://nvie.com/posts/a-successful-git-branching-model/) branching model considering the difference between scientific and enterprise software.

In particular, it means:

1. The `master` branch is used only for releases (e.g. freezing the code to refer to it in a publication, internal documentation, etc) and is not used to carry the main development.
2. The essential development history is reflected by the branch `develop` and various feature branches.
3. The bleeding-edge state of the code is usually reflected by the branch `experimental`.

Consider using of the following command to overview the development state:
```
git log --all --decorate --oneline --graph
```
