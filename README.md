FEniCS model for the heat equation with phase transition
========================================================

This repository contains a numerical model for the laser beam welding of aluminium alloys.

## Usage

Ordinarily you want to run `python3 run.py`. For the list of optional arguments run `python3 run.py --help` 

In order to implement CI/CD the script `calculate.sh` is provided.

Commits tagged as `experiments/*` are supposed to reproduce a certain numerical experiment described in its commit message.
