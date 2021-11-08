OptiPuls: a numerical model for single spot pulsed laser beam welding
=====================================================================

This repository contains a Python package `optipuls` which implements a numerical model for simulation and mathematical optimization of the single spot pulsed laser beam welding of aluminium alloys. Its implementation strongly relies on the [FEniCS][fenics] computing platform.

---

**Author:** Dmytro Strelnikov <dmytro.strelnikov@math.tu-chemnitz.de>  
**Funding:** Funding: Project IGF 20.826B (DVS I2.3005) in Forschungsvereinigung Schweißen und verwandte Verfahren e.V. of the [Deutschen Verbandes für Schweißen und verwandte Verfahren e.V.](https://www.die-verbindungs-spezialisten.de/)  
**Project page:** [Simulation Based Optimization of the Time-Dependent Pulse Power for Laser Beam Welding of Aluminum Alloys in Order to Avoid Hot Cracks](https://www.tu-chemnitz.de/mathematik/part_dgl/projects/optipuls/index.en.php)

---

![OptiPuls-preview](assets/3d_resized.png)


## Requirements

To run simulations and solve optimization problems, `optipuls` requires a working [FEniCS][fenics/installation] installation. Python packages `numpy`, `scipy` and `matplotlib` must be already installed as FEniCS' dependencies.

[ParaView](https://www.paraview.org/) is recommended to inspect the simulation output.

Notice, that installing FEniCS with its dependencies might be difficult on systems other than Debian or Ubuntu, which are the [officially supported][fenics/installation/ubuntu] by FEniCS developers.
Therefore, it is recommended to use `optipuls` in a docker container using [optipulsproject/optipuls] docker image which is built on top of [fenicsproject/stable] docker image. Please refer to the section below.


## Installing `optipuls` on the host system

Provided FEniCS is correctly deployed on the host system, `optipuls` can be simply installed via `pip`.

Creating a Python virtual environment and switching to it (optional):
```
python3 -m venv optipulsenv
source optipulsenv/bin/activate
```

Installing `optipuls`:
```
python3 -m pip install git+https://github.com/optipulsproject/optipuls
```


### Running `optipuls` in a docker container

Assuming that `/scratch/optipuls` is the desired location for the numerical artifacts and dijitso cache on the host system, the following command will mount the scratch directory and the current working directory inside a docker container and execute `run.py`:
```
$ docker run \
  -v $(pwd):/home/fenics/shared \
  -v /scratch/optipuls/cache/dijitso:/home/fenics/.cache/dijitso \
  -v /scratch/optipuls:/scratch \
  optipulsproject/optipuls:latest python3 run.py --scratch /scratch"
```


## Related papers

- [An Optimal Control Problem for Single-Spot Pulsed Laser Welding][paper-onespot]


## License

The sorce code is licensed under the terms of [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html).



[fenics]: https://fenicsproject.org/
[fenics/installation]: https://fenics.readthedocs.io/en/latest/installation.html
[fenics/installation/ubuntu]: https://fenics.readthedocs.io/en/latest/installation.html#debian-ubuntu-packages

[optipulsproject/optipuls]: https://hub.docker.com/r/optipulsproject/optipuls
[fenicsproject/stable]: https://quay.io/fenicsproject/stable

[paper-onespot]: https://github.com/optipulsproject/paper-onespot