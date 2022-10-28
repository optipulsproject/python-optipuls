An Optimal Control Problem for Single Spot Laser Pulse Welding
==============================================================

| **Repository:** https://github.com/optipulsproject/optimal-control-spot-welding
| **arXiv:** https://arxiv.org/abs/2109.10788v2


Abstract
--------

We consider an optimal control problem for a single-spot pulsed laser welding problem.
The distribution of thermal energy is described by a quasilinear heat equation.
Our emphasis is on materials which tend to suffer from hot cracking when welded, such as aluminum alloys.
A simple precursor for the occurrence of hot cracks is the velocity of the solidification front.
We therefore formulate an optimal control problem whose objective contains a term which penalizes excessive solidification velocities.
The control function to be optimized is the laser power over time, subject to pointwise lower and upper bounds.
We describe the finite element discretization of the problem and a projected gradient scheme for its solution.
Numerical experiments for material data representing the EN~AW~6082-T6 aluminum alloy exhibit interesting laser pulse patterns which perform significantly better than standard ramp-down patterns.


.. _paper-reproduce:

Reproducable numerical results
------------------------------

.. note::

	This paper is witten in a fully reproducible way, i.e. all the numerical artifacts used in the paper are being created every time the paper is being built. While the paper explains the mathematical model behind the core, its source code can be also used as a tutorial for running simulations and optimizations with ``optipuls``.

The numerical results presented in the paper can be easily reproduced using following the instructions. These results are based on the corresponding numerical model `optipuls <https://github.com/optipulsproject/optipuls>`_.


Why reproducing the result?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We believe that any numerical results presented in a scientific publication must be considered reliable only if the exaсt way they were obtained is clear and hence they can be verified by a reader. The most transparent way to go is to provide an explicit instruction on reproducing of the results, requiring only free software.

Despite it is often not the case in many scientific publications, we intend to encourage reproducibility culture in computational science by setting an example.


Reproducing (local host system)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A working FEniCS_ computing platform installation is required as well as the following additional python packages (including their dependencies):

.. _FEniCS: https://fenicsproject.org/

- `optipuls <https://github.com/optipulsproject/optipuls>`_
- `matplotlib <https://pypi.org/project/matplotlib/>`_
- `tabulate <https://pypi.org/project/tabulate/>`_

We suppose that `make <https://www.gnu.org/software/make/>`_ is already installed on your machine provided a UNIX-like system is used.

If you already have FEniCS installed locally, you can use python virtual environments to install the remaining dependencies without cluttering your system::

	python3 -m venv --system-site-packages ~/.local/optipuls
	source ~/.local/optipuls/bin/activate
	pip install git+https://github.com/optipulsproject/optipuls
	pip install matplotlib tabulate


Once the depencdencies are satisfied, reproducing of the results is as simple as running ``make`` in the root of the project::

	git clone https://github.com/optipulsproject/optimal-control-spot-welding
	cd optimal-control-spot-welding
	make -j$(nproc)

Make will run the computations, produce the plots, the tables, and the final ``manuscript-numapde-preprint.pdf`` file.


Reproducing (docker)
^^^^^^^^^^^^^^^^^^^^

Prebuilt `optipuilsproject <https://hub.docker.com/orgs/optipulsproject>`_ docker images can be used to reproduce the results provided docker is installed on your system.

Pull neccessary images::

	docker pull optipulsproject/optipuls:optimal-control-spot-welding
	docker pull optipulsproject/tabulate:latest
	docker pull optipulsproject/publications:latest

Make plots (entails making of the numerical artifacts)::

	docker run \
	  -v $(pwd):/home/fenics/shared \
	  optipulsproject/optipuls:optimal-control-spot-welding \
	  make plots.all -j$(nproc)

Make tables::

	docker run \
	  -u $UID \
	  -v $(pwd):/data \
	  optipulsproject/tabulate:latest \
	  make tables.all

Make paper::

	docker run \
	  -u $UID \
	  -v $(pwd):/data \
	  optipulsproject/publications:latest \
	  make preprint

