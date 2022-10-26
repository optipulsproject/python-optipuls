Installing ``optipuls``
=======================

.. note::

   The ``optipuls`` package is not a click-to-run software. While this documentation covers the main points for a quick start, it remains an expert oriented system. Some knowledge of the linux command line, docker, and python programming language is required.


Requirements
------------

To run simulations and solve optimization problems, ``optipuls`` requires a working FEniCS installation. Python packages ``numpy``, ``scipy`` and ``matplotlib`` must be already installed as FEniCS' dependencies.

`ParaView`_ is recommended to inspect the simulation output.

.. _ParaView: https://www.paraview.org/

Notice, that installing FEniCS with its dependencies might be difficult on systems other than Debian or Ubuntu, which are the ones [officially supported][fenics/installation/ubuntu] by FEniCS developers.
Therefore, it is recommended to use ``optipuls`` in a docker container using [optipulsproject/optipuls] docker image which is built on top of [fenicsproject/stable] docker image. Please refer to the section below.


Installing ``optipuls`` on the host system
------------------------------------------

Provided FEniCS is correctly deployed on the host system, ``optipuls`` can be simply installed via ``pip``.

Creating a Python virtual environment and switching to it (optional)::

	python3 -m venv optipulsenv
	source optipulsenv/bin/activate

Installing ``optipuls``::

	python3 -m pip install git+https://github.com/optipulsproject/optipuls


Running ``optipuls`` in a docker container
------------------------------------------

The officially recommended way to run numerical simulations and optimizations with ``optipuls`` package is to use docker containers. Please refer to :doc:`docker` page.
