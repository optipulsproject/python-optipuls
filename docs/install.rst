Installing ``optipuls``
=======================

.. note::

   The ``optipuls`` package is not a click-to-run software. While this documentation covers the main points for a quick start, it remains an expert oriented system. Some knowledge of the linux command line, docker, and python programming language is required.


Requirements
------------

To run simulations and solve optimization problems, ``optipuls`` requires a working FEniCS installation. Python packages ``numpy``, ``scipy`` and ``matplotlib`` must be already installed as FEniCS' dependencies.

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

The ``optipuls`` python package is based on the free and open-source `FEniCS`_ computing platform for solving partial differential equations (PDEs). The FEniCS project ships with its `official docker images <https://fenics.readthedocs.io/projects/containers/en/latest/>`_.

.. _FEniCS: https://fenicsproject.org/

In view of this, the ``optipuls`` python package comes in a bundle with the official optipuls docker images available at `<https://hub.docker.com/r/optipulsproject/>`_.

Getting the OptiPuls docker image on a system with docker software installed is as simple as::

	$ docker pull optipulsproject/optipuls:latest
	latest: Pulling from optipulsproject/optipuls
	Digest: sha256:89015703048a0ad76d0a11880f763e20bbe8cb8903db977b785b702c432e22df
	Status: Downloaded newer image for optipulsproject/optipuls:latest
	docker.io/optipulsproject/optipuls:latest


Useful Links
^^^^^^^^^^^^

- `Docker Overview <https://docs.docker.com/get-started/overview/>`_
- `Getting Started with Docker <https://docs.docker.com/get-started/>`_
- `Install Docker Engine <https://docs.docker.com/engine/install/>`_


Extras
------

`ParaView`_ is recommended to inspect the simulation output and `Gmsh`_ is needed to view ``.MSH`` files.

.. _ParaView: https://www.paraview.org/
.. _Gmsh: http://gmsh.info/
