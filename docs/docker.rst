OptiPuls in Docker
==================

The ``optipuls`` python package is based on the free and open-source `FEniCS`_ computing platform for solving partial differential equations (PDEs). The FEniCS project ships with its `official docker images <https://fenics.readthedocs.io/projects/containers/en/latest/>`_.

.. _FEniCS: https://fenicsproject.org/

In view of this, the ``optipuls`` python package comes in a bundle with the official optipuls docker images available at `<https://hub.docker.com/r/optipulsproject/>`_.

Getting the OptiPuls docker image on a system with docker software installed is as simple as::

	$ docker pull optipulsproject/optipuls:latest
	latest: Pulling from optipulsproject/optipuls
	Digest: sha256:89015703048a0ad76d0a11880f763e20bbe8cb8903db977b785b702c432e22df
	Status: Downloaded newer image for optipulsproject/optipuls:latest
	docker.io/optipulsproject/optipuls:latest

References
----------

- `Docker Overview <https://docs.docker.com/get-started/overview/>`_
- `Getting Started with Docker <https://docs.docker.com/get-started/>`_
- `Install Docker Engine <https://docs.docker.com/engine/install/>`_
