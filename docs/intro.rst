Introduction
============

| **Authors:** Dmytro Strelnikov, Roland Herzog
| **Funding:** Funding: Project IGF 20.826B (DVS I2.3005) in Forschungsvereinigung Schweißen und verwandte Verfahren e.V. of the `Deutschen Verbandes für Schweißen und verwandte Verfahren e.V. <https://www.die-verbindungs-spezialisten.de/>`_
| **Project page:** `Simulation Based Optimization of the Time-Dependent Pulse Power for Laser Beam Welding of Aluminum Alloys in Order to Avoid Hot Cracks <https://www.tu-chemnitz.de/mathematik/part_dgl/projects/optipuls/index.en.php>`_

This documentations is dedicated to the python package `optipuls` which implements a numerical model for simulation and mathematical optimization of the single spot pulsed laser beam welding of aluminium alloys. Its implementation strongly relies on the free and open-source `FEniCS`_ computing platform.

.. _FEniCS: https://fenicsproject.org/

This software package allows an expert user to create numerical simulations for the single-spot pulsed laser beam welding process, formulate in a flexible way and solve the corresponding optimal control problems, and finally generate optimized laser pulse shapes. These pulse shapes can be further programmed into a laser's power source in order to obtain fast, energy efficient, and crack-free welds.

References
----------

- https://github.com/optipulsproject/python-optipuls