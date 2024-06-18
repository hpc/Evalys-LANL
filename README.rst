===============================
Evalys-LANL - Overview
===============================

Copyright Notice
----------------
Extensions to the Evalys library Authored by Vivian Hafener are copyright © 2024 Triad National Security, LLC 
Release Approved Under O#4697 - LANL-contributed enhancements to BatSim toolstack.

The original Evalys library copyright (c) 2015, Olivier Richard and contributors.
See AUTHORS for more details.


.. image:: https://img.shields.io/pypi/v/evalys.svg
    :target: https://pypi.python.org/pypi/evalys


"Infrastructure Performance Evaluation Toolkit"

It is a data analytics library made to load, compute, and plot data from
job scheduling and resource management traces. It allows scientists and
engineers to extract useful data and visualize it interactively or in an
exported file.

* Free software: BSD license
* Documentation: https://evalys.readthedocs.org.


Extensions upon oar-team/evalys
---------
This repository provides a series of extensions to Evalys enabling visualization of a wide range of trends present on HPC clusters. Extensions include the addition of tools to highlight reservations and many new coloration methods to increase the visibility of a wide range of trends within the data. This was produced in conjunction with https://github.com/hpc/BatsimGantt-LANL, a tool to visualize outputs from our modified version of batsim (https://github.com/hpc/Batsim-LANL), and https://github.com/hpc/LiveGantt-LANL, a tool that uses Evalys to visualize current cluster state based on the output of Sacct on clusters running Slurm.


Features
--------

* Load and all `Batsim <https://github.com/oar-team/batsim>`_ outputs files

  + Compute and plot free slots
  + Simple Gantt visualisation
  + Compute utilisation / queue
  + Compute fragmentation
  + Plot energy and machine state

* Load SWF workload files from `Parallel Workloads Archive
  <http://www.cs.huji.ac.il/labs/parallel/workload/>`_

  + Compute standard scheduling metrics
  + Show job details
  + Extract periods with a given mean utilisation


Examples
--------

You can get a simple example directly by running ipython and discover the
evalys interface. For example::

  from evalys.jobset import JobSet
  import matplotlib.pyplot as plt

  js = JobSet.from_csv("evalys/examples/jobs.csv")
  js.plot(with_details=True)
  plt.show()

This also works for SWF files but the Gantt chart is not provided because
job placement information is not provided in this format.

You can find a lot of examples in the `./examples` directory.

Gallery
-------

.. image:: ./docs/_static/out_jobs_example.png
.. image:: ./docs/_static/jobset_plot.png
.. image:: ./docs/_static/gantt_comparison.svg
.. image:: ./docs/_static/gantt_off_mstates.svg

