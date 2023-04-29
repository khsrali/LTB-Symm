Usage
=====

.. _installation:

Download
--------
LTB-Symm is publicly available at GitHub ``GNU under General Public License v3.0`` :

`https://github.com/khsrali/LTB-Symm <https://github.com/khsrali/LTB-Symm>`_

If your use of LTB-Symm leads to an academic publication, please acknowledge that fact by citing the our paper:

*Bending Stiffness Collapse, Buckling, Topological Bands of Freestanding Twisted Bilayer Graphene*, J. Wang, A. Khosravi, A. Silva, et. al 2023


Installation and Requirements
-----------------------------

There are two ways to install and use the package, depending on your needs.

Through pip
+++++++++++

Use this option if you do not need to highly modify source code.
In this case you simply run:

.. code-block:: console

    $ pip install ltb-symm


Through source code
+++++++++++++++++++

This option is suitable if you would like to customize the source code for your needs. For this purpose simply download and run like an usual python script.

However make sure you have all dependencies correctly installed. That includes;
The usual one that probebly you already have,

.. code-block:: console

    $ pip install numpy scipy matplotlib mpi4py tqdm spglib primme
    
and some others that necessarily to function properly:

.. code-block:: console

    $ pip install mpi4py tqdm spglib primme

Note you need a working MPI implementation for ``mpi4py`` to succesfully function.



How to run
----------

.. this makes red      ``blah``
.. this looks like a function     :py:func:`blah`

Simply need to write a ``input`` file and run using you python interpreter:


For example:


.. code-block:: console

   $ python input.py
 
