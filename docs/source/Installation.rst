Install
=======


Requirements
------------

LTB-Symm is written in Python. To function properly ``Python 3.8+`` is required.



Operating system
-----------------

LTB-Symm has been developed and tested on both **Linux** and **MacOS**.
In both of these operating systems, installation process is similar. Although on **MacOS** there is an extra pre-step, i.e. make sure you have ``pip`` installed by typing the command below in your terminal:

.. code-block:: console

    % python -m ensurepip

In principle it should be possible to install the code on **Windows** machines, but we have not tested it, yet. Again you can install ``pip`` by executing the following command in ``cmd``:

.. code-block:: console

    > curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    > python get-pip.py


.. After having ``pip`` installed on your machine, install the package with one of the two following ways. Choose depending on your needs and preferences.


Install through pip
-------------------

You can simply install with:

.. code-block:: console

    $ pip install ltb-symm

If you have it already install, you may use option :code:`--upgrade` to have the latest version.

If installation failed, please make sure all dependencies correctly installed. That includes;

.. code-block:: none

    numpy scipy matplotlib mpi4py tqdm primme

Required versions:

    * NumPy 1.19.5+
    * Scipy 1.10.0+
    * Primme 3.2

Note you need a working MPI implementation for ``mpi4py`` to succesfully function.

.. (AS: to run in parallel or at all?) Ali: at the moment it must be installed.


Source code
-----------

LTB-Symm is publicly available at GitHub under ``GNU General Public License v3.0`` :
`https://github.com/khsrali/LTB-Symm <https://github.com/khsrali/LTB-Symm>`_

For doubt and question, feel free to directly contact us. Email: khsrali@gmail.com




How to run
----------

.. this makes red      ``blah``
.. this looks like a function     :py:func:`blah`

Simply write an *input* file (see examples) and run using you python interpreter:


For example:


.. code-block:: console

   $ python input.py

Or alternatively if you want to use ``MPI`` feature, for instance on a HPC with 32 core each node:

.. code-block:: console

   $ mpirun --map-by ppr:32:node python input.py
