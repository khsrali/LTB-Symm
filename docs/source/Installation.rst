Install
=======


Requirements
-----------------------------
.. LTB-Symm is written in Python. We suggest you to install or update to the latest available version, at least Python 3.6. is required.
.. (AS: I wouldn't suggest to install the latest, just say the minimum required, that is usually enough)
LTB-Symm is written in Python. Requirments: Python 3.8+ and NumPy 1.19.5

Operating system
++++++++++++++++

LTB-Symm has been developed and tested on both **Linux** and **MacOS**.
In both of these operating systems, installation process is similar. Although on **MacOS** there is an extra pre-step, i.e. make sure you have ``pip`` installed by typing the command below in your terminal:

.. code-block:: console

    % python -m ensurepip or python3 -m ensurepip

In principle it should be possible to install the code on **Windows** machines, but we have not tested it, yet. Again you can install ``pip`` by executing the following command in ``cmd``:

.. code-block:: console

    > curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    > python get-pip.py


After having ``pip`` installed on your machine, install the package with one of the two following ways. Choose depending on your needs and preferences.

Through pip
+++++++++++

You can simply install through:

.. code-block:: console

    $ pip install ltb-symm

.. If you have it already install, you may use option :code-block:`--upgrade` to have the latest version.

Please make sure all dependencies correctly installed. That includes;

.. code-block:: none

    numpy scipy matplotlib mpi4py tqdm spglib primme


Note you need a working MPI implementation for ``mpi4py`` to succesfully function.
(AS: to run in parallel or at all?)


Download the source code
++++++++++++++++++++++++

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
