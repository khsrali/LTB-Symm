Usage
=====


Download
--------
LTB-Symm is publicly available at GitHub ``GNU under General Public License v3.0`` :
`https://github.com/khsrali/LTB-Symm <https://github.com/khsrali/LTB-Symm>`_
. For any question and problem, please don't hesitate to `contact <khsrali@gmail.com>`_ us.

If your use of LTB-Symm leads to an academic publication, please acknowledge that fact by citing the our paper:
*Bending Stiffness Collapse, Buckling, Topological Bands of Freestanding Twisted Bilayer Graphene*, J. Wang, A. Khosravi, A. Silva, et. al 2023


Installation and Requirements
-----------------------------
LTB-Symm iw written in Python. We suggest you to install or update to the latest available version, at least Python 3.6. is required.

Operating system
++++++++++++++++

LTB-Symm has been developed and tested on both **Linux** and **MacOS**. 
In both of these operating systems, installation process is basically be similar. Although in **MacOS** there is an extra pre-step, that is only to make sure you have ``pip`` installed. You can make sure of that by typing command below in your terminal:

.. code-block:: console

    % python -m ensurepip or python3 -m ensurepip
    
In **Windows** in principle it should be possible to install the code, but we have not tested, yet. Again you can install ``pip`` by executing the following command in ``cmd``:

.. code-block:: console

    > curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    > python get-pip.py


After having ``pip`` installed on your machine, install the package with one of the two following ways. Choose depending on your needs and preferences.

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

Simply write an *input* file and run using you python interpreter:


For example:


.. code-block:: console

   $ python input.py
 
