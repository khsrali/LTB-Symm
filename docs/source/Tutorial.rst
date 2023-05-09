Tutorial
========

For any use of LTB-Symm, you need to properly import `ltbsymm`:

.. code:: ipython3

    import ltbsymm as ls


Input file
----------
First thing needed is a coordinate file which can be created with various softwares. For instance on can directly use a `LAMMPS` date file. 
This is particularly handy, because usually large structure require relaxation.

Data file should include information about lattice vectors, XYZ coordinates and optionally type of each atom. Indices are ignored.


.. code:: ipython3

    mytb.set_configuration('1.08_1AA.data', r_cut = 5.7, local_normal=True, nl_method='RS')


.. Notice in the case of non-symmorphic groups, Cmat is only implemented for C2 symmetries.



How to run
----------

.. this makes red      ``blah``
.. this looks like a function     :py:func:`blah`


 
