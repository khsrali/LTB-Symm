logo test:

.. image:: https://github.com/khsrali/LTB-Symm/blob/main/docs/source/logo_V_0.1.png?raw=true
    :width:  1200



LTB-Symm is a publicly available code for large scale tight-binding (TB) calculation of 2D materials. Moreover it is also able to check topological symmetries of wave functions.

.. To this moment, there is no publicly available code for large scale tight-binding (TB) calculation of 2D materials, e.g. twisted bilayer graphene.


LTB-Symm is an ideal choice for researchers looking for a ready-to-use, easy-to-modify, and MPI-implemented TB code for large size structures. Up to 1 (0.1) Milions atoms for limited (vast) K-points , is easily managable.
All input needed is (a) coordinate of atoms/orbitals, e.g. lammpstrj, XYZ  (b) functional for of Hamiltoninan

Some mid level calculations are automated. For instance no worries about:
Indices of atoms, 
Detecting neghibors withing a cutoff,
Periodic boundary condition,
Orientation of orbitals like local normal vercors

Possible outputs are: 
Bands structure,
Density of States, 
Check topological symmetries of wave functions.

Advantage of LTB-Symm:
.. ----------------------
MPI implemented, able to run on HPC clusters.
Object Oriented, easy to modify for multi purpose.
Automated many routings, no need to bother.
Ideal for 2D materials, e.g. graphene, MoS2
