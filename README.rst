.. image:: https://github.com/khsrali/LTB-Symm/blob/main/docs/source/logo_V_0.1.png?raw=true
    :width:  1200

.. .. include:: https://github.com/khsrali/LTB-Symm/blob/main/docs/source/home.rst?raw=true


What is LTB-Symm?
-----------------
LTB-Symm is a publicly available code for large scale tight-binding (TB) calculation of 2D materials. Moreover it is also able to check topological symmetries of wave functions.


For who is LTB-Symm?
--------------------
LTB-Symm is an ideal choice for researchers looking for a ready-to-use, easy-to-modify, and MPI-implemented TB code for large scale structures. Up to 1 (0.1) Milions atoms for limited (vast) K-points, is easily managable.

All input needed are:
    #. Coordinate of atoms/orbitals, e.g. lammpstrj, XYZ  
    #. Functional form of Hamiltoninan


And possible outputs are:
    * Bands structure,
    * Density of States, 
    * Check topological symmetries of wave functions.
    * Shape of the wavefunction

Why LTB-Symm?
-------------
    * MPI implemented, able to run on HPC clusters.
    * Object Oriented, easy to modify for multi purpose.
    * Ideal for 2D materials, e.g. graphene, MoS2
    * Many routings are automated. For instance no worries about:
        * Indices of atoms, 
        * Detecting neghibors withing a cutoff,
        * Periodic boundary condition,
        * Orientation of orbitals like local normal vercors
    * Simply because there is no other open-source code that we know of. 
    That is why we wrote this code!
