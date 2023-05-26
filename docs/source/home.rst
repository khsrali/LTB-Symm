LTB-Symm
========

LTB-Symm does two things: **large scale tight-binding** (LTB) calculation of 2D materials, and checks **topological symmetries** (Symm) of their wave functions (the TB eigenvectors).



Who benefits
-------------
LTB-Symm is an ideal choice for researchers looking for a ready-to-use, easy-to-modify, and MPI-implemented TB code for large scale 2D structures. Up to 1 (0.1) Milions atoms for few (many) K-points, is managable.

At the current state of development, this code is particualrly suited for the community studying twisted bilayer/multilayer graphene.

All input needed are:
    #. Coordinate of atoms/orbitals, e.g. lammpstrj, XYZ
    #. Functional form of Hamiltoninan


And possible outputs are:
    * Bands structure,
    * Density of States,
    * Check topological symmetries of wave functions.
    * Shape of the wavefunction



Features
-------------
    * MPI implemented, able to run on HPC clusters.
    * Object Oriented, easy to modify for multi purpose.
    * Efficient (calculate only the subset of energy levels that are physically relevant).
    * Ideal for 2D materials, e.g. graphene.
    * Many routines are automated.
    * The first open-source code (to the best of our knowledge) which is able to investigate group symmetries in these systems in a systematic way.

.. , thanks to pre-developed implementtaions of ``LANCZOS`` algorithm.
.. For instance no worries about:
        * Orientation of orbitals like local normal vercors
        * Indexing neghiboring atoms,
        * Detecting neghibors withing a cutoff,
        * Periodic boundary condition,
