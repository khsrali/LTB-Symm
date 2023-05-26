<img
src="https://github.com/khsrali/LTB-Symm/blob/develop/docs/source/_images/logo_V_0.1.png?raw=true"
width="1200" alt="image" />


# LTB-Symm

LTB-Symm does two things: **large scale tight-binding** (LTB) calculation of 2D materials, and checks
**topological symmetries** (Symm) of their wave functions.

## Who benefits

LTB-Symm is an ideal choice for researchers looking for a ready-to-use, easy-to-modify, and MPI-implemented TB code for large scale 2D structures. Up to 1 (0.1) Milions atoms for few (many) K-points, is managable.

Specially communities who deal with twisted bilayer/multilayer graphene.

All input needed are:  
1.  Coordinate of atoms/orbitals, e.g. lammpstrj, XYZ
2.  Functional form of Hamiltoninan

And possible outputs are:  
-   Bands structure
-   Density of States
-   Check topological symmetries of wave functions
-   Shape of the wavefunction

## Features

-   MPI implemented, able to run on HPC clusters.
-   Object Oriented, easy to modify for multi purpose.
-   Efficient (calculate only the subset of energy levels that are physically relevant).
-   Ideal for 2D materials, e.g. graphene.
-   Many routings are automated.
-   The first open-source code (to the best of our knowledge) which is able to investigate group symmetries in these systems in a systematic way.

## Install

    pip install ltb-symm

## Documentation

Examples are provided at [https://ltb-symm.readthedocs.io](https://ltb-symm.readthedocs.io) 
