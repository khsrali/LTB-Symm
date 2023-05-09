Introduction
============


Tight binding is a model used to describe the behavior of electrons in a solid, particularly in metals and semiconductors. It is based on the idea that the electrons in a solid are tightly bound to the atoms, but can also move around and interact with neighboring atoms. 

This model model is particularly useful to study the electronic properties of large sized materials with a simple crystal structure, that could not be studied by DFT due to its expensive computation time. 

Solving a tight binding model, always narrows down to diagonalizing the Hamiltonian. Resulting in a eigenvalue and eigenvectors. The former produces band structure of the solid, and the later is studied for topological interest.

The eigenvectors represent the probability amplitudes for the electrons to be found at each site in the crystal lattice.
By studying topology of eigenvectors, we care how the electronic states of a crystal are distributed in momentum space. 

One apprach to study that, is to use systematic tools of group theory, and to analyse the symmetries of the crystal lattice and their impact on the electronic structure. Namely spatial and time-reversal symmetries.

Most researchers in the field, prefer to have a framework that not only does the tight binding, but also leaves their hand open to implement and investigate more with symmetry properties of the eigenvectors.


**LTB-Symm** started as a handy tool for our own research, but now worth sharing with others!

