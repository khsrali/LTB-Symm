Introduction
============


Tight binding is a model used to describe the behavior of electrons in a solid, particularly in metals (AS:?? plane wave would be more appropriate no?) and semiconductors. The core physical idea is that the electrons in a solid are strongly bound to the atoms but can ajump between atoms with some probability. In other words, the electrons are localised on the atomic lattice site and hop between sites.

This model model is particularly useful to study the electronic properties of large sized materials with a simple crystal structure, where more sophisticated Denisty Functional Theory (DFT) methods become too expensive in terms of computation time.

Solving a tight binding model, always narrows down to diagonalizing the Hamiltonian. Resulting in a eigenvalue and eigenvectors. The former produces band structure of the solid, and the latrer econdes topological properties. Explict examples of such Hamiltonians are presented in the example section.

The eigenvectors represent the probability amplitudes for the electrons to be found at each site in the crystal lattice.
By studying topology of eigenvectors, we care how the electronic states of a crystal are distributed in momentum space (AS: unclear sentence. Band structure is energy level in momentum space, topology deals with the symmetry of eigenvector in real space. By using together band structure and topology you can understand how symmetry evolve across momentum space.).

The concepts and tools of group theory, allows us to systematically analyse the symmetries of the crystal lattice and their impact on the electronic structure. Namely spatial and time-reversal symmetries (AS: this seems a bit random here, especially time-reversal).

Most researchers in the field of twistronics, prefer to have a framework that not only diagonalises tight binding Hamiltonians, but also allows them to implement and investigate more with symmetry properties of the eigenvectors (AS: unclear).


**LTB-Symm** started as a handy tool for our own research, but grew into a code general and flexible enought to be worth sharing with others!
