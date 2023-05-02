#from setuptools import setup, find_packages
from distutils.core import setup


setup(
    name='LTB-Symm',
    version='0.1',
    license='GNU under General Public License v3.0',
    author="Ali Khosravi, Andrea Silva",
    author_email='khsrali@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/khsrali/LTB-Symm',
    keywords=' tight-binding and wave function symmetries',
    install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'mpi4py',
          'tqdm',
          'spglib',
          'primme',
      ],

) 
