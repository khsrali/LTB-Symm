from setuptools import setup, find_packages

setup(
    name='LTB-Symm',
    version='1.0.0',
    license='GNU under General Public License v3.0',
    author="Ali Khosravi, Andrea Silva",
    author_email='khsrali@gmail.com',
    description='Large Scale Tight Binding + Symmetries',
    long_description=open('README.md').read(),
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.7',
    url='https://github.com/khsrali/LTB-Symm',
    keywords=' tight-binding wave-function symmetries',
    classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Physics',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'mpi4py',
          'tqdm',
          'primme',
      ],

) 


