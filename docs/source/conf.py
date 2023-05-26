# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#sys.path.insert(0, os.path.abspath('../../src/ltbsymm/'))
sys.path.insert(0, os.path.abspath('../../src/'))

# -- Project information -----------------------------------------------------

project = 'LB-Symm'
copyright = 'GNU under General Public License v3.0'
author = 'Ali Khosravi, Andrea Silva'

# The full version, including alpha/beta/rc tags
release = 'v.1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.duration',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.doctest',]
              #'sphinx_reredirects']
#"nbsphinx" # to include Jupyter notebooks "raw"


#redirects = {
    #'custom_head': '_static/custom_head.html',
#}

#, 'sphinx.ext.autosummary', 'sphinx_autopackagesummary']
#autosummary_generate = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['misc.py']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_logo = "logo_sphinx.png"
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
#html_extra_path = ['_static']
