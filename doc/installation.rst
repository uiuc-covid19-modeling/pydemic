.. highlight:: sh

.. _installation:

Installation
============

1. Run ::

    python setup.py develop

   which should `pip` install the other dependencies if needed.

3. To build the full documentation locally, install Sphinx and the required
   theme and build::

        conda install sphinx sphinx_rtd_theme
        python setup.py build_sphinx
