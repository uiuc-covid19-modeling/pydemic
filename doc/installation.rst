.. highlight:: sh

.. _installation:

Installation
============

1. Make sure to clone the repository with::

    git clone --recurse-submodules ...

1. Run ``sh setup.sh``, which installs pydemic via::

        python setup.py develop

   which should `pip` install the other dependencies if needed.

   and then fetches and parses data via::

        python data/parse_all.py --fetch --output-population assets --output-cases assets`

3. To build the full documentation locally, install Sphinx and the required
   theme and build::

        conda install sphinx sphinx_rtd_theme
        python setup.py build_sphinx
