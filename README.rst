pydemic: a python driver for epidemic modeling and inference
============================================================

.. image:: https://github.com/uiuc-covid19-modeling/pydemic/workflows/CI/badge.svg?branch=master
    :alt: Github Build Status
    :target: https://github.com/uiuc-covid19-modeling/pydemic/actions?query=branch%3Amaster+workflow%3ACI
.. image:: https://readthedocs.org/projects/pydemic/badge/?version=latest
    :target: https://pydemic.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://badge.fury.io/py/pydemic.svg
    :target: https://badge.fury.io/py/pydemic

``pydemic`` provides implementations of standard Markovian as well as non-Markovian models of epidemics.
Parameter inference is enabled with interfaces to
`emcee <https://emcee.readthedocs.io/en/stable/>`_ and SciPy's global optimization
routines.
In addition, ``pydemic`` provides simple data parsers for COVID-19 case data
from the `COVID Tracking Project <https://covidtracking.com/>`_ and
the `New York Times <https://github.com/nytimes/covid-19-data>`_.
Pull requests that contribute new parsers are welcome!

To get started, see the example notebooks for
`running simulations <https://github.com/uiuc-covid19-modeling/pydemic/blob/master/examples/SEIR%2B%2B.ipynb>`_
and for `calibrating models to data <https://github.com/uiuc-covid19-modeling/pydemic/blob/master/examples/calibration.ipynb>`_.
We describe the "SEIR++" model in depth in a
`recent preprint <https://arxiv.org/abs/2006.02036>`_.

``pydemic`` is available on PyPI and can be installed via::

    pip install pydemic

``pydemic`` is in beta.
While effort will be made to preserve backwards compatibility with staged
deprecation, we do not make guarantees that features will be preserved or that interfaces will not change.
However, we will attempt to ensure that backwards-incompatible changes are demarcated by incrementing the major version.

``pydemic`` is `fully documented <https://pydemic.readthedocs.io/en/latest/>`_
and is licensed under the liberal `MIT license
<http://en.wikipedia.org/wiki/MIT_License>`_.
