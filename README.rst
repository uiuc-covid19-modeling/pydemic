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

``pydemic`` implements standard Markovian as well as non-Markovian models of epidemics.
Parameter inference is enabled with interfaces to
`emcee <https://emcee.readthedocs.io/en/stable/>`_ and SciPy's global optimization
routines.

Note that ``pydemic`` is in beta.
While effort will be made to preserve backwards compatibility with staged
deprecation, we make no guarantee that features will be preserved nor that
interfaces will not change.
However, we will attempt to ensure that backwards-incompatible changes are
demarcated by incrementing the major version.

``pydemic`` is `fully documented <https://pydemic.readthedocs.io/en/latest/>`_
and is licensed under the liberal `MIT license
<http://en.wikipedia.org/wiki/MIT_License>`_.
