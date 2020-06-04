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

The ``pydemic`` package comprises a set of tools for modeling epidemic trajectories. The code in this repository provides implementations of Markovian and non-Markovian 
`compartmental models <https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology>`_ 
including (1) the standard SEIR model and (2) our SEIR++ model, which is an extension to the Kermack and McKendrick 
`age-of-infection model <https://royalsocietypublishing.org/doi/10.1098/rspa.1927.0118>`_
that tracks the flow of individuals through the healthcare system. ``pydemic`` also provides base reaction and non-Markovian simulation classes that can be extended to implement your own epidemic model. The dynamics of new infections and movement through the healthcare system is set several model parameters that define things like time delays (e.g., the serial interval—time delay between becoming infected and infecting someone) and the severity model (i.e., the likelihood of progressing from one stage of the disease to another). 

Simulations are designed to output time series data for different population states, e.g., the number of infected persons or the number of individuals in the hospital. By comparing the model output data to real world data, the model parameters can be calibrated to real world epidemic dynamics. This parameter inference task is supported by the ``pydemic`` package with interfaces to `emcee <https://emcee.readthedocs.io/en/stable/>`_
as well as SciPy's global optimization routines. 

``pydemic`` provides simple example COVID-19 case data parsers for
the `COVID Tracking Project <https://covidtracking.com/>`_ and
the `New York Times <https://github.com/nytimes/covid-19-data>`_
data sources.
Pull requests that contribute robust, new parsers are welcome!

We provide a detailed description of our SEIR++ model, along with the results of calibrating it to data from COVID-19 in Illinois, in a
`recent preprint <https://arxiv.org/abs/2006.02036>`_on the arxiv.

``pydemic`` is in beta.
While effort will be made to preserve backwards compatibility with staged
deprecation, we cannot guarantee that interfaces will not change or that features will be preserved.
However, we will attempt to ensure that backwards-incompatible changes are demarcated by incrementing the major version.

``pydemic`` is `fully documented <https://pydemic.readthedocs.io/en/latest/>`_
and is licensed under the liberal `MIT license
<http://en.wikipedia.org/wiki/MIT_License>`_.

See the 
`docs <https://pydemic.readthedocs.io/en/latest/citing.html>`_
for citation info.

Getting started
#####

``pydemic`` is available on PyPI and can be installed via

    pip install pydemic

If you want a feel for how to run the SEIR++ model and simulate epidemic trajectories, see the 
`SEIR++ simulations <https://github.com/uiuc-covid19-modeling/pydemic/blob/master/examples/SEIR%2B%2B.ipynb>`_
notebook. This notebook walks you through setting up a simulation object, running the simulation, and visualizing the results.

The `calibrating models to data <https://github.com/uiuc-covid19-modeling/pydemic/blob/master/examples/calibration.ipynb>`_
notebook provides a quick example of model calibration that shows how to generate the posterior probability distribution for a set of model parameters using public Illinois data from the COVID Tracking Project.



