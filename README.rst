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

The ``pydemic`` package comprises a set of tools for modeling population dynamics of epidemics and evaluating models against data.
``pydemic`` provides implementations of various types of
`compartmental models <https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology>`_,
including not only SIR/SEIR models and their extensions but also Kermack and McKendrick
`age-of-infection models <https://royalsocietypublishing.org/doi/10.1098/rspa.1927.0118>`_
which generalize the former class to model state transitions described by
arbitrary time-delay distributions.
More generally, ``pydemic`` provides frameworks for specifying reaction-based and non-Markovian simulations, enabling users to specify custom epidemic models.

To evalaute a model's applicability to an actual epidemic, its predictions
(e.g., for the rate of new cases or deaths) must be compared to real world data.
Parameter inference---the task of calibrating a model's input parameters by likelihood estimation---is supported by ``pydemic``'s interfaces to `emcee <https://emcee.readthedocs.io/en/stable/>`_
as well as SciPy's global optimization routines.

``pydemic`` provides simple parsers for COVID-19 case data sourced by
the `COVID Tracking Project <https://covidtracking.com/>`_ and
the `New York Times <https://github.com/nytimes/covid-19-data>`_.
Pull requests that contribute robust, new parsers are welcome!

As a complete example, the
`SEIR++ <https://pydemic.readthedocs.io/en/latest/ref_models.html#pydemic.models.SEIRPlusPlusSimulation>`_
model we implement treats the dynamics of infection and subsequent hospitalization.
The model is described by a large set of parameters that specify
the distribution of various time delays (e.g., for the spread of infection, symptom onset, hospitalization, etc.), the severity of infection, and the degree to which non-pharmaceutical interventions mitigate the spread of the disease.
We calibrate these parameters by comparing the model's predictions
to COVID-19--related hospitalizations and deaths in Illinois.
We describe the model and calibration procedure in detail in a `recent preprint <https://arxiv.org/abs/2006.02036>`_ on the arXiv.

``pydemic`` is in beta.
While effort will be made to preserve backwards compatibility with staged
deprecation, we cannot guarantee that interfaces will not change or that features will be preserved.
However, we will attempt to ensure that backwards-incompatible changes are demarcated by incrementing the major version.

``pydemic`` is `fully documented <https://pydemic.readthedocs.io/en/latest/>`_
and is licensed under the liberal `MIT license
<http://en.wikipedia.org/wiki/MIT_License>`_. See the docs for
`citation info <https://pydemic.readthedocs.io/en/latest/citing.html>`_.

Getting started
---------------

``pydemic`` is available on PyPI and can be installed with pip::

    pip install pydemic

Several examples exhibit how to use ``pydemic``:

* The `SEIR++ tutorial notebook <https://github.com/uiuc-covid19-modeling/pydemic/blob/master/examples/SEIR%2B%2B.ipynb>`_ demonstrates how to set up a simulation object, specify parameters, run the simulation, and visualize the results.
* An example of `callibration <https://github.com/uiuc-covid19-modeling/pydemic/blob/master/examples/calibration.ipynb>`_ illustrates how to set up likelihood estimation, run Markov chain Monte Carlo, and plot the posterior probability distribution of model parameters, using the SEIR++ model and public Illinois data from the COVID Tracking Project as an example.
