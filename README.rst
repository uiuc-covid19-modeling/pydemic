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

The ``pydemic`` package comprises a set of tools for modeling the population dynamics of an epidemic and calibrating models to data.
``pydemic`` provides implementations of various types of
`compartmental models <https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology>`_,
including not only SIR/SEIR models but also the more general Kermack and McKendrick
`age-of-infection models <https://royalsocietypublishing.org/doi/10.1098/rspa.1927.0118>`_,
which allow transitions between disease states to be described by arbitrary time-delay distributions.
More generally, ``pydemic`` provides a framework for users to specify custom epidemic models by extending the base classes for reaction-based and non-Markovian simulations.

To evaluate a model's applicability to an actual epidemic, its predictions
(e.g., for the rate of new cases or deaths) can be compared to real world data.
Parameter inference—the task of calibrating a model's input parameters via likelihood estimation—is supported by ``pydemic``'s interfaces to the Markov chain Monte Carlo package `emcee <https://emcee.readthedocs.io/en/stable/>`_
and SciPy's global optimization routines.

``pydemic`` provides simple parsers for COVID-19 case data sourced from
the `COVID Tracking Project <https://covidtracking.com/>`_ and
the `New York Times <https://github.com/nytimes/covid-19-data>`_.
Pull requests that contribute new, robust parsers are welcome!

As an example, we consider our
`SEIR++ <https://pydemic.readthedocs.io/en/latest/ref_models.html#pydemic.models.SEIRPlusPlusSimulation>`_
model, which treats the dynamics of infection and subsequent hospitalization.
The model includes a large set of parameters that specify
the distribution of various time delays (e.g., for the spread of infection, symptom onset, and hospitalization), the severity of infection (i.e., the likelihood of progressing from one stage of the disease to the next), and the time-dependent degree to which interventions mitigate the spread of the disease.
We calibrate these parameters by comparing the model's predictions
to COVID-19–related hospitalizations and deaths in Illinois.
We describe the model and calibration procedure in detail in a `recent preprint <https://arxiv.org/abs/2006.02036>`_ on the arXiv.

``pydemic`` is in beta.
While effort will be made to preserve backwards compatibility with staged
deprecation, we cannot guarantee that features will be preserved or that interfaces will not change.
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
* An example of `calibration <https://github.com/uiuc-covid19-modeling/pydemic/blob/master/examples/calibration.ipynb>`_ illustrates how to set up likelihood estimation, run Markov chain Monte Carlo, and plot the posterior probability distribution of model parameters, using the SEIR++ model and public Illinois data from the COVID Tracking Project as an example.
