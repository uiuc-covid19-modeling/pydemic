__copyright__ = """
Copyright (C) 2020 George N Wong
Copyright (C) 2020 Zachary J Weiner
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import pandas as pd
import emcee

import h5py
string_dt = h5py.string_dtype(encoding="ascii")

__doc__ = """
Backends with HDF5
------------------

.. currentmodule:: pydemic.hdf
.. autoclass:: HDFBackend
.. currentmodule:: pydemic
"""


class BackendMixIn:
    def set_fixed_values(self, fixed_values):
        with self.open("a") as f:
            f.create_group("fixed_values")
            for key, value in fixed_values.items():
                f["fixed_values"][key] = value

    def set_sample_parameters(self, sample_parameters):
        with self.open("a") as f:
            def nanify(x):
                return x if x is not None else np.nan

            f.create_group("fit_parameters")
            f["fit_parameters/names"] = np.array(
                [par.name for par in sample_parameters], dtype=string_dt
            )
            f["fit_parameters/bounds"] = [par.bounds for par in sample_parameters]
            f["fit_parameters/mean"] = [nanify(par.mean)
                                        for par in sample_parameters]
            f["fit_parameters/guess"] = [nanify(par.guess)
                                         for par in sample_parameters]
            f["fit_parameters/uncertainty"] = [nanify(par.uncertainty)
                                               for par in sample_parameters]
            f["fit_parameters/sigma"] = [nanify(par.sigma)
                                         for par in sample_parameters]

    def set_simulator(self, simulator):
        with self.open("a") as f:
            if simulator is not None:
                if isinstance(simulator, str):
                    name = simulator
                else:
                    name = simulator.__name__
                f.attrs["simulator"] = name

    def set_data(self, data):
        data.to_hdf(self.filename, "data")

    @property
    def fixed_values(self):
        with self.open() as f:
            return {key: val[()] for key, val in f["fixed_values"].items()}

    @property
    def sample_parameters(self):
        from pydemic.sampling import SampleParameter
        with self.open() as f:
            def denanify(x):
                return x if np.isfinite(x) else None

            names = [name.decode("utf-8") for name in f["fit_parameters/names"][:]]
            pars = []
            for i, name in enumerate(names):
                args = [f["fit_parameters/bounds"][i]]
                args += [denanify(f["fit_parameters"][key][i])
                         for key in ("mean", "uncertainty", "sigma", "guess")
                         if key in f["fit_parameters"].keys()]
                pars.append(SampleParameter(name, *args))
            return pars

    @property
    def data(self):
        with self.open() as f:
            if "data" in f.keys():
                return pd.read_hdf(f.filename, key="data")
            elif "df_data" in f.keys():
                return pd.read_hdf(f.filename, key="df_data")
            else:
                return None

    @property
    def simulator(self):
        """
        The simulation class whose ``get_model_data`` method is used for sampling.
        If the class is defined in :mod:`pydemic.models`, that class will be
        returned; otherwise the name of the class will be returned.
        """

        with self.open() as f:
            if "simulator" in f.attrs:
                name = f.attrs["simulator"]
                try:
                    import pydemic.models as models
                    return getattr(models, name)
                except AttributeError:
                    return name
            else:
                return None


class HDFBackend(emcee.backends.HDFBackend, BackendMixIn):
    """
    A subclass of :class:`emcee.backends.HDFBackend` which stores additional
    information used by :class:`~pydemic.LikelihoodEstimator` to automate
    resuming sampling.

    .. note::

        This class requires :mod:`h5py`.

    :arg filename: The name of the HDF5 file to create.

    The following optional parameters (corresponding to those passed to
    :class:`pydemic.LikelihoodEstimator`) will be stored in the file if passed.

    :arg sample_parameters:

    :arg fixed_values:

    :arg data:

    :arg simulator:

    Any remaining keyword arguments are passed to
    :class:`emcee.backends.HDFBackend`.

    The following attributes will be available if they were passed
    to :meth:`~pydemic.hdf.HDFBackend` upon creation of
    the file, and may be used to resume sampling:

    .. autoattribute:: fixed_values
    .. autoattribute:: sample_parameters
    .. autoattribute:: data
    .. autoattribute:: simulator
    """

    def __init__(self, filename, sample_parameters=None,
                 fixed_values=None, data=None, simulator=None, **kwargs):

        super().__init__(filename, **kwargs)

        if fixed_values is not None:
            self.set_fixed_values(fixed_values)
        if sample_parameters is not None:
            self.set_sample_parameters(sample_parameters)
        if simulator is not None:
            self.set_simulator(simulator)
        if data is not None:
            self.set_data(data)


class HDFOptimizationBackend(BackendMixIn):
    """
    A backend similar to :class:`emcee.backends.HDFBackend` for direct optimization
    routines.

    .. note::

        This class requires :mod:`h5py`.

    :arg filename: The name of the HDF5 file to create.

    The following optional parameters (corresponding to those passed to
    :class:`pydemic.LikelihoodEstimator`) will be stored in the file if passed.

    :arg sample_parameters:

    :arg fixed_values:

    :arg data:

    :arg simulator:

    Any remaining keyword arguments are passed to
    :class:`emcee.backends.HDFBackend`.

    The following attributes will be available if they were passed
    to :meth:`~pydemic.hdf.HDFBackend` upon creation of
    the file, and may be used to resume sampling:

    .. autoattribute:: fixed_values
    .. autoattribute:: sample_parameters
    .. autoattribute:: data
    .. autoattribute:: simulator
    """

    def __init__(self, filename, name="_optimizer", read_only=False, dtype=None,
                 sample_parameters=None, fixed_values=None, data=None,
                 simulator=None, **kwargs):
        self.filename = filename
        self.name = name
        self.read_only = read_only
        if dtype is None:
            self.dtype_set = False
            self.dtype = np.float64
        else:
            self.dtype_set = True
            self.dtype = dtype

        if fixed_values is not None:
            self.set_fixed_values(fixed_values)
        if sample_parameters is not None:
            self.set_sample_parameters(sample_parameters)
        if simulator is not None:
            self.set_simulator(simulator)
        if data is not None:
            self.set_data(data)

    def open(self, mode="r"):
        if self.read_only and mode != "r":
            raise RuntimeError(
                "The backend has been loaded in read-only "
                "mode. Set `read_only = False` to make "
                "changes."
            )
        f = h5py.File(self.filename, mode)
        return f

    @property
    def initialized(self):
        from pathlib import Path
        if not Path(self.filename).exists():
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OSError, IOError):
            return False

    def save_optimizer(self, optimizer):
        import pickle
        pickled_obj = pickle.dumps(optimizer)
        with self.open("a") as f:
            if self.name in f:
                del f[self.name]
            from h5py import string_dtype
            dt = string_dtype(length=len(pickled_obj))
            f.create_dataset(self.name, data=pickled_obj, dtype=dt)

    def load_optimizer(self):
        import pickle
        with self.open("a") as f:
            string = f[self.name][()]
        optimizer = pickle.loads(string)
        return optimizer

    def set_result(self, result, tol=None, popsize=None):
        with self.open("a") as f:
            f.attrs["x"] = result.x
            f.attrs["fun"] = result.fun
            f.attrs["message"] = np.array(result.message, dtype=string_dt)
            f.attrs["nfev"] = result.nfev
            f.attrs["nit"] = result.nit
            f.attrs["success"] = 1 if result.success else 0
            if tol is not None:
                f.attrs["tol"] = tol
            if popsize is not None:
                f.attrs["popsize"] = popsize

    @property
    def result(self):
        from scipy.optimize import OptimizeResult
        with self.open() as f:
            res = OptimizeResult(
                x=self.best_fit,
                fun=f.attrs["fun"],
                nfev=f.attrs["nfev"],
                nit=f.attrs["nit"],
                message=f.attrs["message"].decode("utf-8"),  # pylint: disable=E1101
                success=bool(f.attrs["success"]),
            )
        return res

    @property
    def best_fit(self):
        fit_pars = self.sample_parameters
        with self.open() as f:
            if fit_pars is not None:
                names = [par.name for par in fit_pars]
                return dict(zip(names, f.attrs["x"]))
            else:
                return f.attrs["x"]

    @property
    def tol(self):
        with self.open() as f:
            return f.attrs.get("tol", None)
