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


__doc__ = """
.. currentmodule:: pydemic
.. autoclass:: Reaction
.. autoclass:: ErlangProcess
.. autoclass:: GammaProcess
.. autoclass:: PassiveReaction
"""


class Reaction:
    """
    .. attribute:: lhs

        A :class:`string` specifying the compartment sourcing
        the reaction.
        Only valid python identifiers are allowed, as their state is accessed
        as an attribute.

    .. attribute:: rhs

        A :class:`string` specifying the compartment being sourced
        by the reaction.
        Only valid python identifiers are allowed, as their state is accessed
        as an attribute.

    .. attribute:: evaluator

        A :class:`callable` with signature ``(t, y)``
        of the time and :class:`SimulationState` returning the rate
        associated with the reaction.
        See the documentation of :class:`~pydemic.simulation.SimulationState`
        for guidance on accessing compartment state information from ``y``.

    .. automethod:: get_reactions
    """

    def __init__(self, lhs, rhs, evaluator):

        self.lhs = lhs
        self.rhs = rhs
        self.evaluator = evaluator

    def __repr__(self):
        return "{0:s} --> {1:s}".format(str(self.lhs), str(self.rhs))

    def get_reactions(self):
        """
        :returns: A :class:`tuple` of the fundamental :class:`Reaction`'s
            comprising the full process.
        """

        return tuple([self])


class PassiveReaction(Reaction):
    """
    A reaction used for bookkeeping purposes.

    .. automethod:: get_reactions
    """
    pass


class ErlangProcess(Reaction):
    """
    .. attribute:: lhs

        A :class:`string` specifying the compartment sourcing
        the reaction.
        Only valid python identifiers are allowed, as their state is accessed
        as an attribute.

    .. attribute:: rhs

        A :class:`string` specifying the compartment being sourced
        by the reaction.
        Only valid python identifiers are allowed, as their state is accessed
        as an attribute.

    .. attribute:: shape

    .. attribute:: scale
    .. automethod:: get_reactions
    """

    def __init__(self, lhs, rhs, shape, scale):
        self.lhs = lhs
        self.rhs = rhs
        self.shape = shape
        self.scale = scale

    def get_reactions(self):
        """
        :returns: A :class:`tuple` of the fundamental :class:`Reaction`'s
            comprising the full process.
        """

        intermediaries = [
            self.lhs + ":{0:s}:{1:d}".format(self.rhs, k)
            for k in range(0, self.shape-1)
        ]
        stages = [self.lhs] + intermediaries + [self.rhs]
        reactions = [
            # hack so that lhs resolves to current value rather than final
            # see https://stackoverflow.com/a/2295372
            Reaction(lhs, rhs, lambda t, y, lhs=lhs: y.y[lhs] / self.scale(t, y))
            for lhs, rhs in zip(stages[:-1], stages[1:])
        ]
        return tuple(reactions)


class GammaProcess(ErlangProcess):
    """
    Currently, an alias to :class:`ErlangProcess`.
    """

    pass
