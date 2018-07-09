"""Weight generators."""

import math


def uniform_weighter():
    """Builds a generator that produces a stream of uniform weights (of 1).

    Example
    -------
    >>> weighter = uniform_weighter()
    >>> [weighter.__next__() for i in range(6)]
    [1, 1, 1, 1, 1, 1]
    """
    while True:
        yield 1


def first_n_uniform_weighter(n):
    """A generator whose first n weights are of weight 1 and the rest are 0.

    Parameters
    ----------
    n : int
        The number of first samples to have a weight of 1.

    Example
    -------
    >>> weighter = first_n_uniform_weighter(3)
    >>> [weighter.__next__() for i in range(6)]
    [1, 1, 1, 0, 0, 0]
    """
    for i in range(n):
        yield 1
    while True:
        yield 0


def exp_weighter(base):
    """A generator producing exponentialy increasing/decreasing weights.

    Parameters
    ----------
    base : float
        The base to raise by the index of the weight. Thus, the i-th weight
        produced will be base ^ i. The first weight is thus always 1. Also,
        bases larger than 1 produce increasing weights while bases between 0
        and 1 produce exponentially decreasing weights.

    Example
    -------
    >>> weighter = exp_weighter(3/4)
    >>> [weighter.__next__() for i in range(5)]
    [1.0, 0.75, 0.5625, 0.421875, 0.31640625]
    """
    i = 0
    while True:
        yield math.pow(base, i)
        i += 1


def exp_comp_weighter(n, concave_factor=None):
    """A generator producing exponent-complement decreasing weights.

    Parameters
    ----------
    n : int
        The number of weights to produce. All subsequent weights after the
        n-th weight are zero.
    concave : float, optional
        A number in (0,inf) effecting the concavity of the resulting function.
        The closer this number is to zero the more linear the decrease, while
        as it grows larger the decrease moe exponential (the function becomes
        more concave; more starting weights stay closer to 1, and the decrease
        comes later and becomes more pronounced). Defaults to 5.

    Examples
    --------
    >>> weighter = exp_comp_weighter(n=6, concave_factor=5)
    >>> ['{:.3f}'.format(weighter.__next__()) for i in range(6)]
    ['0.952', '0.911', '0.838', '0.702', '0.455', '0.000']
    >>> weighter = exp_comp_weighter(n=6, concave_factor=1.2)
    >>> ['{:.3f}'.format(weighter.__next__()) for i in range(6)]
    ['0.598', '0.518', '0.421', '0.306', '0.167', '0.000']
    """
    if concave_factor is None:
        concave_factor = 5
    base = 1 + concave_factor/n
    norm = math.pow(base, n-1)
    for i in range(n):
        yield 1 - (math.pow(base, i) / norm)
    while True:
        yield 0
