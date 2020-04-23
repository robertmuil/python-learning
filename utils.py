import numpy as np

def rescale(x, new_range, old_range=None):
    """ 
    Rescale x from its original range to a new range, allowing the range to be inverted (hi goes to low)
    
    TODO: properly handle negative `x` (fails at least when the range is inverted)

    
    >>> rescale([0, 1, 2], (0, 10))
    array([ 0.,  5., 10.])
    
    >>> rescale([0, -10, 10], (0, 1))
    array([0.5, 0. , 1. ])
    
    >>> rescale([0, -10, 10], (0, 1), (0, 10))
    array([0., 0., 1.])
    
    >>> rescale([0, -10, 10], (0, 1), (10, 0))
    array([1., 1., 0.])
    
    >>> rescale([0, -10, 10], (1, 0), (0, 10))
    array([1., 1., 0.])
    
    >>> rescale([0, 1, 2], (0, -4))
    array([ 0., -2., -4.])

    """
    x = np.array(x)
    if old_range is None:
        old_range = (np.min(x), np.max(x))
    
    new_span = new_range[1] - new_range[0]
    old_span = old_range[1] - old_range[0]
    span_mult = new_span / old_span
    
    _lo_hi = new_range if new_range[0] < new_range[1] else (new_range[1], new_range[0])
    return (((x - old_range[0]) * span_mult) + new_range[0]).clip(*_lo_hi)
    
    
def getAlphaBeta(mu, sigma):
    """
    Estimate parameters of beta dist.
    from: https://stats.stackexchange.com/a/395329
    """
    alpha = mu**2 * ((1 - mu) / sigma**2 - 1 / mu)
    beta = alpha * (1 / mu - 1)
    return (alpha, beta)

        
if __name__ == "__main__":
    import doctest
    doctest.testmod()