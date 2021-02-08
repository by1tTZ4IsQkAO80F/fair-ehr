from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass
import numpy as np
import pandas as pd

def sanitize_json(d):
    """recursively formats dicts for json serialization"""
    if isinstance(d, list):
        d_new = []
        for v in d:
            d_new.append(sanitize_json(v))
        return d_new
    elif isinstance(d, dict):
        for k in d.keys():
            d[k] = sanitize_json(d[k])
    if isinstance(d,np.ndarray):
        return d.tolist()
    elif d.__class__.__name__.startswith('int'):
        return int(d)
    elif d.__class__.__name__.startswith('float'):
        return float(d)
    elif isinstance(d, pd.DataFrame) or isinstance(d, pd.Series):
        return d.values.tolist()
    return d


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

