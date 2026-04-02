#!/usr/bin/python3

class AttrDict(dict):
    '''
    Allow keys of a dictionary to also be called like class attributes:

        >>> regions = {'Milotic': 'Hoenn'}
        >>> regions['Milotic']
        Hoenn
        >>> regions.Milotic
        AttributeError: 'dict' object has no attribute 'Milotic'

        >>> regions = AttrDict({'Milotic': 'Hoenn'})
        >>> regions['Milotic']  # standard dict behavior
        Hoenn
        >>> regions.Milotic  # behavior enabled by AttrDict
        Hoenn

    Also handles setting and deleting keys/attrs. Nested dicts are also
    converted to AttrDict instances. Less-than-free-range code.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    # Handle setting new keys, turning any nested dicts into AttrDicts as well
    def __setitem__(self, key, val):
        if isinstance(val, dict) and not isinstance(val, AttrDict):
            val = AttrDict(val)
        super().__setitem__(key, val)

    def update(self, *args, **kwargs):
        for key, val in dict(*args, **kwargs).items():
            self.__setitem__(key, val)

    # Allow dict keys to be called, set, and deleted like class attributes
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, val):
        # leave handling private variable assignment to parent dict type
        if name.startswith('_'):
            return super().__setattr__(name, val)
        self.__setitem__(name, val)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)
