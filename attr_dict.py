#!/usr/bin/python3

reserved_dict_attrs = set(dir(dict)) | {'__dict__'}

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
        if isinstance(key, str) and key in reserved_dict_attrs:
            raise KeyError(f"Key '{key}' reserved as original dict attribute")
        if isinstance(val, dict) and not isinstance(val, AttrDict):
            val = AttrDict(val)
        super().__setitem__(key, val)

    def update(self, *args, **kwargs):
        for key, val in dict(*args, **kwargs).items():
            self.__setitem__(key, val)

    # Allow dict keys to be called, set, and deleted like class attributes
    def __getattr__(self, name):
        try:
            # not concerned with reserved attrs since __getattr__ is only called
            # when __getattribute__ (which will find them in dict) fails
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

    # Allow tab completion
    def __dir__(self):
        # include normal dict attrs and methods
        attrs = set(super().__dir__())

        # include AttrDict keys that are valid variable names
        # (e.g., no fully numeric or hyphen-containing keys)
        attrs.update([key for key in self.keys()
                      if isinstance(key, str) and key.isidentifier()])

        return sorted(attrs)
