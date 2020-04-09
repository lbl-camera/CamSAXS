#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
from collections import OrderedDict

# Fitting related
_fixed = {'name': 'Fixed', 'value': True, 'type': 'bool'}
_bounds = [{'name': 'Lower', 'value': '-\u221E'}, {'name': 'Upper', 'value': '\u221E'}]
_bounded = {'name': 'Bounded', 'value': False, 'type': 'bool',
            'children': _bounds, 'visible': False, 'enabled': False}


class XicamParameter(yaml.YAMLObject):
    yaml_tag = u'!YMLParameter'

    def __init__(self, name, description, value, units, **kwags):
        self.name = name
        self.description = description
        self.value = value
        self.units = units
        self.fixed = True
        self.bounds = [None, None]
    def __repr__(self):
        return "%s (%r, value=%r, units=%r)" % \
               (self.__class__.__name__, self.name, self.value, self.units)

    @classmethod
    def from_yaml(cls, loader, node):
        opts = loader.construct_mapping(node)
        return cls(**opts)


def load_models():
    path,_= os.path.split(os.path.realpath(__file__))
    config = os.path.join(path, 'config.yml')
    if not os.path.isfile(config):
        raise ImportWarning('Unable to load config file')
        return None

    with open(config) as fp:
        yml = yaml.load(fp, Loader=yaml.Loader)

    model_tree = OrderedDict()
    for key, val in yml.items():
        models = OrderedDict()
        for name, params in val.items():
            _params = [p['param'] for p in params]
            models[name] = {'params': _params}
        model_tree[key] = models
    return model_tree
