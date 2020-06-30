# (c) Floyd Kretschmar (floydkretschmar)

import copy
import inspect
import json
import numbers

import numpy as np
from numpy.random import RandomState


"""
(c) Floyd Kretschmar (https://github.com/floydkretschmar/master-thesis)
Specifies helper functions used in other classes
"""


np.seterr(divide='ignore', invalid='ignore', over='ignore')



def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def save_dictionary(dictionary, path):
    try:
        with open(path, 'w+') as file_path:
            json.dump(dictionary, file_path)
    except Exception as e:
        print('Saving file {} failed with exception: \n {}'.format(path, str(e)))


def serialize_value(value):
    if isinstance(value, dict):
        return serialize_dictionary(value)
    elif isinstance(value, list):
        return serialize_list(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif inspect.isfunction(value):
        return value.__name__
    elif not (
            isinstance(value, str) or isinstance(value, numbers.Number) or isinstance(value, list) or isinstance(value,
                                                                                                                 bool)):
        return type(value).__name__
    else:
        return value


def serialize_dictionary(dictionary):
    serialized_dict = copy.deepcopy(dictionary)
    for key, value in serialized_dict.items():
        serialized_dict[key] = serialize_value(value)

    return serialized_dict


def serialize_list(unserialized_list):
    serialized_list = []
    for value in unserialized_list:
        serialized_list.append(serialize_value(value))

    return serialized_list
