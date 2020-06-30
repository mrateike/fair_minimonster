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


def get_random(seed=None):
    if seed is None:
        return RandomState()
    else:
        return RandomState(seed)


def get_list_of_seeds(number_of_seeds):
    max_value = 2 ** 32 - 1
    seeds = get_random().randint(
        0,
        max_value,
        size=number_of_seeds,
        dtype=np.dtype("int64"))
    return seeds


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


def load_dictionary(path):
    try:
        with open(path, 'r') as file_path:
            return json.load(file_path)
    except Exception as e:
        print('Loading file {} failed with exception: \n {}'.format(path, str(e)))
        return None


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



def train_test_split(x, y, s, test_size):
    indices = np.array(range(x.shape[0]))

    boundary = int(len(indices) * test_size)
    test_indices, train_indices = np.split(get_random().permutation(indices), [boundary])
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices], s[train_indices], s[test_indices]