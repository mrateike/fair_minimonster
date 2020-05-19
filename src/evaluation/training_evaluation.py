# from Floyd

import os
import sys

root_path = os.path.abspath(os.path.join(''))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import numbers
from copy import deepcopy

from data.util import stack, serialize_dictionary

# Result Format
MEAN = "MEAN"
STANDARD_DEVIATION = "STDDEV"
MEDIAN = "MEDIAN"
FIRST_QUARTILE = "FIRST_QUARTILE"
THIRD_QUARTILE = "THIRD_QUARTILE"


def _unserialize_value(value):
    if isinstance(value, dict):
        return _unserialize_dictionary(value)
    elif isinstance(value, list):
        return np.array(value)
    elif value == "NoneType":
        return None
    else:
        return value


def _unserialize_dictionary(dictionary):
    unserialized_dict = deepcopy(dictionary)
    for key, value in unserialized_dict.items():
        unserialized_dict[key] = _unserialize_value(value)

    return unserialized_dict


def build_result_dictionary(measure):
    return {
        MEAN: np.mean(measure, axis=1),
        MEDIAN: np.median(measure, axis=1),
        STANDARD_DEVIATION: np.std(measure, axis=1),
        FIRST_QUARTILE: np.percentile(measure, q=25, axis=1),
        THIRD_QUARTILE: np.percentile(measure, q=75, axis=1)
    }


class ModelParameters(object):
    # Result Format
    MEAN = MEAN
    STANDARD_DEVIATION = STANDARD_DEVIATION
    MEDIAN = MEDIAN
    FIRST_QUARTILE = FIRST_QUARTILE
    THIRD_QUARTILE = THIRD_QUARTILE

    def __init__(self, model_parameter_dict):
        self.dict = deepcopy(model_parameter_dict)
        self.dict["lambdas"] = np.array(self.dict["lambdas"], dtype=float).reshape(-1, 1)

    @staticmethod
    def build_from_serialized_dictionary(serialized_dict):
        results = _unserialize_dictionary(serialized_dict)
        return ModelParameters(results)

    def merge(self, model_parameters):
        self.dict["model_parameters"].append(model_parameters.dict["model_parameters"])
        self.dict["lambdas"] = stack(self.dict["lambdas"],
                                     model_parameters.dict["lambdas"].reshape(-1, 1),
                                     axis=1)

    def get_lagrangians(self, result_format):
        return build_result_dictionary(self.dict["lambdas"])[result_format]

    def to_dict(self):
        return deepcopy(self.dict)


def TPR(statistics, prot):
    return statistics.results[prot][Statistics.TRUE_POSITIVES] / statistics.results[prot][Statistics.NUM_POSITIVES]


def FPR(statistics, prot):
    return statistics.results[prot][Statistics.FALSE_POSITIVES] / statistics.results[prot][Statistics.NUM_POSITIVES]


def TNR(statistics, prot):
    return statistics.results[prot][Statistics.TRUE_NEGATIVES] / statistics.results[prot][Statistics.NUM_NEGATIVES]


def FNR(statistics, prot):
    return statistics.results[prot][Statistics.FALSE_NEGATIVES] / statistics.results[prot][Statistics.NUM_NEGATIVES]


def PPV(statistics, prot):
    return statistics.results[prot][Statistics.TRUE_POSITIVES] / statistics.results[prot][Statistics.NUM_PRED_POSITIVES]


def NPV(statistics, prot):
    return statistics.results[prot][Statistics.TRUE_NEGATIVES] / statistics.results[prot][Statistics.NUM_PRED_NEGATIVES]


def FDR(statistics, prot):
    return statistics.results[prot][Statistics.FALSE_POSITIVES] / statistics.results[prot][
        Statistics.NUM_PRED_POSITIVES]


def FOR(statistics, prot):
    return statistics.results[prot][Statistics.FALSE_NEGATIVES] / statistics.results[prot][
        Statistics.NUM_PRED_NEGATIVES]


def ACC(statistics, prot):
    return (statistics.results[prot][Statistics.TRUE_POSITIVES] + statistics.results[prot][Statistics.TRUE_NEGATIVES]) / \
           statistics.results[prot][Statistics.NUM_INDIVIDUALS]


def ERR(statistics, prot):
    return statistics._get_measure(prot, Statistics.ACCURACY)


def SEL(statistics, prot):
    return statistics.results[prot][Statistics.NUM_PRED_POSITIVES] / statistics.results[prot][
        Statistics.NUM_INDIVIDUALS]


def F1(statistics, prot):
    return (2 * statistics.results[prot][Statistics.TRUE_POSITIVES]) / (
            2 * statistics.results[prot][Statistics.TRUE_POSITIVES] + statistics.results[prot][
        Statistics.FALSE_POSITIVES] + statistics.results[prot][Statistics.FALSE_POSITIVES])


def DI(statistics, prot):
    return statistics._get_measure("protected", Statistics.SELECTION_RATE) / statistics._get_measure("unprotected",
                                                                                                     Statistics.SELECTION_RATE)


def DP(statistics, prot):
    return statistics._get_measure("protected", Statistics.SELECTION_RATE) - statistics._get_measure("unprotected",
                                                                                                     Statistics.SELECTION_RATE)


def EOP(statistics, prot):
    return statistics._get_measure("protected", Statistics.TRUE_POSITIVE_RATE) - statistics._get_measure("unprotected",
                                                                                                         Statistics.TRUE_POSITIVE_RATE)


class Statistics(object):
    # Scale Measures:
    X_VALUES = "X_VALUES"
    X_SCALE = "X_SCALE"
    X_NAME = "X_NAME"

    # Performance Measures
    UTILITY = "U"
    NUM_INDIVIDUALS = "A"
    NUM_NEGATIVES = "N"
    NUM_POSITIVES = "P"
    NUM_PRED_NEGATIVES = "NPRED"
    NUM_PRED_POSITIVES = "NPRED"
    TRUE_POSITIVES = "TP"
    TRUE_NEGATIVES = "TN"
    FALSE_POSITIVES = "FP"
    FALSE_NEGATIVES = "FN"
    TRUE_POSITIVE_RATE = "TPR"
    FALSE_POSITIVE_RATE = "FPR"
    TRUE_NEGATIVE_RATE = "TNR"
    FALSE_NEGATIVE_RATE = "FNR"
    POSITIVE_PREDICTIVE_VALUE = "PPV"
    NEGATIVE_PREDICTIVE_VALUE = "NPV"
    FALSE_DISCOVERY_RATE = "FDR"
    FALSE_OMISSION_RATE = "FOR"
    ACCURACY = "ACC"
    ERROR_RATE = "ERR"
    SELECTION_RATE = "SEL"
    F1 = "F1"

    # Fairness Measures
    DISPARATE_IMPACT = "DI"
    DEMOGRAPHIC_PARITY = "DP"
    EQUALITY_OF_OPPORTUNITY = "EOP"
    FAIRNESS = "FAIR"

    # Result Format
    MEAN = MEAN
    STANDARD_DEVIATION = STANDARD_DEVIATION
    MEDIAN = MEDIAN
    FIRST_QUARTILE = FIRST_QUARTILE
    THIRD_QUARTILE = THIRD_QUARTILE

    def __init__(self, results):
        self.functions = {
            Statistics.TRUE_POSITIVE_RATE: TPR,
            Statistics.FALSE_POSITIVE_RATE: FPR,
            Statistics.TRUE_NEGATIVE_RATE: TNR,
            Statistics.FALSE_NEGATIVE_RATE: FNR,
            Statistics.POSITIVE_PREDICTIVE_VALUE: PPV,
            Statistics.NEGATIVE_PREDICTIVE_VALUE: NPV,
            Statistics.FALSE_DISCOVERY_RATE: FDR,
            Statistics.FALSE_OMISSION_RATE: FOR,
            Statistics.ACCURACY: ACC,
            Statistics.ERROR_RATE: ERR,
            Statistics.SELECTION_RATE: SEL,
            Statistics.F1: F1,
            Statistics.DISPARATE_IMPACT: DI,
            Statistics.DEMOGRAPHIC_PARITY: DP,
            Statistics.EQUALITY_OF_OPPORTUNITY: EOP
        }
        self.results = deepcopy(results)

    @staticmethod
    #def build(predictions, protected_attributes, ground_truths):
    # y_pred, a_test, y_test
    def build(predictions, observations, fairness, utility, protected_attributes, ground_truths):
        results = {
            Statistics.X_VALUES: list(range(0, predictions.shape[1])),
            Statistics.X_SCALE: "linear",
            Statistics.X_NAME: "Timestep"
        }

        for prot in ["all", "unprotected", "protected"]:
            if prot == "unprotected":
                filtered_predictions = predictions[(protected_attributes == 0).squeeze(), :]
                filtered_ground_truths = np.expand_dims(ground_truths[protected_attributes == 0], axis=1)
            elif prot == "protected":
                filtered_predictions = predictions[(protected_attributes == 1).squeeze(), :]
                filtered_ground_truths = np.expand_dims(ground_truths[protected_attributes == 1], axis=1)
            else:
                filtered_predictions = predictions
                filtered_ground_truths = ground_truths

            utility_matching_gt = np.repeat(filtered_ground_truths, filtered_predictions.shape[1], axis=1)

            # calculate base statistics during creation of statistics object
            results[prot] = {
                Statistics.UTILITY: utility,
                Statistics.NUM_INDIVIDUALS: len(filtered_ground_truths),
                Statistics.NUM_NEGATIVES: len(filtered_ground_truths[filtered_ground_truths == 0]),
                Statistics.NUM_POSITIVES: len(filtered_ground_truths[filtered_ground_truths == 1]),
                Statistics.NUM_PRED_NEGATIVES: np.expand_dims(np.sum((1 - filtered_predictions), axis=0), axis=1),
                Statistics.NUM_PRED_POSITIVES: np.expand_dims(np.sum(filtered_predictions, axis=0), axis=1),
                Statistics.TRUE_POSITIVES: np.expand_dims(
                    np.sum(np.logical_and(filtered_predictions == 1, utility_matching_gt == 1), axis=0), axis=1),
                Statistics.TRUE_NEGATIVES: np.expand_dims(
                    np.sum(np.logical_and(filtered_predictions == 0, utility_matching_gt == 0), axis=0), axis=1),
                Statistics.FALSE_POSITIVES: np.expand_dims(
                    np.sum(np.logical_and(filtered_predictions == 1, utility_matching_gt == 0), axis=0), axis=1),
                Statistics.FALSE_NEGATIVES: np.expand_dims(
                    np.sum(np.logical_and(filtered_predictions == 0, utility_matching_gt == 1), axis=0), axis=1)
            }

            # calculate futher statistics based on the base statistics only on demand to save memory
            results[prot][Statistics.TRUE_POSITIVE_RATE] = None
            results[prot][Statistics.FALSE_POSITIVE_RATE] = None
            results[prot][Statistics.TRUE_NEGATIVE_RATE] = None
            results[prot][Statistics.FALSE_NEGATIVE_RATE] = None
            results[prot][Statistics.POSITIVE_PREDICTIVE_VALUE] = None
            results[prot][Statistics.NEGATIVE_PREDICTIVE_VALUE] = None
            results[prot][Statistics.FALSE_DISCOVERY_RATE] = None
            results[prot][Statistics.FALSE_OMISSION_RATE] = None
            results[prot][Statistics.ACCURACY] = None
            results[prot][Statistics.ERROR_RATE] = None
            results[prot][Statistics.SELECTION_RATE] = None
            results[prot][Statistics.F1] = None

        results["all"][Statistics.DISPARATE_IMPACT] = None
        results["all"][Statistics.DEMOGRAPHIC_PARITY] = None
        results["all"][Statistics.EQUALITY_OF_OPPORTUNITY] = None
        results["all"][Statistics.FAIRNESS] = fairness
        return Statistics(results)

    @staticmethod
    def build_from_serialized_dictionary(serialized_dict):
        results = _unserialize_dictionary(serialized_dict)
        return Statistics(results)

    def _get_measure(self, prot, measure_key):
        if self.results[prot][measure_key] is None:
            measure = self.functions[measure_key](self, prot)
            self.results[prot][measure_key] = measure
        else:
            measure = self.results[prot][measure_key]

        return measure

    def performance(self, measure_key, result_format, protected=None):
        if protected:
            prot = "protected"
        elif protected is None:
            prot = "all"
        else:
            prot = "unprotected"

        measure = self._get_measure(prot, measure_key)
        return build_result_dictionary(measure)[result_format]

    def fairness(self, measure_key, result_format):
        measure = self._get_measure("all", measure_key)
        return build_result_dictionary(measure)[result_format]

    def to_dict(self):
        return serialize_dictionary(self.results)

    def merge(self, statistics):
        for (protected_key, protected_value) in statistics.results.items():
            if protected_key != Statistics.X_VALUES and protected_key != Statistics.X_SCALE and protected_key != Statistics.X_NAME:
                for (measure_key, measure_value) in protected_value.items():
                    if measure_key != Statistics.NUM_INDIVIDUALS and measure_key != Statistics.NUM_NEGATIVES and measure_key != Statistics.NUM_POSITIVES:
                        # if the measure value has not been calculated for either of the merging statistics: reset and let it be recalculated during next call
                        if measure_value is None or self.results[protected_key][measure_key] is None:
                            self.results[protected_key][measure_key] = None
                        else:
                            self.results[protected_key][measure_key] = stack(self.results[protected_key][measure_key],
                                                                             measure_value, axis=1)


class MultiStatistics(Statistics):
    def __init__(self, results):
        super(MultiStatistics, self).__init__(results)

    @staticmethod
    def build(x_scale, x_values, x_name):
        results = {}
        results[MultiStatistics.X_VALUES] = x_values
        results[Statistics.X_SCALE] = x_scale
        results[Statistics.X_NAME] = x_name

        for prot in ["all", "unprotected", "protected"]:
            results[prot] = {
                MultiStatistics.UTILITY: None,
                MultiStatistics.NUM_INDIVIDUALS: None,
                MultiStatistics.NUM_NEGATIVES: None,
                MultiStatistics.NUM_POSITIVES: None,
                MultiStatistics.NUM_PRED_NEGATIVES: None,
                MultiStatistics.NUM_PRED_POSITIVES: None,
                MultiStatistics.TRUE_POSITIVES: None,
                MultiStatistics.TRUE_NEGATIVES: None,
                MultiStatistics.FALSE_POSITIVES: None,
                MultiStatistics.FALSE_NEGATIVES: None,
                MultiStatistics.TRUE_POSITIVE_RATE: None,
                MultiStatistics.FALSE_POSITIVE_RATE: None,
                MultiStatistics.TRUE_NEGATIVE_RATE: None,
                MultiStatistics.FALSE_NEGATIVE_RATE: None,
                MultiStatistics.POSITIVE_PREDICTIVE_VALUE: None,
                MultiStatistics.NEGATIVE_PREDICTIVE_VALUE: None,
                MultiStatistics.FALSE_DISCOVERY_RATE: None,
                MultiStatistics.FALSE_OMISSION_RATE: None,
                MultiStatistics.ACCURACY: None,
                MultiStatistics.ERROR_RATE: None,
                MultiStatistics.SELECTION_RATE: None,
                MultiStatistics.F1: None
            }

        results["all"][MultiStatistics.DISPARATE_IMPACT] = None
        results["all"][MultiStatistics.DEMOGRAPHIC_PARITY] = None
        results["all"][MultiStatistics.EQUALITY_OF_OPPORTUNITY] = None
        results["all"][MultiStatistics.FAIRNESS] = None
        return MultiStatistics(results)

    def log_run(self, statistics):
        for (protected_key, protected_value) in statistics.results.items():
            if protected_key != MultiStatistics.X_VALUES and protected_key != MultiStatistics.X_SCALE and protected_key != MultiStatistics.X_NAME:
                for measure_key in protected_value:
                    measure_value = statistics._get_measure(protected_key, measure_key)

                    if isinstance(measure_value, numbers.Number):
                        value = measure_value
                    else:
                        value = measure_value[-1, :].reshape(1, -1)

                    if self.results[protected_key][measure_key] is None:
                        self.results[protected_key][measure_key] = value
                    else:
                        self.results[protected_key][measure_key] = np.vstack(
                            (self.results[protected_key][measure_key], value))