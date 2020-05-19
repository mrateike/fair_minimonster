# from Floyd

import os
import sys

root_path = os.path.abspath(os.path.join(''))
if root_path not in sys.path:
    sys.path.append(root_path)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tikzplotlib as tpl
from src.training_evaluation import Statistics, ModelParameters


def _plot_results(
        utility,
        fairness,
        demographic_parity,
        equality_of_opportunity,
        xaxis,
        xlable,
        xscale,
        utility_uncertainty=None,
        fairness_uncertainty=None,
        demographic_parity_uncertainty=None,
        equality_of_opportunity_uncertainty=None,
        lambdas=None,
        lambdas_uncertainty=None,
        file_path=None,
        plot_fairness=False):
    if lambdas is not None:
        if plot_fairness:
            f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharex=True, figsize=(25, 10))
        else:
            f, (ax1, ax2, ax3, ax5) = plt.subplots(1, 4, sharex=True, figsize=(25, 10))

        ax5.plot(xaxis, lambdas)
        ax5.set_xlabel(xlable)
        ax5.set_ylabel("Lambda")
        ax5.set_xscale(xscale)
        # ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        if lambdas_uncertainty is not None:
            ax5.fill_between(xaxis, lambdas_uncertainty[0], lambdas_uncertainty[1], alpha=0.3, edgecolor='#060080',
                             facecolor='#928CFF')
    else:
        if plot_fairness:
            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, figsize=(25, 10))
        else:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(25, 10))

    ax1.plot(xaxis, utility)
    ax1.set_xlabel(xlable)
    ax1.set_ylabel("Utility")
    ax1.set_xscale(xscale)
    # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    if utility_uncertainty is not None:
        ax1.fill_between(xaxis, utility_uncertainty[0], utility_uncertainty[1], alpha=0.3, edgecolor='#060080',
                         facecolor='#928CFF')

    ax2.plot(xaxis, demographic_parity)
    ax2.set_xlabel(xlable)
    ax2.set_ylabel("Benefit Delta (Disparate Impact)")
    ax2.set_xscale(xscale)
    # ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    if demographic_parity_uncertainty is not None:
        ax2.fill_between(xaxis, demographic_parity_uncertainty[0], demographic_parity_uncertainty[1], alpha=0.3,
                         edgecolor='#060080', facecolor='#928CFF')

    ax3.plot(xaxis, equality_of_opportunity)
    ax3.set_xlabel(xlable)
    ax3.set_ylabel("Benefit Delta (Equality of Opportunity)")
    ax3.set_xscale(xscale)
    # ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    if equality_of_opportunity_uncertainty is not None:
        ax3.fill_between(xaxis, equality_of_opportunity_uncertainty[0], equality_of_opportunity_uncertainty[1],
                         alpha=0.3, edgecolor='#060080', facecolor='#928CFF')

    if plot_fairness:
        ax4.plot(xaxis, fairness)
        ax4.set_xlabel(xlable)
        ax4.set_ylabel("Fairness Function")
        ax4.set_xscale(xscale)
        # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        if fairness_uncertainty is not None:
            ax4.fill_between(xaxis, fairness_uncertainty[0], fairness_uncertainty[1], alpha=0.3, edgecolor='#060080',
                             facecolor='#928CFF')

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)
        tpl.save(file_path.replace(".png", ".tex"), figure=f, axis_width='\\figwidth', axis_height='\\figheight',
                 tex_relative_path_to_data='')

    plt.close('all')


def plot_median(statistics, file_path=None, model_parameters=None, plot_fairness=False):
    if model_parameters is not None:
        lambdas = model_parameters.get_lagrangians(result_format=ModelParameters.MEDIAN)
        lambdas_uncertainty = (model_parameters.get_lagrangians(result_format=ModelParameters.FIRST_QUARTILE),
                               model_parameters.get_lagrangians(result_format=ModelParameters.THIRD_QUARTILE))
    else:
        lambdas = None
        lambdas_uncertainty = None

    _plot_results(
        utility=statistics.performance(measure_key=Statistics.UTILITY, result_format=Statistics.MEDIAN),
        fairness=statistics.fairness(measure_key=Statistics.FAIRNESS, result_format=Statistics.MEDIAN),
        demographic_parity=statistics.fairness(measure_key=Statistics.DEMOGRAPHIC_PARITY,
                                               result_format=Statistics.MEDIAN),
        equality_of_opportunity=statistics.fairness(measure_key=Statistics.EQUALITY_OF_OPPORTUNITY,
                                                    result_format=Statistics.MEDIAN),
        xaxis=statistics.results[Statistics.X_VALUES],
        xlable=statistics.results[Statistics.X_NAME],
        xscale=statistics.results[Statistics.X_SCALE],
        utility_uncertainty=
        (statistics.performance(measure_key=Statistics.UTILITY, result_format=Statistics.FIRST_QUARTILE),
         statistics.performance(measure_key=Statistics.UTILITY, result_format=Statistics.THIRD_QUARTILE)),
        fairness_uncertainty=
        (statistics.fairness(measure_key=Statistics.FAIRNESS, result_format=Statistics.FIRST_QUARTILE),
         statistics.fairness(measure_key=Statistics.FAIRNESS, result_format=Statistics.THIRD_QUARTILE)),
        demographic_parity_uncertainty=
        (statistics.fairness(measure_key=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.FIRST_QUARTILE),
         statistics.fairness(measure_key=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.THIRD_QUARTILE)),
        equality_of_opportunity_uncertainty=
        (statistics.fairness(measure_key=Statistics.EQUALITY_OF_OPPORTUNITY, result_format=Statistics.FIRST_QUARTILE),
         statistics.fairness(measure_key=Statistics.EQUALITY_OF_OPPORTUNITY, result_format=Statistics.THIRD_QUARTILE)),
        lambdas=lambdas,
        lambdas_uncertainty=lambdas_uncertainty,
        file_path=file_path,
        plot_fairness=plot_fairness)


def plot_mean(statistics, file_path=None, model_parameters=None, plot_fairness=False):
    u_mean = statistics.performance(measure_key=Statistics.UTILITY, result_format=Statistics.MEAN)
    u_stddev = statistics.performance(measure_key=Statistics.UTILITY, result_format=Statistics.STANDARD_DEVIATION)

    f_mean = statistics.fairness(measure_key=Statistics.FAIRNESS, result_format=Statistics.MEAN)
    f_stddev = statistics.fairness(measure_key=Statistics.FAIRNESS, result_format=Statistics.STANDARD_DEVIATION)

    dp_mean = statistics.fairness(measure_key=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.MEAN)
    dp_stddev = statistics.fairness(measure_key=Statistics.DEMOGRAPHIC_PARITY,
                                    result_format=Statistics.STANDARD_DEVIATION)

    eop_mean = statistics.fairness(measure_key=Statistics.EQUALITY_OF_OPPORTUNITY, result_format=Statistics.MEAN)
    eop_stddev = statistics.fairness(measure_key=Statistics.EQUALITY_OF_OPPORTUNITY,
                                     result_format=Statistics.STANDARD_DEVIATION)

    if model_parameters is not None:
        lambdas = model_parameters.get_lagrangians(result_format=ModelParameters.MEAN)
        lambdas_uncertainty = (
        model_parameters.get_lagrangians(result_format=ModelParameters.MEAN) - model_parameters.get_lagrangians(
            result_format=ModelParameters.STANDARD_DEVIATION),
            model_parameters.get_lagrangians(result_format=ModelParameters.MEAN) + model_parameters.get_lagrangians(result_format=ModelParameters.STANDARD_DEVIATION))
    else:
        lambdas = None
        lambdas_uncertainty = None

    _plot_results(
        utility=u_mean,
        fairness=f_mean,
        demographic_parity=dp_mean,
        equality_of_opportunity=eop_mean,
        xaxis=statistics.results[Statistics.X_VALUES],
        xlable=statistics.results[Statistics.X_NAME],
        xscale=statistics.results[Statistics.X_SCALE],
        utility_uncertainty=(u_mean - u_stddev, u_mean + u_stddev),
        fairness_uncertainty=(f_mean - f_stddev, f_mean + f_stddev),
        demographic_parity_uncertainty=(dp_mean - dp_stddev, dp_mean + dp_stddev),
        equality_of_opportunity_uncertainty=(eop_mean - eop_stddev, eop_mean + eop_stddev),
        lambdas=lambdas,
        lambdas_uncertainty=lambdas_uncertainty,
        file_path=file_path,
        plot_fairness=plot_fairness)