# from Floyd

import os
import sys

root_path = os.path.abspath(os.path.join(''))
if root_path not in sys.path:
    sys.path.append(root_path)

import matplotlib

# mpl.use('Qt5Agg')
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import tikzplotlib as tpl
from src.evaluation.training_evaluation import MEAN, MEDIAN


def _plot_results(plotting_dictionary, file_path, figsize, plots_per_row):
    x = plotting_dictionary["plot_info"]["x_axis"]
    x_scale = plotting_dictionary["plot_info"]["x_scale"]
    x_label = plotting_dictionary["plot_info"]["x_label"]

    performance_measures = plotting_dictionary["performance_measures"]
    fairness_measures = plotting_dictionary["fairness_measures"]

    num_columns = min(len(performance_measures.items()), plots_per_row)
    if num_columns < plots_per_row:
        num_columns = min(max(len(fairness_measures.items()), num_columns), plots_per_row)

    # get num_rows for maximum of plots_per_row graphs per row
    num_rows = (len(performance_measures.items()) // num_columns) + (
        1 if len(performance_measures.items()) % num_columns > 0 else 0)
    num_rows += (len(fairness_measures.items()) // num_columns) + (
        1 if len(fairness_measures.items()) % num_columns > 0 else 0)

    if figsize is None:
        figure = plt.figure(constrained_layout=True)
    else:
        figure = plt.figure(constrained_layout=True, figsize=figsize, dpi=80)

    grid = GridSpec(nrows=num_rows, ncols=num_columns, figure=figure)

    current_row = 0
    current_column = 0

    for measure_dict in [performance_measures, fairness_measures]:
        for y_label, y_dict in measure_dict.items():
            y = y_dict["value"]
            y_uncertainty_lower = y_dict["uncertainty_lower_bound"]
            y_uncertainty_upper = y_dict["uncertainty_upper_bound"]

            axis = figure.add_subplot(grid[current_row, current_column])
            axis.plot(x, y)
            axis.set_xlabel(x_label)
            axis.title.set_text(y_label)
            axis.set_xscale(x_scale)
            axis.fill_between(x,
                              y_uncertainty_lower,
                              y_uncertainty_upper,
                              alpha=0.3,
                              edgecolor='#060080',
                              facecolor='#928CFF')

            if current_column < num_columns - 1:
                current_column += 1
            else:
                current_column = 0
                current_row += 1

        if current_column > 0:
            current_row += 1
            current_column = 0

    plt.savefig(file_path)
    tpl.save(file_path.replace(".png", ".tex"),
             figure=figure,
             axis_width='\\figwidth',
             axis_height='\\figheight',
             tex_relative_path_to_data='.',
             extra_groupstyle_parameters={"horizontal sep=1.2cm"},
             extra_axis_parameters={"scaled y ticks = false, \n yticklabel style = {/pgf/number format/fixed, /pgf/number format/precision=3}"})
    plt.close('all')


def plot_median(x_values,
                x_label,
                x_scale,
                performance_measures,
                fairness_measures,
                file_path,
                figsize=None,
                plots_per_row=4):
    plotting_dict = _build_plot_dict(x_values, x_label, x_scale, performance_measures, fairness_measures, MEDIAN)
    _plot_results(plotting_dict, file_path, figsize, plots_per_row)


def plot_mean(x_values,
              x_label,
              x_scale,
              performance_measures,
              fairness_measures,
              file_path,
              figsize=None,
              plots_per_row=4):
    plotting_dict = _build_plot_dict(x_values, x_label, x_scale, performance_measures, fairness_measures, MEAN)
    _plot_results(plotting_dict, file_path, figsize, plots_per_row)


def _build_plot_dict(x_values,
                     x_label,
                     x_scale,
                     performance_measures,
                     fairness_measures,
                     result_format):
    p_measures = performance_measures if isinstance(performance_measures, list) else [performance_measures]
    f_measures = fairness_measures if isinstance(fairness_measures, list) else [fairness_measures]

    plotting_dict = {
        "plot_info": {
            "x_axis": x_values,
            "x_label": x_label,
            "x_scale": x_scale
        },
        "performance_measures": {},
        "fairness_measures": {}
    }

    for measure_set, measure_set_key in [(p_measures, "performance_measures"), (f_measures, "fairness_measures")]:
        for measure in measure_set:
            measure_label = measure.name

            if result_format == MEAN:
                value = measure.mean()
                measure_stddev = measure.standard_deviation()
                lower_bound = value - measure_stddev
                upper_bound = value + measure_stddev
            elif result_format == MEDIAN:
                value = measure.median()
                lower_bound = measure.first_quartile()
                upper_bound = measure.third_quartile()

            plotting_dict[measure_set_key][measure_label] = {
                "value": value,
                "uncertainty_lower_bound": lower_bound,
                "uncertainty_upper_bound": upper_bound
            }

    return plotting_dict