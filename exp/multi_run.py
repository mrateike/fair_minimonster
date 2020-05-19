import argparse
import subprocess
from copy import deepcopy
import numpy as np
import os
import sys

root_path = os.path.abspath(os.path.join("."))
if root_path not in sys.path:
    sys.path.append(root_path)





# def _fairness_extensions(args, fairness_rates, build=False):
#     extensions = []
#     for fairness_rate in fairness_rates:
#         if build:
#             extension = "-f {} -fv {}".format(args.fairness_type, fairness_rate)
#         else:
#             extension = ["-f", str(args.fairness_type), "-fv", str(fairness_rate)]
#
#         if args.fairness_learning_rates is not None:
#             for learning_rate in args.fairness_learning_rates:
#                 for batch_size in args.fairness_batch_sizes:
#                     for epochs in args.fairness_epochs:
#                         if build:
#                             extensions.append("{} -flr {} -fbs {} -fe {}".format(extension,
#                                                                                  learning_rate,
#                                                                                  batch_size,
#                                                                                  epochs))
#                         else:
#                             temp_extension = deepcopy(extension)
#                             temp_extension.extend(["-flr", str(learning_rate),
#                                                    "-fbs", str(batch_size),
#                                                    "-fe", str(epochs)])
#                             extensions.append(temp_extension)
#         else:
#             extensions.append(extension)
#
#     return extensions


def _build_submit_file(args, base_path):
    print('///// In build submit /////')
    sub_file_name = "./{}.sub".format(args.data) if args.fairness_type is None else "./{}_{}.sub".format(args.data,
                                                                                                         args.fairness_type)
    print("## Started building {} ##".format(sub_file_name))

    with open(sub_file_name, "w") as file:
        file.write("# ----------------------------------------------------------------------- #\n")
        file.write("# RUNTIME LIMITATION                                                      #\n")
        file.write("# ----------------------------------------------------------------------- #\n\n")
        file.write("# Maximum expected execution time for the job, in seconds\n")
        file.write("# 43200 = 12h\n")
        file.write("# 86400 = 24h\n")
        file.write("MaxTime = 43200\n\n")
        file.write("# Kill the jobs without warning\n")
        file.write("periodic_remove = (JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))\n\n")
        file.write("# ----------------------------------------------------------------------- #\n")
        file.write("# RESSOURCE SELECTION                                                     #\n")
        file.write("# ----------------------------------------------------------------------- #\n\n")
        file.write("request_memory = {}\n".format(args.ram))
        file.write("request_cpus = {}\n\n".format(args.cpu))
        file.write("# ----------------------------------------------------------------------- #\n")
        file.write("# FOLDER SELECTION                                                        #\n")
        file.write("# ----------------------------------------------------------------------- #\n\n")
        file.write("environment = \"PYTHONUNBUFFERED=TRUE\"\n")
        file.write("executable = /home/mrateike/miniconda3/envs/fairlearn_original/bin/python\n\n")
        file.write("error = {}/error/experiment.$(Process).err\n".format(base_path))
        file.write("output = {}/output/experiment.$(Process).out\n".format(base_path))
        file.write("log = {}/log/experiment.$(Process).log\n".format(base_path))
        file.write("# ----------------------------------------------------------------------- #\n")
        file.write("# QUEUE                                                                   #\n")
        file.write("# ----------------------------------------------------------------------- #\n\n")

        for time_steps_1 in args.time_steps_1:
            for time_steps_2 in args.time_steps_2:
                for nu in args.nu:
                    for eps in args.eps:
                            command = "run.py " \
                                      "-T1 {} " \
                                      "-T2 {} " \
                                      "-TT{} " \
                                      "-f{} " \
                                      "-bt {}" \
                                      "-eps {} " \
                                      "-nu {} " \
                                      "-p {}" \
                                      "{} " \
                                      "{}".format(time_steps_1,
                                                  time_steps_2,
                                                  args.time_steps_testing,
                                                  args.fairness_type,
                                                  args.batch_type,
                                                  eps,
                                                  nu,
                                                  args.data,
                                                  "{}/results ".format(base_path),
                                                  "--plot " if args.plot else "",
                                                  "-pid $(Process)" if args.queue_num else "")

                            # if args.fairness_type is not None:
                            #     for extension in _fairness_extensions(args, lambdas, build=True):
                            #         file.write("arguments = {} {}\n".format(command, extension))
                            #         file.write("queue {}\n".format(args.queue_num
                            #                                        if args.queue_num is not None else ""))
                            # else:

                            file.write("arguments = {}\n".format(command))
                            file.write("queue {}\n".format(args.queue_num
                                                               if args.queue_num is not None else ""))

    print("## Finished building {} ##".format(sub_file_name))


def _multi_run(args, base_path):
    # this is called when executed
    for time_steps_1 in args.time_steps_1:
            for time_steps_2 in args.time_steps_2:
                for nu in args.nu:
                    for eps in args.eps:
                        command = ["python3", "run.py",
                                   "-t1", str(time_steps_1),
                                   "-t2", str(time_steps_2),
                                   "-tt", str(args.time_steps_testing),
                                   "-f", str(args.fairness_type),
                                   "-bt", str(args.batch_type),
                                   "-e", str(eps),
                                   "-nu", str(nu),
                                   "-d", str(args.data),
                                   "-p", str(base_path)
                                   ]
                        if args.plot:
                            command.append("--plot")

                        # if args.fairness_type is not None:
                        #     for extension in _fairness_extensions(args, lambdas, build=False):
                        #         temp_command = deepcopy(command)
                        #         temp_command.extend(extension)
                        #         subprocess.run(temp_command)
                        # else:
                        print('before subprocess.run - command', command)

                        subprocess.run(command)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # Policy training parameters
    parser.add_argument('-T1', '--time_steps_1', type=int, nargs='+', required=True, help='list of phase 1 time steps to be used')
    parser.add_argument('-T2', '--time_steps_2', type=int, nargs='+', required=True,
                        help='list of phase 2 time steps to be used')
    parser.add_argument('-TT', '--time_steps_testing', type=int, required=False,
                        help='testing time steps to be used', default=10000)
    # parser.add_argument('-bs', '--batch_sizes', type=int, nargs='+', required=True,
    #                     help='list of batch sizes to be used')

    # Fairness parameters
    parser.add_argument('-f', '--fairness_type', type=str, required=False,
                        help="select the type of fairness (DP, FPR)"
                             "if none is selected no fairness criterion is applied")
    parser.add_argument('-bt', '--batch_type', type=str, required=True,
                        help='batches type used (exp, lin)')
    parser.add_argument('-epspython3 /Users/mrateike/PycharmProjects/fair_minimonster/exp/multi_run.py', '--eps', type=float, nargs='+', required=True,
                        help="list of statistical unfairness paramenters to be used")
    parser.add_argument('-nu', '--nu', type=float, nargs='+', required=True,
                        help="list of accuracy parameters of the oracle to be used")

    # Configuration parameters
    parser.add_argument('-d', '--data', type=str, required=True,
                        help="select the distribution (FICO, COMPAS, ADULT, GERMAN, Uncalibrated)")
    parser.add_argument('-p', '--path', type=str, required=False, help="save path for the results")

    parser.add_argument('--plot', required=False, action='store_true')



    # Build script parameters
    parser.add_argument('--build_submit', required=False, action='store_true')
    parser.add_argument('-pp', '--python_path', type=str, required=False, help="path of the python executable")
    parser.add_argument('-q', '--queue_num', type=int, required=False,
                        help="the number of process that should be queued")
    parser.add_argument('--ram', type=int, required=False, help='the RAM requested (default = 6144)', default=6144)
    parser.add_argument('--cpu', type=int, required=False, help='the number of CPUs requested (default = 1)', default=1)

    args = parser.parse_args()



    if args.build_submit and args.python_path is None:
        parser.error('when using --build_submit, --python_path has to be specified')

   # optional: define parser errors

    base_path = "{}/{}".format(args.path, args.data)

    print('args.build_submit', args.build_submit)

    if args.build_submit:
        print('args', args)
        print('base_path', base_path)
        _build_submit_file(args, base_path)
    else:
        _multi_run(args, base_path)