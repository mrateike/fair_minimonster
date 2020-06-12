import argparse
import subprocess
from copy import deepcopy
import numpy as np
import os
import sys
from pathlib import Path
import time

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

    data_folder = "Uncalibrated"
    base_path = "{}/{}".format(base_path, data_folder)
    Path(base_path).mkdir(parents=True, exist_ok=True)

    for fair in args.fairness_type:

        fair_folder = fair
        base_save_path = "{}/{}".format(base_path, fair_folder)
        Path(base_save_path).mkdir(parents=True, exist_ok=True)

        timestamp = time.gmtime()
        ts_folder = time.strftime("%Y-%m-%d-%H-%M-%S", timestamp)
        ex_folder = fair
        base_save_path = "{}/{}_Oracle_Uncalibrated_{}".format(base_save_path, ts_folder, ex_folder)
        Path(base_save_path).mkdir(parents=True, exist_ok=True)

        err_path = "{}/error".format(base_save_path)
        Path(err_path).mkdir(parents=True, exist_ok=True)
        log_path = "{}/log".format(base_save_path)
        Path(log_path).mkdir(parents=True, exist_ok=True)
        output_path = "{}/output".format(base_save_path)
        Path(output_path).mkdir(parents=True, exist_ok=True)


        print('///// In build submit /////')
        sub_file_name = "./{}.sub".format('Uncalibrated') if fair is None else "./{}_{}.sub".format('Uncalibrated',
                                                                                                             fair)
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
            file.write("error = {}/error/experiment.$(Process).err\n".format(base_save_path))
            file.write("output = {}/output/experiment.$(Process).out\n".format(base_save_path))
            file.write("log = {}/log/experiment.$(Process).log\n".format(base_save_path))
            file.write("# ----------------------------------------------------------------------- #\n")
            file.write("# QUEUE                                                                   #\n")
            file.write("# ----------------------------------------------------------------------- #\n\n")

            print('eps', args.eps)

            for N in args.total_data:
                for a in args.alpha:
                    for s in args.seeds:
                        for mu in args.mu:
                            for nu in args.nu:
                                for eps in args.eps:
                                        command = "exp/main_expg.py " \
                                                      "-N {} " \
                                                      "-a {} " \
                                                      "-s {} " \
                                                      "-i {} " \
                                                      "-f {} " \
                                                      "-eps {} " \
                                                      "-nu {} " \
                                                      "-mu {} " \
                                                      "-p {} " \
                                                      "{} ".format(N,
                                                                  a,
                                                                  s,
                                                                  args.iterations,
                                                                  fair,
                                                                  eps,
                                                                  nu,
                                                                  mu,
                                                                  base_path,
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
    data_folder = "Uncalibrated"
    base_path = "{}/{}".format(base_path, data_folder)
    Path(base_path).mkdir(parents=True, exist_ok=True)

    for fair in args.fairness_type:

        fair_folder = fair
        base_save_path = "{}/{}".format(base_path, fair_folder)
        Path(base_save_path).mkdir(parents=True, exist_ok=True)

        timestamp = time.gmtime()
        ts_folder = time.strftime("%Y-%m-%d-%H-%M-%S", timestamp)
        ex_folder = fair
        base_save_path = "{}/{}_Oracle_Uncalibrated_{}".format(base_save_path, ts_folder, ex_folder)
        Path(base_save_path).mkdir(parents=True, exist_ok=True)

        err_path = "{}/error".format(base_save_path)
        Path(err_path).mkdir(parents=True, exist_ok=True)
        log_path = "{}/log".format(base_save_path)
        Path(log_path).mkdir(parents=True, exist_ok=True)
        output_path = "{}/output".format(base_save_path)
        Path(output_path).mkdir(parents=True, exist_ok=True)

        for N in args.total_data:
            for a in args.alpha:
                for s in args.seeds:
                    for mu in args.mu:
                            for nu in args.nu:
                                for eps in args.eps:
                                    command = ["python3", "exp/main_expg.py",
                                               "-N", str(N),
                                               "-a", str(a),
                                               "-s", str(s),
                                               "-i", str(args.iterations),
                                               "-f", str(fair),
                                               "-eps", str(eps),
                                               "-nu", str(nu),
                                               "-mu", str(mu),
                                               "-p", str(base_save_path)
                                               ]

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
    parser.add_argument('-N', '--total_data', type=int, nargs='+', required=True,
                        help='list of toatl data s to be used')
    parser.add_argument('-a', '--alpha', type=float, nargs='+', required=True,
                        help='phase 1 phase 2 data split parameter')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', required=False,
                        help='seeds for phase 1, 2, testing', default=967)
    parser.add_argument('-i', '--iterations', type=int, required=False,
                        help='how many policies should be computed per setting', default=100)
    parser.add_argument('-f', '--fairness_type', type=str, nargs='+', required=True,
                        help="select the type of fairness (DP, EO)")
    parser.add_argument('-eps', '--eps', type=float, nargs='+', required=True,
                        help="list of statistical unfairness paramenters (beta) to be used")
    parser.add_argument('-nu', '--nu', type=float, nargs='+', required=True,
                        help="list of accuracy parameters of the oracle to be used")
    parser.add_argument('-mu', '--mu', type=float, nargs='+', required=True,
                        help="minimum probability for simulating the bandit")
    parser.add_argument('-p', '--path', type=str, required=True, help="save path for the results")

    # Build script parameter
    parser.add_argument('--build_submit', required=False, action='store_true')
    # parser.add_argument('-pp', '--python_path', type=str, required=False, help="path of the python executable")

    parser.add_argument('-q', '--queue_num', type=int, required=False,
                        help="the number of process that should be queued")
    parser.add_argument('--ram', type=int, required=False, help='the RAM requested (default = 6144)', default=6144)
    parser.add_argument('--cpu', type=int, required=False, help='the number of CPUs requested (default = 1)', default=1)

    args = parser.parse_args()


    if args.build_submit is None:
        parser.error('when using --build_submit, --python_path has to be specified')



    if args.build_submit:
        _build_submit_file(args, args.path)
    else:
        _multi_run(args, args.path)