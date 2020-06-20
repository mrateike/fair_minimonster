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




def _build_submit_file(args, base_path):

    bandit_path = "{}/Bandit".format(base_path)
    Path(bandit_path).mkdir(parents=True, exist_ok=True)

    for bt in args.batch_type:

        batch_path = "{}/{}".format(bandit_path, bt)
        Path(batch_path).mkdir(parents=True, exist_ok=True)

        for d in args.data:
            data_path = "{}/{}".format(batch_path, d)
            Path(data_path).mkdir(parents=True, exist_ok=True)

            for f in args.fairness_type:
                fair_path = "{}/{}".format(data_path, f)
                Path(fair_path).mkdir(parents=True, exist_ok=True)

                timestamp = time.gmtime()
                ts_folder = time.strftime("%Y-%m-%d-%H-%M-%S", timestamp)
                ex_folder = 'Bandit'
                experiment_path = "{}/{}_{}_{}_{}_{}".format(fair_path, ts_folder, ex_folder, bt, d, f)
                Path(experiment_path).mkdir(parents=True, exist_ok=True)

                err_path = "{}/error".format(experiment_path)
                Path(err_path).mkdir(parents=True, exist_ok=True)
                log_path = "{}/log".format(experiment_path)
                Path(log_path).mkdir(parents=True, exist_ok=True)
                output_path = "{}/output".format(experiment_path)
                Path(output_path).mkdir(parents=True, exist_ok=True)

                print('///// In build submit /////')
                sub_file_name = "./Bandit_{}_{}_{}.sub".format(bt, d, f)
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
                    file.write("error = {}/error/experiment.$(Process).err\n".format(experiment_path))
                    file.write("output = {}/output/experiment.$(Process).out\n".format(experiment_path))
                    file.write("log = {}/log/experiment.$(Process).log\n".format(experiment_path))
                    file.write("# ----------------------------------------------------------------------- #\n")
                    file.write("# QUEUE                                                                   #\n")
                    file.write("# ----------------------------------------------------------------------- #\n\n")

                    # for bs in args.batch_size:
                    #     if bs == '1':
                    #         batch_size_path = experiment_path
                    #     else:
                    for bs in args.batch_size:
                        batch_size_path = "{}/batch_{}".format(experiment_path, bs)
                        Path(batch_size_path).mkdir(parents=True, exist_ok=True)

                        for a in args.alpha:
                            alpha_path = "{}/alpha_{}".format(batch_size_path, a)
                            Path(alpha_path).mkdir(parents=True, exist_ok=True)

                            for s in args.seeds:
                                seed_path = "{}/seed_{}".format(alpha_path, s)
                                Path(seed_path).mkdir(parents=True, exist_ok=True)

                                for T in args.total_data:
                                    for eps in args.eps:
                                        for mu in args.mu:
                                            for nu in args.nu:
                                                for i in args.iterations:
                                                    command = "exp/run.py " \
                                                              "-T {} " \
                                                              "-a {} " \
                                                              "-s {} " \
                                                              "-f {} " \
                                                              "-bt {} " \
                                                              "-bs {} " \
                                                              "-eps {} " \
                                                              "-nu {} " \
                                                              "-mu {} " \
                                                              "-d {} " \
                                                              "-i {} " \
                                                              "-p {} " \
                                                              "{} ".format(T,
                                                                          a,
                                                                          s,
                                                                          f,
                                                                          bt,
                                                                          bs,
                                                                          eps,
                                                                          nu,
                                                                          mu,
                                                                          d,
                                                                          i,
                                                                          seed_path,
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


    bandit_path = "{}/Bandit".format(base_path)
    Path(bandit_path).mkdir(parents=True, exist_ok=True)

    for bt in args.batch_type:
        batch_path = "{}/{}".format(bandit_path, bt)
        Path(batch_path).mkdir(parents=True, exist_ok=True)

        for d in args.data:
            # print('data', d)
            data_path = "{}/{}".format(batch_path, d)
            # print('data_path', data_path)
            Path(data_path).mkdir(parents=True, exist_ok=True)

            for f in args.fairness_type:
                fair_path = "{}/{}".format(data_path, f)
                Path(fair_path).mkdir(parents=True, exist_ok=True)

                timestamp = time.gmtime()
                ts_folder = time.strftime("%Y-%m-%d-%H-%M-%S", timestamp)
                ex_folder = 'Bandit'
                experiment_path = "{}/{}_{}_{}_{}_{}".format(fair_path, ts_folder, ex_folder, bt, d, f)
                print('experiment_path', experiment_path)
                Path(experiment_path).mkdir(parents=True, exist_ok=True)

                # err_path = "{}/error".format(experiment_path)
                # Path(err_path).mkdir(parents=True, exist_ok=True)
                # log_path = "{}/log".format(experiment_path)
                # Path(log_path).mkdir(parents=True, exist_ok=True)
                # output_path = "{}/output".format(experiment_path)
                # Path(output_path).mkdir(parents=True, exist_ok=True)

                for bs in args.batch_size:
                    # if bs == '1':
                    #     batch_size_path = experiment_path
                    # else:
                    batch_size_path = "{}/batch_{}".format(experiment_path, bs)
                    Path(batch_size_path).mkdir(parents=True, exist_ok=True)

                    for a in args.alpha:
                        alpha_path = "{}/alpha_{}".format(batch_size_path, a)
                        Path(alpha_path).mkdir(parents=True, exist_ok=True)

                        for s in args.seeds:
                            seed_path = "{}/seed_{}".format(alpha_path, s)
                            Path(seed_path).mkdir(parents=True, exist_ok=True)

                            for T in args.total_data:
                                for eps in args.eps:
                                    for mu in args.mu:
                                        for nu in args.nu:
                                            for i in args.iterations:

                                                command = ["python3", "exp/run.py",
                                                           "-T", str(T),
                                                           "-a", str(a),
                                                           "-s", str(s),
                                                           "-f", str(f),
                                                           "-bt", str(bt),
                                                           "-bs", str(bs),
                                                           "-eps", str(eps),
                                                           "-nu", str(nu),
                                                           "-mu", str(mu),
                                                           "-d", str(d),
                                                           "-i", str(i),
                                                           "-p", str(seed_path)
                                                           ]

                                                print('before subprocess.run - command', command)

                                                subprocess.run(command)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # Policy training parameters
    parser.add_argument('-T', '--total_data', nargs='+', type=int, required=True,
                        help='list of total data s to be used')
    parser.add_argument('-a', '--alpha', type=float, nargs='+', required=True,
                        help='phase 1 phase 2 data split parameter')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', required=False,
                        help='seeds for phase 1, 2, testing', default=967)

    parser.add_argument('-f', '--fairness_type', type=str, nargs='+', required=True,
                        help="select the type of fairness (DP, EO)")
    parser.add_argument('-bt', '--batch_type', type=str, nargs='+', required=True,
                        help='batches type used (no_batch, exp, lin, warm_start)')
    parser.add_argument('-bs', '--batch_size', type=str, nargs='+', required=False,
                        help='batches size used for lin (required) otherwise ignored')

    parser.add_argument('-eps', '--eps', type=float, nargs='+', required=True,
                        help="list of statistical unfairness paramenters (beta) to be used")
    parser.add_argument('-nu', '--nu', type=float, nargs='+', required=True,
                        help="list of accuracy parameters of the oracle to be used")
    parser.add_argument('-mu', '--mu', type=float, nargs='+', required=True,
                        help="minimum probability for simulating the bandit")

    # Configuration parameters
    parser.add_argument('-d', '--data', type=str, nargs='+', required=True,
                        help="select the distribution (FICO, Uncalibrated)")
    parser.add_argument('-i', '--iterations', type=str, nargs='+', required=True,
                        help="number of iterations of the bandit coordinate decent algo")
    parser.add_argument('-p', '--path', type=str, required=False, help="save path for the results")


    # Build script parameters
    parser.add_argument('--build_submit', required=False, action='store_true')
    # parser.add_argument('-pp', '--python_path', type=str, required=False, help="path of the python executable")
    parser.add_argument('-q', '--queue_num', type=int, required=False,
                        help="the number of process that should be queued")
    parser.add_argument('--ram', type=int, required=False, help='the RAM requested (default = 6144)', default=6144)
    parser.add_argument('--cpu', type=int, required=False, help='the number of CPUs requested (default = 1)', default=1)

    args = parser.parse_args()

    if args.build_submit:
        _build_submit_file(args, args.path)
    else:
        _multi_run(args, args.path)