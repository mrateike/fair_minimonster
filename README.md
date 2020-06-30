This repository contains an implementation of the paper Bechavod, Y., Ligett, K., Roth, A., Waggoner, B., & Wu, S. Z. (2019).
Equal opportunity in online classification with partial feedback. https://arxiv.org/pdf/1902.02242.pdf
The repository is self-contained, i.e. it includes all dependencies, code, and datasets needed to run the experiments of the paper.

# How to run the code

A Conda 'environment.yml' file is provided with all the dependencies needed to run the code.
In order to install all the dependencies (assuming that Conda is already installed), 
run 

```
conda env create -f environment.yml
```

A new environment called fair_minimonster will appeared, to activate it execute 

```
conda activate fair_minimonster
```

To run the algorihtm command line interface is provided
and help is provided through

```
python main.py --help
```

As an example, if we wish to run the algorithm on the FICO dataset with a constraint demographic
parity (DP) and a fairness relaxation eps=0.1, with a total number of T=1032 data and a splitting parameter
alpha = 0.25, such that we obtain a phase 1 dataset of size T1 = 32 and a
 phase 2 dataset of size T2 = 1000, with a linear batch size of 10 and a maximum number of 10 iterations of the coordinate
descent algorithm, and a minimum probabilit for the smoothed distribution mu-0.1 and an accuracy
of the fair oracle of 0.01 and save results in a folder under the path /results, we call

```
python main.py -T 1032 -a 0.4 -s 1 -bt lin -bs 10 -i 3 -f DP -beta 0.1 -nu 1e-6 -mu 0.1 -d FICO -p /results
```

Here is the output of the argument ''--help':

```
  -h, --help            show this help message and exit
  -T TOTAL_DATA [TOTAL_DATA ...], --total_data TOTAL_DATA [TOTAL_DATA ...]
                        total amount of data T to be used (phase 1 and 2)
  -a ALPHA [ALPHA ...], --alpha ALPHA [ALPHA ...]
                        phase 1 phase 2 data split parameter, value between
                        0.25 and 0.5
  -s SEEDS [SEEDS ...], --seeds SEEDS [SEEDS ...]
                        number to fix seeds for phase 1, 2, testing
  -f FAIRNESS_TYPE [FAIRNESS_TYPE ...], --fairness_type FAIRNESS_TYPE [FAIRNESS_TYPE ...]
                        type of fairness (DP, EO)
  -bt BATCH_TYPE [BATCH_TYPE ...], --batch_type BATCH_TYPE [BATCH_TYPE ...]
                        batches type used (no_batch, exp, lin)
  -bs BATCH_SIZE [BATCH_SIZE ...], --batch_size BATCH_SIZE [BATCH_SIZE ...]
                        batch size for lin otherwise set to 1
  -beta BETA [BETA ...], --beta BETA [BETA ...]
                        fairness relaxation parameter (unfairness) paramenter
                        beta
  -nu NU [NU ...], --nu NU [NU ...]
                        accuracy parameter of the fair oracle
  -mu MU [MU ...], --mu MU [MU ...]
                        minimum probability for smoothening distribution Q
  -d DATA [DATA ...], --data DATA [DATA ...]
                        distribution (FICO, Uncalibrated)
  -i ITERATIONS [ITERATIONS ...], --iterations ITERATIONS [ITERATIONS ...]
                        number of iterations of the bandit coordinate decent
                        algorithm
  -p PATH, --path PATH  save path for the results

```
