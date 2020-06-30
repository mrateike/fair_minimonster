This repository contains an implementation of the paper Bechavod, Y., Ligett, K., Roth, A., Waggoner, B., & Wu, S. Z. (2019).
Equal opportunity in online classification with partial feedback. https://arxiv.org/pdf/1902.02242.pdf
The repository is self-contained, i.e. it includes all dependencies, code, and datasets needed to run the experiments of the paper.

# How to run the code

A Conda 'environment.yml' file is provided with all the dependencies needed to run the code.
In order to install all the dependencies (assuming that Conda is already installed), 
run 

```python
conda env create -f environment.yml
```

A new environment called fair_minimonster will appeared, to activate it execute 

```python
conda activate fair_minimonster
```

To run the algorihtm command line interface is provided
and help is provided through

```python
python main.py --help
```

As an example, if we wish to run the algorithm on the FICO dataset with a constraint demographic
parity (DP) and a fairness relaxation eps=0.1, with a total number of T=1032 data and a splitting parameter
alpha = 0.25, such that we obtain a phase 1 dataset of size T1 = 32 and a
 phase 2 dataset of size T2 = 1000, with a linear batch size of 10 and a maximum number of 10 iterations of the coordinate
descent algorithm, and a minimum probabilit for the smoothed distribution mu-0.1 and an accuracy
of the fair oracle of 0.01 and save results in a folder under the path /results, we call

```python
python main.py -T 1032 -a 0.4 -s 1 -bt lin -bs 10 -i 3 -f DP -eps 0.1 -nu 1e-6 -mu 0.1 -d FICO -p /results
```

Here is the output of the argument ''--help':

```python
  -h, --help            show this help message and exit
  -T TOTAL_DATA [TOTAL_DATA ...], --total_data TOTAL_DATA [TOTAL_DATA ...]
                        list of total data s to be used
  -a ALPHA [ALPHA ...], --alpha ALPHA [ALPHA ...]
                        phase 1 phase 2 data split parameter
  -s SEEDS [SEEDS ...], --seeds SEEDS [SEEDS ...]
                        seeds for phase 1, 2, testing
  -f FAIRNESS_TYPE [FAIRNESS_TYPE ...], --fairness_type FAIRNESS_TYPE [FAIRNESS_TYPE ...]
                        select the type of fairness (DP, EO)
  -bt BATCH_TYPE [BATCH_TYPE ...], --batch_type BATCH_TYPE [BATCH_TYPE ...]
                        batches type used (no_batch, exp, lin, warm_start)
  -bs BATCH_SIZE [BATCH_SIZE ...], --batch_size BATCH_SIZE [BATCH_SIZE ...]
                        batches size used for lin (required) otherwise 1
  -eps EPS [EPS ...], --eps EPS [EPS ...]
                        list of statistical unfairness paramenters (beta) to
                        be used
  -nu NU [NU ...], --nu NU [NU ...]
                        list of accuracy parameters of the oracle to be used
  -mu MU [MU ...], --mu MU [MU ...]
                        minimum probability for simulating the bandit
  -d DATA [DATA ...], --data DATA [DATA ...]
                        select the distribution (FICO, Uncalibrated)
  -i ITERATIONS [ITERATIONS ...], --iterations ITERATIONS [ITERATIONS ...]
                        number of iterations of the bandit coordinate decent
                        algo
  -p PATH, --path PATH  save path for the results

```
