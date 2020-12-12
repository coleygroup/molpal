# MolPAL: Molecular Pool-based Active Learning
# Efficient Exploration of Virtual Chemical <br/> Libraries through Active Learning

![overview of molpal structure and implementation](molpal_overview.png)

## Overview
This repository contains the source of MolPAL, both a library and software for the accelerated discovery of compounds in high-throughput virtual screening environments.

## Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Requirements](#requirements)
- [Installation](#installation)
- [Object Model](#object-model)
- [Preprocessing](#preprocessing)
- [Running MolPAL](#running-molpal)
  * [Required Settings](#required-settings)
  * [Optional Settings](#optional-settings)
- [Future Directions](#future-directions)
- [Reproducing Experimental Results](#reproducing-experimental-results)

## Requirements
- Python (>= 3.6)

NN and MPN models can make use of GPUs for significantly faster model training and inference. To utilize this, you must have the following:
- CUDA (>= 8.0)
- cuDNN

## Installation
The first step in installing MolPAL is to clone this repository: `git clone <this_repo>`

The easiest way to install all dependencies is to use conda along with the supplied [environment.yml](environment.yml) file, but you may also install them manually, if desired. All libraries listed in that file are __required__ before using `MolPAL`

The following packages are _optional_ to install before running MolPAL:
- cudatoolkit (whichever version matches your CUDA build if utilizing GPU acceleration for PyTorch-based models (MPN); _note_: you must reinstall PyTorch according to the GPU installation instructions on the website if using the `environment.yml` approach)
- [map4](https://github.com/reymond-group/map4) and [tmap](https://github.com/reymond-group/tmap) (if utilizing the map4 fingerprint)
- [optuna](https://optuna.readthedocs.io/en/stable/installation.html) (if planning to perform hyperparameter optimization)

#### setup via conda 
0. (if necessary) [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
1. `cd /path/to/molpal`
1. `conda env create -f environment.yml`

Before running MolPAL, be sure to first activate the environment: `conda activate molpal`

## Object Model
MolPAL is a software for batched, Bayesian optimization in a virtual screening environment. At the core of this software is the `molpal` library, which implements defines several classes that implement various elements of the optimization routine.

__Explorer__: An [`Explorer`](molpal/explorer.py) is the abstraction of the optimization routine. It ties together the `MoleculePool`, `Acquirer`, `Encoder`, `Model`, and `Objective`, which each handle (roughly) a single step of a Bayesian optimization loop, into a full optimization procedure. Its main functionality is defined by the `run()` method, which performs the optimization until a stopping condition is met, but it also defines other convenience functions that make it amenable to running a single iteration of the optimization loop and interrogating its current state if optimization is desired to be run interactively.

__MoleculePool__: A [`MoleculePool`](molpal/pools/base.py) defines the virtual library (i.e., domain of inputs)

__Acquirer__: An [`Acquirer`](molpal/acquirer/acquirer.py) handles acquisition of unlabeled inputs from the MoleculePool according to its `metric` and the prior distribution over the data. The [`metric`](molpal/acquirer/metrics.py) is a function that takes an input array of predictions and returns an array of equal dimension containing acquisition utilities.

__Encoder__: An [`Encoder`](molpal/encoder.py) computes the uncompressed feature representation of an input based on its identifier for use with clustering and models that expect vectors as inputs.

__Model__: A [`Model`](molpal/model/base.py) is trained on labeled data to produce a posterior distribution that guides the sequential round of acquisition

__Objective__: An [`Objective`](molpal/objectives/base.py) handles calculation of the objective function for unlabeled inputs

## Preprocessing

For model expecting vectors as inputs (e.g., random forest and feed-forward neural network models,) molecular fingerprints must be calculated first. Given that the set of fingerprints used for inference is the same each time, it makes sense to cache these fingerprints and that's exactly what the base `MoleculePool` does (also referred to as an `EagerMoleculePool`.) However, the complete set of fingerprints for most libraries would be too large to cache entirely in memory on most systems, so we instead store them in an HDF5 file that is transparently prepared for the user during MolPAL startup (if not already provided with the `--fps` option.) If you wish to prepare this file ahead of time, you can use [`scripts/fingerprints.py`](scripts/fingerprints.py) to do just this. __Note__: if MolPAL prepares the file for you, it prints a message saying where the file was written to (usually under the $TMP directory) and whether there were invalid SMILES. To reuse this fingerprints file, simply move this file to a persistent directory after MolPAL has completed its run. Additionally, if there were __no__ invalid smiles, you can pass the `--validated` flag in the options to further speed up MolPAL startup.

To prepare the fingerprints file corresopnding to the sample command below, issue the following command: `python scripts/fingerprints.py --library libraries/Enamine50k.csv.gz --fingerprint pair --length 2048 --radius 2 --name fps_enamine50k.h5`

The resulting fingerprint file will be located in your current working directory as `fps_enamine50k.h5`. To use this in the sample command below, add `--fps fps_enamine50k.h5` to the argument list.

## Running MolPAL

### Examples
The general command to run MolPAL is as follows:

`python molpal.py -o <objective_type> [additional objective arguments] --libary <path/to/library.csv[.gz]> [additional library arguments] [additional model/encoding/acquistion/stopping/logging arguments]`

Alternatively, you may use a configuration file to run MolPAL, like so:

`python molpal.py --config <path/to/config_file>`

Two sample configuration files are provided: [minimal_config.ini](config/minimal_config.ini), a configuration file specifying only the necessary arguments to run MolPAL, and [sample_config.ini](config/sample_config.ini), a configuration file containing a few common options to specify (but not _all_ possible options.)

Configuration files accept the following syntaxes:
- `--arg value` (argparse)
- `arg: value` (YAML)
- `arg = value` (INI)
- `arg value`

A sample command to run one of the experiments used to generate data in the initial publication is as follows:

`python run.py --config config_expts/Enamine50k_retrain.ini --name molpal_50k --metric greedy --init-size 0.01 --batch-size 0.01 --model rf`

or the full command:

`python run.py --name molpal_50k --write-intermediate --write-final --retrain-from-scratch --library libraries/Enamine50k.csv.gz --validated --metric greedy --init-size 0.01 --batch-size 0.01 --model rf --fingerprint pair --length 2048 --radius 2 --objective lookup --lookup-path data/4UNN_Enamine50k_scores.csv.gz --lookup-smiles-col 1 --lookup-data-col 2 --minimize --top-k 0.01 --window-size 10 --delta 0.01 --max-epochs 5`

### Required Settings
The primary purpose of MolPAL is to accelerate virtual screens in a prospective manner. Currently (December 2020), MolPAL supports computational docking screens using the [`pyscreener`](https://github.com/coleygroup/pyscreener) library

`-o` or `--objective`: The objective function you would like to use. Choices include `docking` for docking objectives and `lookup` for lookup objectives. There are additional arguments for each type of objective.
- `docking`: given the variety of screening options allowed by the `pyscreener` library, it's likely easiest to specify an `--objective-config` rather than providing these options on the command line. The `objective-config` file must be provided in the format of a `pyscreener` configuration file, so some options might have different names (e.g., `size` in that file rather than `--box-size`). Any options specified on the command line will override any options provided in the configuration file. 
  * `--software`: the docking software you would like to use. Choices: 'vina', 'smina', 'psovina', 'qvina', and 'ucsfdock' (Default = 'vina').
  * `--receptor`': the filepath of the receptor you are attempting to dock ligands into.
  * `--box-center`: the x-, y-, and z-coordinates (Å) of the center of the docking box.
  * `--box-size`: the x-, y-, and z- radii of the docking box in Å.
  * `--docked-ligand-file`: the name of a file containing the coordinates of a docked/bound ligand. If using Vina-type software, this file must be a PDB format file. Either `--box-center` and `--box-size` must be specified or a docked ligand file must be provided. In the case that both are provided, 
  * `--score-mode`: the method by which to calculate an overall score from multiple scored conformations
- `lookup`
  * `--lookup-path`: the filepath of a CSV file containing score information for each input

`--library`: the filepath of a CSV file containing the virtual library as SMILES strings
- (optional) `--fps`: the filepath of an hdf5 file containing the precomputed fingerprints of your virtual library. MolPAL relies on the assumption that the ordering of the fingerprints in this file is exactly the same as that of the library file and that the encoder used to generate these fingerprints is exactly the same as the one used for model training. MolPAL handles writing this file for you if unspecified, so this option is mostly useful for avoiding the overhead at startup of running MolPAL again with the same library/encoder settings.

### Optional Settings
MolPAL has a number of different model architectures, encodings, acquisition metrics, and stopping criteria to choose from. Many of these choices have default settings that were arrived at through hyperparameter optimization, but your circumstances may call for modifying these choices. To see the full list, run MolPAL with either the `-h` or `--help` flags. A few common options to specify are shown below.

`-k`: the number of top scores to evaluate when calculating an average. This number may be either a float representing a fraction of the library or an integer to specify precisely how many top scores to average. (Default = 0.005)

`--window-size` and `--delta`: the principal stopping criterion of MolPAL is whether or not the current top-k average score is better than the moving average of the `window_size` most recent top-k average scores by at least `delta`. (Default: `window_size` = 3, `delta` = 0.1)

`--max-explore`: if you would like to limit MolPAL to exploring a fixed fraction of the libary or number of inputs, you can specify that by setting this value. (Default = 1.0)

`--max-epochs`': Alternatively, you may specify the maximum number of epochs of exploration. (Default = 50)

`--model`: the type of model to use. Choices include `rf`, `gp`, `nn`, and `mpn`. (Default = `rf`)  
  - `--conf-method`: the confidence estimation method to use for the NN or MPN models. Choices include `ensemble`, `dropout`, `mve`, and `none`. (Default = 'none'). NOTE: the MPN model does not support ensembling

`--metric`: the acquisition metric to use. Choices include `random`, `greedy`, `ucb`, `pi`, `ei`, `thompson`, and `threshold` (Default = `greedy`.) Some metrics include additional settings (e.g. the β value for `ucb`.) 

## Hyperparameter Optimization
While the default settings of MolPAL were chosen based on hyperparameter optimization with Optuna, they were calculated based on the context of structure-based discovery our computational resources. It is possible that these settings are not optimal for your particular problem. To adapt MolPAL to new circumstances, we recommend first generating a dataset that is representative of your particular problem then peforming hyperparameter optimization of your own using the `LookupObjective` class. This class acts as an Oracle for your particular objective function, enabling both consistent and near-instant calculation of the objective function for a particular input, saving time during hyperparameter optimization.

## Future Directions
Though MolPAL was originally intended for use with protein-ligand docking screens, it was designed with modularity in mind and is easily extendable to other settings as well. In principal, all that is required to adapt MolPAL to a new problem is to write a custom `Objective` subclass that implements the `calc` method. This method takes a sequence SMILES strings as an input and returns a mapping from SMILES string -> objective function value to be utilized by the Explorer. _To this end, we are currently exploring the extension of MolPAL to subsequent stages of virtual discovery (MD, DFT, etc.)_ If you make use of the MolPAL library by implementing a new `Objective` subclass, we would be happy to include your work in the main branch.

## Reproducing Experimental Results
### Generating data
The data used in the original publication was generated through usage of the [`scripts/submit_molpal.py`](scripts/submit_molpal.py) script along with the corresponding configuration file located in `config_experiments` and the library name (e.g., '10k', '50k', 'HTS', or 'AmpC') as the two command line arguments. The submission script was designed to be used with a SLURM scheduler, but if you want to rerun the experiemnts on your machine, then you can simply follow the submission script logic to generate the proper command line arguments or write a new configuration file. The AmpC data was too large to include in this repo, but it may be downloaded from [here](https://figshare.com/articles/AmpC_screen_table_csv_gz/7359626).

### Analyzing data
Once all of the data were generated, the directories were containing the data from each run organized according to the following structure:
```
<library>
├── online
│   ├── <batch_size>
|   |   ├── <library>_<model>_<metric>_<batch_size>_<repeat_number>_[extra]
│   │   └── ...
│   ├── <batch_size>
|   |   ├── <library>_<model>_<metric>_<batch_size>_<repeat_number>_[extra]
│   │   └── ...
|   └── ...
└── retrain
    ├── <batch_size>
    |   ├── <library>_<model>_<metric>_<batch_size>_<repeat_number>_[extra]
    │   └── ...
    ├── <batch_size>
    |   ├── <library>_<model>_<metric>_<batch_size>_<repeat_number>_[extra]
    │   └── ...
    └── ...
```
where everything between angled brackets is a single word that describes the corresponding parameter (e.g., `<model>` = `mpn`.)

After the data was organized as above, `scripts/analyze_data.py` was used to produce the data that is included `scripts/figures.py`. Though it isn't necessary, if you wish to call `analyze_data.py` yourself, use the following command: `python scripts/analyze_data.py <full_score_dict.pkl> <parent_score_dir> <k>`, where `<full_score_dict.pkl>` is a pickled python dictionary generated by `scripts/make_dict.py` (just a dictionary of the included score CSV files), `<parent_score_dir>` would be `<library/online/batch_size>` from the directory structure above and `<k>` is the number of top-k results to analyze (100, 500, 1000, 50000 for the 10k, 50k, HTS, and AmpC libraries, respectively.)

The results from the corresponding commands are in `scripts/molpal_analysis.ipynb`, which also contains directions on how to generate many of the figures used in the main text. For the remaining figures (e.g., UMAP and histogram figures,) use the corresponding scripts in the `scripts` directory. To figure out how to run them, use the following command `python <script>.py --help`.
