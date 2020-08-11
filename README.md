# MolPAL: Molecular Pool-based Active Learning
# Efficient Exploration of Virtual Chemical <br/> Libraries through Active Learning 

## Overview
This repository contains the source of MolPAL, a framework for the accelerated discovery of compounds in high throughput virtual screening environments. Additionally, this repository contains pyscreener, a python program for conducting high throughput virtual screens of small molecules through docking simulations. It currently supports the most commonly used AutoDock Vina family of programs: Vina, Smina, PSOVina, QVina2)

At the core of MolPAL is the __Explorer__ class. The Explorer is an abstraction of a batched, Bayesian optimization routine on a discrete domain of inputs and it relies on five classes
1. __MoleculePool__: A `MoleculePool` defines the virtual library (i.e., domain of inputs)
2. __Acquirer__: An `Acquirer` handles acquisition of unlabeled inputs from the MoleculePool according to its acquisition metric (e.g. random, greedy, upper confidence bound, etc...) and the prior distribution over the data
3. __Encoder__: An `Encoder` computes the uncompressed feature representation of an input based on its identifier for use with clustering and models that expect vectors as inputs
4. __Model__: A `Model` is trained on labeled data to produce a posterior distribution that guides the sequential round of acquisition
5. __Objective__: An `Objective` handles calculation of the objective function for unlabeled inputs

## Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running MolPAL](#running-molpal)
  * [Novel Targets](#novel-targets)
  * [Hyperparameter Optimization](#hyperparameter-optimization)
- [Running pyscreener](#running-pyscreener)
- [Future Directions](#future-directions)

## Requirements
- Python (>= 3.6)
- all desired docking programs to be installed and located on a user's PATH
- (optional) OpenBabel to be installed and located on a user's path if input file preparation is to be performed online

NN and MPN models can make use of GPUs for significantly faster model regression and inference. To utilize this, you must have the following:
- CUDA (>= 8.0)
- cuDNN


## Installation
The first step in installing MolPAL is to clone this repository: `git clone <this_repo>`

The easiest way to install all dependencies necessary to run MolPAL is to use conda along with the supplied environment YAML file, but you may also install them manually, if desired.

### conda
1. (if necessary) install conda. Instructions will vary based on your system
1. `cd /path/to/molpal`
1. `conda env create -f environment.yml`

Before running MolPAL, be sure to first activate the environment: `conda activate molpal`

### manual
The following packages are __necessary__ to install before running MolPAL:
- [chemprop](https://github.com/chemprop/chemprop) (0.0.2)
- h5py (2.10)
- numpy (1.18)
- rdkit (2019.09.3)
- scikit-learn (0.22)
- scipy (1.4)
- pytorch (1.5; built with CUDA if utilizing GPU acceleration)
- tensorflow (2.2)
- tqdm (4.42)

The following packages are _optional_ to install before running MolPAL:
- cudatoolkit (if utilizing GPU acceleration; whichever version matches your CUDA build)
- cudnn (if utilizing GPU acceleration; whichever version matches your cuDNN build)
- [map4](https://github.com/reymond-group/map4) (if utilizing the map4 fingerprint)
- optuna (1.4; if planning to perform hyperparameter optimization)
- [tmap](https://github.com/reymond-group/tmap) (if utilizing the map4 fingerprint)


## Running MolPAL

### Novel Targets

The general command to run MolPAL is as follows:

`python molpal.py -o <lookup|docking> [additional objective arguments] --libary <path/to/library.csv> [additional library arguments] [additional model/encoding/acquistion/stopping/logging arguments]`

Alternatively, you may use a configuration file to run MolPAL, like so:

`python molpal.py --config <path/to/config_file>`

Two sample configuration files are provided: minimal_config.ini, a configuration file specifying only the necessary arguments to run MolPAL, and sample_config.ini, a configuration file containing a few common options to specify (but not _all_ possible options.)

Configuration files accept the following syntaxes:
- `--arg value` (argparse)
- `arg: value` (YAML)
- `arg = value` (INI)
- `arg value`

#### Required Settings
There a few required settings to specify before running MolPAL, and they are highlighted below along with their relevant required settings.

`-o` or `--objective`: The objective function you would like to use. Choices include `docking` for docking objectives and `lookup` for lookup objectives. There are additional arguments for each type of objective
- `docking`
  * `-d, --docker`: the docking software you would like to use. Choices: 'vina', 'smina', 'psovina', 'qvina.'
  * `-r, --receptor`': the filepath of the receptor you are attempting to dock ligands into.
  * `-c, --center`: the x-, y-, and z-coordinates (Å) of the center of the docking box.
  * `-s, --size`: the x-, y-, and z- radii of the docking box in Å.
- `lookup`
  * `--lookup-path`: the filepath of a CSV file containing score information for each input

`--library`: the filepath of a CSV file containing the virtual library as SMILES strings
- (optional) `--fps`: the filepath of an hdf5 file containing the precomputed fingerprints of your virtual library. MolPAL relies on the assumption that the ordering of the fingerprints in this file is exactly the same as that of the library file and that the encoder used to generate these fingerprints is exactly the same as the one used for model training. MolPAL handles writing this file for you if unspecified, so this option is mostly useful for avoiding the overhead at startup if running MolPAL again with the same library/encoder settings.

#### Optional Settings
MolPAL has a number of different model architectures, encodings, acquisition metrics, and stopping criteria to choose from. Many of these choices have default settings that were arrived at through hyperparameter optimization, but your circumstances may call for modifying these choices. To see the full list, run MolPAL with either the `-h` or `--help` flags. A few common options to specify are shown below.

`-k`: the number of top scores to evaluate when calculating an average. This number may be either a float representing a fraction of the library or an integer to specify precisely how many top scores to average. (Default = 0.005)

`--window-size` and `--delta`: the principal stopping criterion of MolPAL is whether or not the current top-k average score is better than the moving average of the `window_size` most recent top-k average scores by at least `delta`. (Default: `window_size` = 3, `delta` = 0.1)

`--max-explore`: if you would like to limit MolPAL to exploring a fixed fraction of the libary or number of inputs, you can specify that by setting this value. (Default = 1.0)

`--max-epochs`': Alternatively, you may specify the maximum number of epochs of exploration. (Default = 50)

`--model`: the type of model to use. Choices include `rf`, `gp`, `nn`, and `mpn`. (Default = `rf`)  
  - `--conf-method`: the confidence estimation method to use for the NN or MPN models. Choices include `ensemble`, `dropout`, `mve`, and `none`. (Default = 'none'). NOTE: the MPN model does not support ensembling

`--metric`: the acquisition metric to use. Choices include `random`, `greedy`, `ucb`, `pi`, `ei`, `thompson`, and `threshold` (Default = `greedy`.) Some metrics include additional settings (e.g. the threshold value for `threshold`.) 

### Hyperparameter Optimization
While the default settings of MolPAL were chosen based on hyperparameter optimization with Optuna, they were calculated based on the context of structure-based discovery our computational resources. It is possible that these settings are not optimal for your particular problem. To adapt libary_explorer to new circumstances, we recommend first generating a dataset that is representative of your particular problem then peforming hyperparameter optimization of your own using the LookupObjective class. This class acts as an Oracle for your particular objective function, enabling both consistent and, more importantly, near-instant calculation of the objective function for a particular input to save time during hyperparameter optimization.

## Running pyscreener

## Future Directions
Though MolPAL was originally intended for use with protein-ligand docking screens, it was designed with modularity in mind and is easily extendable to other settings as well. All that is required to adapt MolPAL to a new problem is to write a custom `Objective` subclass that implements the `calc` method. This method takes a sequence SMILES strings as an input and returns a mapping from SMILES string -> objective function value to be utilized by the Explorer. _To this end, we are currently exploring the extension of MolPAL to subsequent stages of virtual discovery (mmPBSA, MD, etc.)_ If you make use of the MolPAL library by implementing a new `Objective` subclass, we would be happy to include your work in the main branch.

Related to this, we are looking at implementing distributed computing support into calculations of the objective function. Currently, `Objective` subclasses are implemented using python's multiprocessing module, but this limits parallelization to cores local to the current machine. This severely underutilizes available resources in most HPC setups, so we are looking to address this. We are planning right now to use the [Dask](https://dask.org/) library to address this shortcoming, but we welcome any input or advice from those who have experience in this area.

Lastly, the `Explorer` class was written with abstraction at its core and is thus generalizable to problems outside of virtual chemical discovery. We plan to utilize this fact by further developing the MolPAL library into a general library for batched, Bayesian optimization (after further algorithmic work is performed.)
