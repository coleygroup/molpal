# MolPAL: Molecular Pool-based Active Learning
# Efficient Exploration of Virtual Chemical <br/> Libraries through Active Learning

## Overview
This repository contains the source of MolPAL, both a library and software for the accelerated discovery of compounds in high-throughput virtual screening environments.

## Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Requirements](#requirements)
- [Installation](#installation)
- [Object Model](#object-model)
- [Directory Structure](#directory-structure)
- [Running MolPAL](#running-molpal)
  * [Novel Targets](#novel-targets)
  * [Hyperparameter Optimization](#hyperparameter-optimization)
- [Future Directions](#future-directions)

## Requirements
- Python (>= 3.6)
- the `pyscreener` library and all of its dependencies

NN and MPN models can make use of GPUs for significantly faster model training and inference. To utilize this, you must have the following:
- CUDA (>= 8.0)
- cuDNN

## Installation
The first step in installing MolPAL is to clone this repository: `git clone <this_repo>`

The easiest way to install all dependencies is to use conda along with the supplied environment YAML file, but you may also install them manually, if desired.

### conda
1. (if necessary) install conda. Instructions will vary based on your system
1. `cd /path/to/molpal`
1. `conda env create -f environment.yml`

Before running MolPAL, be sure to first activate the environment: `conda activate molpal`

### manual (FIX ME)

The following packages are __necessary__ to install before running MolPAL:
- [chemprop](https://github.com/chemprop/chemprop)
- configargparse
- h5py
- numpy
- rdkit
- scikit-learn
- scipy
- pytorch (built with CUDA if utilizing GPU acceleration)
- tensorflow
- tqdm

The following packages are _optional_ to install before running MolPAL:
- cudatoolkit (if utilizing GPU acceleration; whichever version matches your CUDA build)
- cudnn (if utilizing GPU acceleration; whichever version matches your cuDNN build)
- [map4](https://github.com/reymond-group/map4) (if utilizing the map4 fingerprint)
- optuna (if planning to perform hyperparameter optimization)
- [tmap](https://github.com/reymond-group/tmap) (if utilizing the map4 fingerprint)

## Object Model

MolPAL is a software for batched, Bayesian optimization in a virtual screening environment. At the core of this software is the `molpal` library, which implements defines several classes that implement various elements of the optimization routine.

__Explorer__: An [`Explorer`](molpal/explorer.py) is the abstraction of the optimization routine. It ties together the `MoleculePool`, `Acquirer`, `Encoder`, `Model`, and `Objective`, which each handle (roughly) a single step of a Bayesian optimization loop, into a full optimization procedure. Its main functionality is defined by the `run()` method, which performs the optimization until a stopping condition is met, but it also defines other convenience functions that make it amenable to running a single iteration of the optimization loop and interrogating its current state if optimization is desired to be run interactively.

__MoleculePool__: A [`MoleculePool`](molpal/pools/base.py) defines the virtual library (i.e., domain of inputs)

__Acquirer__: An [`Acquirer`](molpal/acquirer/acquirer.py) handles acquisition of unlabeled inputs from the MoleculePool according to its `metric` and the prior distribution over the data. The [`metric`](molpal/acquirer/metrics.py) is a function that takes an input array of predictions and returns an array of equal dimension containing acquisition utilities.

__Encoder__: An [`Encoder`](molpal/encoders.py) computes the uncompressed feature representation of an input based on its identifier for use with clustering and models that expect vectors as inputs.

__Model__: A [`Model`](molpal/model/base.py) is trained on labeled data to produce a posterior distribution that guides the sequential round of acquisition

__Objective__: An [`Objective`](molpal/objectives.base.py) handles calculation of the objective function for unlabeled inputs

## Directory Structure
<pre>
molpal
├── acquirer
│   ├── acquirer.py        # Acquirer class implementation
│   └── metrics.py         # metric functions
├── encoders.py            # Encoder abstract base (ABC) class interface definition and various implementations thereof
├── explorer.py
├── models
│   ├── base.py            # Model ABC interface definition
│   ├── mpnmodels.py       # implementations of Model subclasses that use pytorch message-passing neural nets in the backend 
│   ├── mpnn/              # submodule containing functions used by MPNN models
│   ├── nnmodels.py        # "..." that use tensorflow FFN model in the backend
│   ├── sklmodels.py       # "..." that use scikit-learn models in the backend
│   └── utils.py           # utility functions used in model code
├── objectives
│   ├── base.py            # Objective ABC interface definition
│   ├── docking.py         # implementation of the DockingObjective
│   └── lookup.py          # implementation of the LookupObjective
└── pools
    ├── cluster.py         # functions for clustering a feature matrix
    ├── fingerprints.py    # functions for calculating a feature matrix    
    ├── base.py            # implementation of the base (Eager) MoleculePool, which precomputes the feature matrix
    └── lazypool.py        # implementation of the LazyMoleculePool, which doesn't precompute the feature matrix
</pre>

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
- (optional) `--fps`: the filepath of an hdf5 file containing the precomputed fingerprints of your virtual library. MolPAL relies on the assumption that the ordering of the fingerprints in this file is exactly the same as that of the library file and that the encoder used to generate these fingerprints is exactly the same as the one used for model training. MolPAL handles writing this file for you if unspecified, so this option is mostly useful for avoiding the overhead at startup of running MolPAL again with the same library/encoder settings.

#### Optional Settings
MolPAL has a number of different model architectures, encodings, acquisition metrics, and stopping criteria to choose from. Many of these choices have default settings that were arrived at through hyperparameter optimization, but your circumstances may call for modifying these choices. To see the full list, run MolPAL with either the `-h` or `--help` flags. A few common options to specify are shown below.

`-k`: the number of top scores to evaluate when calculating an average. This number may be either a float representing a fraction of the library or an integer to specify precisely how many top scores to average. (Default = 0.005)

`--window-size` and `--delta`: the principal stopping criterion of MolPAL is whether or not the current top-k average score is better than the moving average of the `window_size` most recent top-k average scores by at least `delta`. (Default: `window_size` = 3, `delta` = 0.1)

`--max-explore`: if you would like to limit MolPAL to exploring a fixed fraction of the libary or number of inputs, you can specify that by setting this value. (Default = 1.0)

`--max-epochs`': Alternatively, you may specify the maximum number of epochs of exploration. (Default = 50)

`--model`: the type of model to use. Choices include `rf`, `gp`, `nn`, and `mpn`. (Default = `rf`)  
  - `--conf-method`: the confidence estimation method to use for the NN or MPN models. Choices include `ensemble`, `dropout`, `mve`, and `none`. (Default = 'none'). NOTE: the MPN model does not support ensembling

`--metric`: the acquisition metric to use. Choices include `random`, `greedy`, `ucb`, `pi`, `ei`, `thompson`, and `threshold` (Default = `greedy`.) Some metrics include additional settings (e.g. the β value for `ucb`.) 

### Hyperparameter Optimization
While the default settings of MolPAL were chosen based on hyperparameter optimization with Optuna, they were calculated based on the context of structure-based discovery our computational resources. It is possible that these settings are not optimal for your particular problem. To adapt MolPAL to new circumstances, we recommend first generating a dataset that is representative of your particular problem then peforming hyperparameter optimization of your own using the LookupObjective class. This class acts as an Oracle for your particular objective function, enabling both consistent and near-instant calculation of the objective function for a particular input, saving time during hyperparameter optimization.

## Future Directions
Though MolPAL was originally intended for use with protein-ligand docking screens, it was designed with modularity in mind and is easily extendable to other settings as well. All that is required to adapt MolPAL to a new problem is to write a custom `Objective` subclass that implements the `calc` method. This method takes a sequence SMILES strings as an input and returns a mapping from SMILES string -> objective function value to be utilized by the Explorer. _To this end, we are currently exploring the extension of MolPAL to subsequent stages of virtual discovery (mmPBSA, MD, etc.)_ If you make use of the MolPAL library by implementing a new `Objective` subclass, we would be happy to include your work in the main branch.

Lastly, the `Explorer` class was written with abstraction at its core and is thus generalizable to problems outside of virtual chemical discovery. We plan to utilize this fact by further developing the MolPAL library into a general library for batched, Bayesian optimization (after further algorithmic work is performed.)
