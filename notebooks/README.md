# Recreating data and figures

## Design Space Pruning

### Step 1: Run MolPAL on the given targets.
The general setup for each run was as follows:
- `model=MPN`
- `--conf-method=mve`
- `--metric=ucb`
- `--top-k=0.001`
- `--pool=lazy`
- `--minimize`
- `--objective=lookup`

When pruning was performed, the following parameters were used:
- `--prune`
- `--prune-min-hit-prob=0.025`

See the following sections for specific details on each experiment

#### DOCKSTRING
download the full dataset TSV file from [here](https://figshare.com/s/95f2fed733dec170b998?file=30562257). You can either use as-is, using `\t` as your delimiter for a `LookupObjective` or convert it to a CSV and remove the inchikey column. Using this file with a `LookupObjective` entails creating an objective config file with the `--score-col` value set corresponding to the column of your selected target. Yes, this means that to run MolPAL across all 58 DOCKSTRING targets, you will need to write 58 separate objective config files. This same file was used as the `--library`. Each experiment was repeated 5 times

The output directories should be organized in the following hierarchy:
```
DOCKSTRING_RUNS_ROOT
├── 0.001
│   ├── prune
│   │   ├── abl1
│   │   │   ├── rep-0 <-- output directory of a MolPAL run
│   │   │  ...
│   │   │   └── rep-N
│   │  ...
│   │   └── thrb
│   └── full
│       ├── abl1
│      ...
│       └── thrb
├── 0.002
|   ├── prune
│   └── full
└── 0.004
```

#### AmpC Glide
download the dataset from [here](http://htttps//www.schrodinger.com/other-downloads) and delete any placeholder scores. When we received the dataset, there were a few molecules with docking scores around +10000 kcal/mol, so we removed any ridiculous outliers like those. Run MolPAL using that dataset as both your `LookupObjective` *and* `MoleculePool`. Each experiment was repeated 3 times.

The output directories should be organized in the following hierarchy:
```
AMPC_RUNS_ROOT
├── 0.001
│   ├── prune
│   │   ├── rep-0 <-- output directory of a MolPAL run
│   │  ...
│   │   └── rep-N
│   └── full
│       ├── rep-0
│      ...
│       └── rep-N
├── 0.002
|   ├── prune
│   └── full
└── 0.004
```

#### Enamine HTS
⚠️ these experiments used a `greedy` metric! ⚠️

Run MolPAL using [this CSV](../data/EnamineHTS_scores.csv.gz) as your `LookupObjective` and the [Enamine HTS library](../libraries/EnamineHTS.csv.gz) as your `MoleculePool`. The active learning experiments were repeated 5 times, and the single-batch experiments were repeated 3 times.

The output directories of the active learning runs should organized like so
```
HTS_SB_ROOT
└── 0.004
    ├── rep-0 <-- output directory of a MolPAL run
   ...
    └── rep-N
```

The output directories of the single-batch runs should be organized like so:
```
HTS_AL_ROOT
├── 0.004
|   ├── rep-0 <-- output directory of a MolPAL run
|  ...
│   └── rep-N
└── 0.020
    ├── rep-0 <-- output directory of a MolPAL run
   ...
    └── rep-N
```

### Step 2: **(DOCKSTRING only)** Process and collate the data
First, run the [`process.py`](../scripts/process.py) script for each group of experiments (i.e., the group of repetitions) as the arguments to `--expts`. Organize the resulting `.npy` files in a similar hierarchy as above, naming each file simply `TARGET.npy`. We analyzed the top 250 scores for each task.

**NOTE:** you must have the same number of repetitions for each group of experiments before proceeding.

Next, run the [`collate.py`](../scripts/collate.py) script, supplying the proper `root` directory of the top of the hierarchy and the batch sizes for which you've run the experiments.

### Step 3: Make the figures
See the corresponding notebooks for the figures you'd like to make.

## Original MolPAL figures
Note that the code to generate figures from the original MolPAL publication has been refactored and largely removed from the repo, but the general scaffold to recreate most of these figures with the new code can be found in [`hts-figures.ipynb`](./hts-figures.ipynb). _Note_: this notebook can be used to recreate figures for non-HTS datasets too! To recreate the UMAP figure, see [`umap_fig.py`](../scripts/umap_fig.py). To recreate the score histograms, see [`dockstring-figures.ipynb`](./dockstring-figures.ipynb).

The general directory structure for the raw data was organized like so:
```
RUNS_HOME               # some arbitrary directory name
├── SIZE                # the init/batch-size: 0.01, 0.004, 0.002, or 0.001
│   ├── MODEL           # the model used: MPN, NN, or RF
│   │   ├── METRIC      # the acquisition metric used: greedy, UCB, PI, EI, TS
│   │   |   ├── rep_0   # the output directory of a MolPAL run
|   |   |  ...
│   │   |   └── rep_N
│   │  ...
│   │   └── METRIC
|  ...
│   └── MODEL
...
└── SIZE
```