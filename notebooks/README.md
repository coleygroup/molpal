# Recreating Figures and Data

## Step 1: Run MolPAL on the given targets.

The general setup for each run was as follows:
- `model = MPN`
- `--conf-method = mve`
- `--metric = ucb`
- `--top-k = 0.001`
- `--pool = lazy`
- `--minimize`
- `--objective = lookup`

#### DOCKSTRING
download the full dataset TSV file from [here](https://figshare.com/s/95f2fed733dec170b998?file=30562257). You can either use as-is, using `\t` as your delimiter for a `LookupObjective` or convert it to a CSV and remove the inchikey column. Using this file with a `LookupObjective` entails setting the `--score-col` value correctly depending on your target of choice.

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
download the dataset from [here](http://htttps//www.schrodinger.com/other-downloads) and delete any placeholder scores. When we received the dataset, there were a few molecules with docking scores around +10000 kcal/mol, so we removed any ridiculous outliers like those.

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

## Step 2: Process and collate the data
**You can skip this step for AmpC data**

First, run the [`process.py`](../scripts/process.py) script for each group of experiments (i.e., the group of repetitions) as the arguments to `--expts`. Organize the resulting `.npy` files in a similar hierarchy as above, naming each file simply `TARGET.npy`. We analyzed the top 250 scores for each task.

**NOTE:** you must have the same number of repetitions for each group of experiments before proceeding.

Next, run the [`collate.py`](../scripts/collate.py) script, supplying the proper `root` directory of the top of the hierarchy and the batch sizes for which you've run the experiments.

## Step 3: Make the figures

See the corresponding notebooks for the figures you'd like to make