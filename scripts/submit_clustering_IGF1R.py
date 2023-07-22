from datetime import datetime
import os
import signal
import sys
from timeit import default_timer as time
import ray
from molpal import args, Explorer
from pathlib import Path
import subprocess

base_config = Path('moo_runs/config/IGF1R_CYP_selectivity.ini')
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
out_dir = Path(f'results/IGF1R_clustering')

seeds = [47, 53, 59, 61, 67]
model_seeds =  [29, 31, 37, 41, 43]
cluster_types = ['fps', 'objs', 'both'] # ['both', 'objs', 'fps'] 
cmds = []
acq = 'pi'

# Pareto AFs
for seed, model_seed in zip(seeds, model_seeds):
    for cluster_type in cluster_types: 
        tags = [f'seed-{seed}-{model_seed}', acq, cluster_type]
        out_folder = out_dir / '_'.join(tags)
        cmd = f'python3 run.py --config {base_config} --output-dir {out_folder} --model-seed {model_seed} --seed {seed} --cluster-type {cluster_type} --metric {acq}'
        
        cmds.append(cmd)

print(f'Running {len(cmds)} molpal runs:')
for cmd in cmds: 
    print(cmd)
    subprocess.call(cmd, shell=True)
