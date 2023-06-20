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
out_dir = Path(f'results/selective_IGF1R')

seeds = [61, 67] # [47, 53, 59] 
model_seeds = [41, 43] # [29, 31, 37] #, 
cmds = []

# scalarization AFs
for acq in ['ei', 'pi', 'greedy']: 
    for seed, model_seed in zip(seeds, model_seeds):
        tags = [f'seed-{seed}-{model_seed}', acq, 'scal']
        out_folder = out_dir / '_'.join(tags)
        cmd = f'python3 run.py --config {base_config} --metric {acq} --output-dir {out_folder} --model-seed {model_seed} --seed {seed} --scalarize'
        
        cmds.append(cmd)

# Pareto AFs
for acq in ['ei', 'pi', 'nds']: 
    for seed, model_seed in zip(seeds, model_seeds):
        tags = [f'seed-{seed}-{model_seed}', acq]
        out_folder = out_dir / '_'.join(tags)
        cmd = f'python3 run.py --config {base_config} --metric {acq} --output-dir {out_folder} --model-seed {model_seed} --seed {seed}'
        
        cmds.append(cmd)

# Random AFs
for seed, model_seed in zip(seeds, model_seeds):
    tags = [f'seed-{seed}-{model_seed}', 'random']
    out_folder = out_dir / '_'.join(tags)
    cmd = f'python3 run.py --config {base_config} --metric random --output-dir {out_folder} --model-seed {model_seed} --seed {seed} --scalarize'
    
    cmds.append(cmd)

print(f'Running {len(cmds)} molpal runs:')
for cmd in cmds: 
    print(cmd)
    subprocess.call(cmd, shell=True)
