from datetime import datetime
import os
import signal
import sys
from timeit import default_timer as time
import ray
from molpal import args, Explorer
from pathlib import Path
import subprocess

base_config = Path('moo_runs/config/JAK2_LCK_selectivity.ini')
base_scal_config = Path('moo_runs/config/JAK2_LCK_selectivity_scalarization.ini')
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
out_dir = Path(f'results/moo_all_w_greedy_baseline')

acq_funcs = ['ei', 'pi', 'nds', 'random']
cluster_types = [None, 'both', 'objs', 'fps'] 
seeds = [47, 53, 59, 61, 67]
model_seeds = [29, 31, 37, 41, 43]

cmds = []

for acq in acq_funcs: 
    for seed, model_seed in zip(seeds, model_seeds):
        for cluster_type in cluster_types:
            tags = [f'seed-{seed}-{model_seed}-{cluster_type}', acq]
            out_folder = out_dir / '_'.join(tags)
            cmd = f'python3 run.py --config {base_config} --metric {acq} --output-dir {out_folder} --model-seed {model_seed} --seed {seed}'

            if cluster_type: 
                cmd = f'{cmd} --cluster-type {cluster_type}'
                
            cmds.append(cmd)

            if cluster_type in {None,'fps'}: # also run greedy scalarization baseline
                tags[-1] = ''.join([tags[-1],'-scal'])
                out_folder = out_dir / '_'.join(tags)
                cmd = f'python3 run.py --config {base_scal_config} --metric greedy --output-dir {out_folder} --model-seed {model_seed} --seed {seed}'
                
                if cluster_type: 
                    cmd = f'{cmd} --cluster-type {cluster_type}'

                cmds.append(cmd)


print(f'Running {len(cmds)} molpal runs:')
for cmd in cmds: 
    print(cmd)
    subprocess.call(cmd, shell=True)
