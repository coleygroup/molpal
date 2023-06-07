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
out_dir = Path(f'results/moo_results_{timestamp}_mace')

acq_funcs = ['ei', 'pi', 'nds'] #, 'random']
cluster_types = [None] # ['both', 'objs', 'fps'] # [None, 'objs', 'fps'] # need to implement both (probably won't run all cluster types here)
seeds = [47, 53, 59] #, 61, 67]
model_seeds = [29, 31, 37] #, 41, 43]

cmds = []

for acq in acq_funcs: 
    for seed, model_seed in zip(seeds, model_seeds):
        for cluster_type in cluster_types:
            # tags = [obj1.stem.split('_')[-1], obj2.stem.split('_')[-1], acq, cluster_type or 'None', f'run-{seed}']
            tags = [f'seed-{seed}-{model_seed}', acq]
            out_folder = out_dir / '_'.join(tags)
            cmd = f'python3 run.py --config {base_config} --metric {acq} --output-dir {out_folder} --model-seed {model_seed} --seed {seed}'

            if cluster_type: 
                cmd = f'{cmd} --cluster-type {cluster_type}'
                
            cmds.append(cmd)

            if acq in {'ei','pi'}: # also run scalarization baseline
                tags[-1] = ''.join([tags[-1],'-scal'])
                out_folder = out_dir / '_'.join(tags)
                cmd = f'python3 run.py --config {base_scal_config} --metric {acq} --output-dir {out_folder} --model-seed {model_seed} --seed {seed}'

                # cmds.append(cmd)


print(f'Running {len(cmds)} molpal runs:')
for cmd in cmds: 
    print(cmd)
    subprocess.call(cmd, shell=True)
