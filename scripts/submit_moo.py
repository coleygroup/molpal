from datetime import datetime
import os
import signal
import sys
from timeit import default_timer as time
import ray
from molpal import args, Explorer
from pathlib import Path
import subprocess

base_config = Path('moo_runs/config/DRD_multiobj_base.ini')
obj_confs = Path('moo_runs/objective')
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
out_dir = Path(f'moo_results_{timestamp}')

num_runs = 3
acq_funcs = ['ei', 'pi', 'nds', 'random']
cluster_types = [None,'objs', 'fps'] # need to implement both (probably won't run all cluster types here)
objective_configs_DRD2 = list(obj_confs.glob('DRD2*'))[0]
objective_configs_DRD3 = list(obj_confs.glob('DRD2*'))[0]

cmds = []
for obj1 in [objective_configs_DRD2]: 
    for obj2 in [objective_configs_DRD3]: 
        for acq in acq_funcs: 
            for cluster_type in cluster_types:
                for seed in range(num_runs):
                    # tags = [obj1.stem.split('_')[-1], obj2.stem.split('_')[-1], acq, cluster_type or 'None', f'run-{seed}']
                    tags = [acq, cluster_type or 'None', f'run-{seed}']
                    out_folder = out_dir / '_'.join(tags)
                    cmd = f'python3 run.py --config {base_config} --objective-config {obj1} {obj2} --metric {acq} --output-dir {out_folder}'
                    
                    if cluster_type: 
                        cmd = f'{cmd} --cluster-type {cluster_type}'
                    
                    cmds.append(cmd)

                    if acq in {'ei','pi'}: # also run scalarization baseline
                        tags.append('scal')
                        out_folder = out_dir / '_'.join(tags)
                        cmd = f'python3 run.py --config {base_config} --objective-config {obj1} {obj2} --metric {acq} --output-dir {out_folder} --scalarize'


                        cmds.append(cmd)


print(f'Running {len(cmds)} molpal runs:')
for cmd in cmds: 
    print(cmd)
    subprocess.call(cmd, shell=True)
