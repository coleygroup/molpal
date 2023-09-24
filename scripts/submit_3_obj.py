from datetime import datetime
from timeit import default_timer as time
from pathlib import Path
import subprocess

base_config = Path('moo_runs/enamine_runs/main.ini') # 
rand_config = Path('moo_runs/enamine_runs/main_random.ini')
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
out_dir = Path(f'results/IGF1R_EGFR_pi')

seeds =  [61, 67, 47, 53, 59] 
model_seeds = [41, 43, 29, 31, 37]  
cmds = []

# Pareto AFs
for seed, model_seed in zip(seeds, model_seeds):
    tags = [f'seed-{seed}-{model_seed}-pi']
    out_folder = out_dir / '_'.join(tags)
    cmd = f'python3 run.py --config {base_config} --output-dir {out_folder} --model-seed {model_seed} --seed {seed}'
    
    cmds.append(cmd)

# Random AFs
for seed, model_seed in zip(seeds, model_seeds):
    tags = [f'seed-{seed}-{model_seed}', 'random']
    out_folder = out_dir / '_'.join(tags)
    cmd = f'python3 run.py --config {rand_config} --output-dir {out_folder} --model-seed {model_seed} --seed {seed}'
    
    cmds.append(cmd)

print(f'Running {len(cmds)} molpal runs:')
for cmd in cmds: 
    print(cmd)
    subprocess.call(cmd, shell=True)
