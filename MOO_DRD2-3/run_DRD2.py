import os, sys
sys.path.append('/home/jfromer/molpal-jenna/')

from moo_utils_DRD import Runner
from dlib import cuda
import ray
import tensorflow as tf 
import numpy as np 
from pareto import Pareto
import imageio
import matplotlib.pyplot as plt 

def plot_acquired(runner: Runner, scores):
    true_scores = runner.objective_values
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(true_scores[:,0], true_scores[:,1], color='gray',label='Not sampled',s=8)
    ax.scatter([],[],color='blue', label='sampled',s=8)
    for smiles in scores:
        ax.scatter(scores[smiles][0], scores[smiles][1], color='blue',s=8)
    
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.legend()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


# run on molpalmoo
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

ray.init(num_cpus=10,num_gpus=1)
runner = Runner(
    acq_func='ei',
    n_iter=5,
    c=[1,0],
    model='nn'
    )
score_record, mse_record, front_record = runner.run()

# quick analysis 
reference_min = [np.min(np.array(list(runner.objectives[j](runner.smis).values()))) for j in range(runner.num_objs)]
reference_max = [np.max(np.array(list(runner.objectives[j](runner.smis).values()))) for j in range(runner.num_objs)]

hvs2 = []
for i in range(len(front_record)):
    pareto = Pareto(num_objectives=runner.num_objs)
    pareto.update_front(front_record[i])
    hvs2.append(pareto.volume_in_dominance(
        ref_min=reference_min,
        ref_max=reference_max)
        )

# get max_hv 
pareto = Pareto(num_objectives=runner.num_objs)
pareto.update_front(runner.objective_values)
max_hv = pareto.volume_in_dominance(        
        ref_min=reference_min,
        ref_max=reference_max
        )

images = [plot_acquired(runner, score_record[0])]
for i in range(1,runner.n_iter):
    images.append(plot_acquired(runner, score_record[i]))

imageio.mimsave('./powers.gif', images)