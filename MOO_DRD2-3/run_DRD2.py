import os, sys
sys.path.append('/home/jfromer/molpal-jenna/')

from moo_utils_DRD import Runner
from dlib import cuda
import ray
import tensorflow as tf 



# run on molpalmoo
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

ray.init(num_cpus=10,num_gpus=1)
runner = Runner()
score_record, mse_record, front_record = runner.run()
print(front_record)