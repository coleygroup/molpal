from moo_utils_DRD import Runner
from dlib import cuda
import ray
import os 
# run on molpalmoo
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

ray.init(num_cpus=10,num_gpus=2)
runner = Runner()
score_record, mse_record = runner.run()
print(mse_record)