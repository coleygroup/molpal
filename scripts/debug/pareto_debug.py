from scripts.debug.test_pareto import Pareto as new_pareto
from molpal.acquirer.pareto import Pareto as old_pareto 
import numpy as np 
import time

with open('scripts/scores.np', 'rb') as f:
    new_points = np.load(f)

print(new_points.shape)

n_points = 1000
rows = np.random.choice([True,False], size=new_points.shape[0], p=[n_points/new_points.shape[0], 1-n_points/new_points.shape[0]])
new_points = new_points[rows]

# N_points = 4e6
# N_obj = 3
# new_points= np.random.uniform(low=-11,high=-2, size=(N_points, N_obj,)).round(1)
# rows = np.random.choice([True,False], size=new_points.shape[0], p=[0.1, 0.9])
# new_points[rows].fill(-np.inf)

start = time.time()
new_p = new_pareto(num_objectives=new_points.shape[1], compute_metrics=False)
new_p.update_front(new_points.copy())
new_p.sort_front()
new_front, front_num_new = new_p.export_front()
end = time.time()
print(f"Time taken: {end - start}")

start = time.time()
# new_points = np.random.randint(0,100,size=(N_points, N_obj))
old_p = old_pareto(num_objectives=new_points.shape[1], compute_metrics=False)
old_p.update_front(new_points.copy())
front_old, front_num_old = old_p.export_front()
end = time.time()
print(f"Time taken: {end - start}")



assert str(front_num_old) == str(front_num_new)