import os, sys
sys.path.append('/home/jfromer/molpal-jenna/')

from molpal.models.nnmodels import (
        NNModel, NNDropoutModel, NNEnsembleModel, NNTwoOutputModel
    )
from molpal import args, Explorer, featurizer, pools, acquirer, models
from molpal.pools.base import MoleculePool
from molpal.objectives.lookup import LookupObjective
from molpal.featurizer import feature_matrix
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

fingerprint='morgan'
radius = 2
length = 2048
n_per_acq = 2000

featurizer = featurizer.Featurizer(
                fingerprint=fingerprint,
                radius=radius, 
                length=length
            ) 
pool = MoleculePool(
                    libraries=['data/drd3_data.csv'],
                    featurizer=featurizer,
                    fps='data/drd2_data.h5', 
                    fps_path='data/drd2_data.h5',
                )  
model = NNEnsembleModel(
                    input_size=length, 
                    ensemble_size=3,
                    activation='relu',
                    test_batch_size=1000
                ) 
acquirer = acquirer.Acquirer(
                size=len(pool),
                init_size=n_per_acq,
                batch_sizes=[n_per_acq],
            )  
objective = LookupObjective(
                'examples/objective/DRD2_docking.ini', minimize=0
            )

             
new_smis = acquirer.acquire_initial(xs=list(objective.data.keys()))
ys = list(objective(new_smis).values())
model.train( new_smis, ys,
                        featurizer=featurizer, retrain = False)

# get residuals 
unacq_smis = [key for key in list(objective.data.keys()) if key not in new_smis]
unacq_true = np.array(list(objective(unacq_smis).values()))
unacq_pred = model.get_means(np.array(feature_matrix(smis=unacq_smis, featurizer=featurizer))) 
residuals = np.abs(np.subtract(unacq_pred,unacq_true))

a, b = np.polyfit(unacq_true, residuals,1)


sort=1
data , x_e, y_e = np.histogram2d( unacq_true, residuals, bins = [1000,1000], density = True )
z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , 
                data , np.vstack([unacq_true, residuals]).T , method = "splinef2d", bounds_error = False)
z[np.where(np.isnan(z))] = 0.0
if sort:
    idx = z.argsort()
    x, y, z = unacq_true[idx], residuals[idx], z[idx]
else:
    x = unacq_true
    y = residuals
fig, ax = plt.subplots()
ax.scatter(x, y,c=z,s=4)
plt.plot(unacq_true, a*unacq_true+b)
ax.set_xlabel('True Docking Score')
ax.set_ylabel('Absolute Residuals')
ax.set_title('Residuals of Predicted Docking Scores to DRD2 for Unacquired Molecules')
plt.savefig('Residuals vs. docking score DRD2.jpg')



print('done')