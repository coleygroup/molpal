from platform import libc_ver
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import pygmo as pg


path = "../libraries/selectivity_data.csv"
a = pd.read_csv(path)


cols = [1,3,2]
# column 4 has void docking scores 
# column  5 has one score of 19.4 (??) 

X1 = -np.array(a.values[:,cols[0]])
X2 = np.array(a.values[:,cols[1]])
X3 = np.array(a.values[:,cols[2]])
S12 = np.empty(X1.shape[0])
S13 = np.empty(X1.shape[0])

ref_point = max(np.amax(X1), np.amax(X2), np.amax(X3)) + 1


for i in range(0,X1.shape[0]):
    try:
        S12[i] = pg.hypervolume([[-1, -1], [X1[i], X2[i]]]).compute([ref_point, ref_point])
        S13[i] = pg.hypervolume([[-1, -1], [X1[i], X3[i]]]).compute([ref_point, ref_point])
    except:
        try: 
            X1[i] = float(X1[i])
            X2[i] = float(X2[i])
            X3[i] = float(X3[i])
            S12[i] = pg.hypervolume([[-1,-1],[X1[i],X2[i]]]).compute([2,2])
            S13[i] = pg.hypervolume([[-1,-1],[X1[i],X3[i]]]).compute([2,2])
        except:
            print('this point was not included in the plots:')
            X1[i]=np.nan
            X2[i]=np.nan
            X3[i]=np.nan
            S12[i]=np.nan
            S13[i]=np.nan
            bad_list = [X1[i],X2[i],X3[i]]
            print(bad_list)

# plot 1, color is hypervolume 
fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
im1 = ax[0].scatter(X1,X2,c=S12,cmap='inferno')
ax[0].set_xlabel(a.columns[cols[0]] + " (negative score) ")
ax[0].set_ylabel(a.columns[cols[1]])
ax[0].set_title('Docking Scores for ' + a.columns[cols[0]] +' and '+ a.columns[cols[1]])
im2 = ax[1].scatter(X1,X3,c=S13,cmap='inferno')
ax[1].set_xlabel(a.columns[cols[0]] + " (negative score) ")
ax[1].set_ylabel(a.columns[cols[2]])
ax[1].set_title('Docking Scores for ' + a.columns[cols[0]] +' and '+ a.columns[cols[2]])
cbar = fig1.colorbar(im1,label='Hypervolume',ax = ax[1])
cbar.ax.set_yticklabels([])
 

# plot 2, color is third objective

plot2 = plt.figure()
plt.scatter(X1,X2,c=X3,cmap='inferno_r')
plt.xlabel(a.columns[cols[0]])
plt.ylabel(a.columns[cols[1]])
plt.title('Docking Scores for ' + a.columns[cols[0]] +', '+ a.columns[cols[1]] + ', and ' + a.columns[cols[2]])
cbar2 = plt.colorbar()
cbar2.set_label('Docking Score of ' + a.columns[cols[2]],rotation=270)
cbar2.ax.set_yticklabels([])

plt.show()

