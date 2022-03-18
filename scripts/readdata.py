import pickle
import numpy as np 
from experiment import Experiment
from utils import (
    extract_smis, build_true_dict, chunk, style_axis, abbreviate_k_or_M
)

a = 1
# iter 1 
file_path = r'C:\Users\ChemeGrad2021\Dropbox (MIT)\github\molpal\molpal_10k\chkpts\iter_2\scores.pkl'
print(file_path)
data = pickle.load(open(file_path, "rb"))

print(data)
file_path2 = r'C:\Users\ChemeGrad2021\Dropbox (MIT)\github\molpal\molpal_10k\chkpts\iter_2\preds.npz'
data2 = np.load(file_path2)
print(data2)

library = r'C:\Users\ChemeGrad2021\Dropbox (MIT)\github\molpal\libraries\Enamine10k.csv.gz'
smis = extract_smis(library)
d_smi_idx = {smi: i for i, smi in enumerate(smis)}

exptpath = r'C:\Users\ChemeGrad2021\Dropbox (MIT)\github\molpal\molpal_10k'
expt = Experiment(exptpath, d_smi_idx)
a = expt.points_in_order()

true_csv = 
smiles_col = 
score_col = 
title_line = 
maximimize = 
d_smi_score = build_true_dict(
    args.true_csv, args.smiles_col, args.score_col,
    args.title_line, args.maximize
    )

true_smis_scores = sorted(d_smi_score.items(), key=lambda kv: kv[1])[::-1]
true_top_k = true_smis_scores[:args.N]