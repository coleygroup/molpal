# do glide sp docking calculation
# ylk 2022-5-5
import subprocess

import numpy as np
import pandas as pd
from configargparse import ArgumentParser

ligperp_inp = '''INPUT_FILE_NAME   ???.csv
OUT_SD   ???.sdf
MAX_ATOMS   500
FORCE_FIELD   16
IONIZATION   0
USE_DESALTER   no
GENERATE_TAUTOMERS   no
DETERMINE_CHIRALITIES   no
IGNORE_CHIRALITIES   no
NUM_STEREOISOMERS   1'''

def smi2csv(smi,iter):
    df = pd.DataFrame({'smiles':smi,'title':smi})
    filename = "input_smiles_iter_{}.csv".format(iter)
    df.to_csv(filename,index=None)
    return filename


def ligperp(ligand_filename,iter):
    config_file = 'ligperp_config_iter_{}.inp'.format(iter)
    outname = 'prepared_ligands_iter_{}.sdf'.format(iter)
    com = 'ligprep -inp {} -HOST localhost:96 -NJOBS 96 -TMPLAUNCHDIR -WAIT'.format(config_file)
    with open(config_file,'w') as f:
        config = ligperp_inp.split('\n')
        config[0] = 'INPUT_FILE_NAME   {}'.format(ligand_filename)
        config[1] = 'OUT_SD   {}'.format(outname)
        config = ['{}\n'.format(i) for i in config]
        f.writelines(config)
    subprocess.run(com,shell=True)
    return outname


def _glide_docking(config_file,ligand_file,iter):
    jobname = "glide_docking_iter_{}".format(iter)
    #config_file = "glide_docking_config_iter_{}.in".format(iter)
    com = "glide {} -OVERWRITE -adjust -HOST localhost:96 -TMPLAUNCHDIR -WAIT -JOBNAME {}".format(config_file,jobname)
    with open(config_file,'r') as f:
        config = f.readlines()
        config[1] = 'LIGANDFILE   {}\n'.format(ligand_file)
        #config = ['{}\n'.format(i) for i in config]
    with open(config_file,'w') as f:
        f.writelines(config)
    subprocess.run(com,shell=True)
    return jobname

def process_docking_result(fliename,c=-1):
    df = pd.read_csv(fliename)
    df = df[['title','r_i_docking_score']]
    df = df.groupby('title').agg('min')
    df.dropna(inplace=True)
    df*=c
    #df1 = df.iloc[:,1]
    new_scores = df.T.to_dict('records')[0]
    #new_scores = df.set_index('title').T.to_dict('records')[0]
    assert len(new_scores) == df.shape[0]
    return new_scores

def glide(smis,glide_config,iter):
    smis_filename = smi2csv(smis,iter=iter)
    prepared_ligand_filename = ligperp(ligand_filename=smis_filename,iter=iter)
    jobname = _glide_docking(config_file=glide_config,ligand_file=prepared_ligand_filename,iter=iter)
    results = process_docking_result("{}.csv".format(jobname))
    return results

