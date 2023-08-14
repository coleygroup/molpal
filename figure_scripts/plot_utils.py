from configargparse import ArgumentParser
from pathlib import Path 
from typing import List, Union, Dict
from molpal import args, pools, featurizer
from molpal.objectives.lookup import LookupObjective
import numpy as np 
import pygmo as pg 
import pickle 
from molpal.acquirer.pareto import Pareto
from pymoo.indicators.gd import GD

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

labels = {'ei': 'EI', 'pi': 'PI', 'nds': 'NDS', 'greedy': 'Greedy'}
acq_labels = {'ei': ['EHI', 'EI'], 'pi': ['PHI', 'PI'], 'nds': ['NDS'],'greedy': ['Greedy'], 'random': ['Random']}
method_colors = {
    'ei':'#405E90',
    'pi':'#D45127',
    'nds':'#AA2165',
    'random':'#818084',
    'greedy': '#AA2165', 
}

cluster_labels = {
    'None': 'No clustering',
    'fps': 'Feature',
    'objs': 'Obj',
    'both': 'Feature + Obj'
}

cluster_colors = {
    'None': '#0E713E',
    'fps': '#44AA99',
    'objs': '#CC6677',
    'both': '#882255'
}

it_colors = ['#3491C1', '#7D2AC1', '#B9305C', '#DC5501', '#DE9A00', '#377501', '#B4B5B4']
shapes = ['^','*','d','s','x','v','o']
it_labels = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5', 'Iteration 6', 'True Front']


def set_style():
    """set_style"""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    sns.set(context="paper", style="ticks") 
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"
    mpl.rcParams["xtick.major.size"] = 2.5
    mpl.rcParams["ytick.major.size"] = 2.5

    mpl.rcParams["xtick.major.width"] = 0.45
    mpl.rcParams["ytick.major.width"] = 0.45

    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.linewidth"] = 0.45
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["axes.titlesize"] = 8
    mpl.rcParams["figure.titlesize"] = 8
    mpl.rcParams["figure.titlesize"] = 8
    mpl.rcParams["legend.fontsize"] = 7
    mpl.rcParams["legend.title_fontsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7
    mpl.rcParams['figure.dpi'] = 300

    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['legend.fancybox'] = False
    mpl.rcParams['legend.facecolor'] = "none"

    mpl.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth

def set_size(w, h, ax=None):
    """w, h: width, height in inches
    Resize the axis to have exactly these dimensions
    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)



