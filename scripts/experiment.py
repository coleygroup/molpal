from collections.abc import Iterable
import csv
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import numpy as np

sys.path.append('../molpal')
from molpal.acquirer import metrics

Point = Tuple[str, float]

class Experiment:
    """An Experiment represents the output of a MolPAL run

    It can be queried for the progress at a given iteration, the order in which
    points were acquired, and other conveniences
    """
    def __init__(self, experiment: str, d_smi_idx: Dict):
        self.experiment = Path(experiment)
        self.d_smi_idx = d_smi_idx

        chkpts_dir = self.experiment / 'chkpts'
        self.chkpts = sorted(
            chkpts_dir.iterdir(), key=lambda p: int(p.stem.split('_')[-1])
        )

        data_dir = self.experiment / 'data'
        scores_csvs = [p for p in data_dir.iterdir() if 'final' not in p.stem]
        self.scores_csvs = sorted(
            scores_csvs, key=lambda p: int(p.stem.split('_')[-1])
        )

        config = Experiment.read_config(self.experiment / 'config.ini')
        self.k = int(config['top-k'])
        self.metric = config['metric']
        self.beta = float(config.get('beta', 2.))
        self.xi = float(config.get('xi', 0.001))

    # NOTE: should len(self) be the total number of points 
    # or number of iterations?
    # def __len__(self) -> int:
    #     return len(self.scores_csvs)

    def __getitem__(self, i: int) -> Dict:
        """Get the score data for iteration i"""
        scores, failures = Experiment.read_scores(self.scores_csvs[i])

        return {**scores, **failures}
    
    def __iter__(self) -> Iterable[Dict]:
        """iterate through all the score data at each iteration"""
        for scores_csv in self.scores_csvs:
            scores, failures = Experiment.read_scores(scores_csv)
            yield {**scores, **failures}
    
    @property
    def num_iters(self) -> int:
        """the total number of iterations in this experiment, including the
        initialization batch"""
        return len(self.scores_csvs)

    @property
    def init_size(self) -> int:
        """the size of this experiment's initialization batch"""
        return len(self[0])

    def predictions(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """get the predictions for exploration iteration i.

        The exploration iterations are 1-indexed to account for the 
        iteration of the initialization batch. So self.predictions[1] corresponds to the first exploration iteteration
        
        Returns
        -------
        means : np.ndarray
        vars : np.ndarray

        Raises
        ------
        ValueError
            if i is less than 1
        """
        if i not in range(1, self.num_iters):
            raise ValueError(
                f'arg: i must be in {{1..{self.num_iters}}}. got {i}'
            )
        preds_npz = np.load(self.chkpts[i] / 'preds.npz')

        return preds_npz['Y_pred'], preds_npz['Y_var']

    # def predss(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    #     try:                         # new way
    #         preds_npzs = [
    #             np.load(chkpt / 'preds.npz') for chkpt in self.chkpts[1:]
    #         ]
    #         predss, varss = zip(*[
    #             (preds_npz['Y_pred'], preds_npz['Y_var'])
    #             for preds_npz in preds_npzs
    #         ])
    #         return predss, varss

    #     except FileNotFoundError:   # old way
    #         predss = [
    #             np.load(chkpt / 'preds.npy') for chkpt in self.chkpts[1:]
    #         ]
    #         return predss, []

    def new_points_by_epoch(self) -> List[Dict]:
        """get the set of new points acquired at each iteration in the list of 
        scores_csvs that are already sorted by iteration"""
        new_points_by_epoch = []
        all_points = {}

        for scores in self:
            new_points = {smi: score for smi, score in scores.items()
                          if smi not in all_points}
            new_points_by_epoch.append(new_points)
            all_points.update(new_points)
        
        return new_points_by_epoch

    def utilities(self) -> List[np.ndarray]:
        Us = []

        for i in range(1, self.num_iters):
            Y_pred, Y_var = self.predictions(i)
            ys = list(self[i-1].values())

            Y = np.nan_to_num(np.array(ys, dtype=float), nan=-np.inf)
            current_max = np.partition(Y, -self.k)[-self.k]

            Us.append(metrics.calc(
                self.metric, Y_pred, Y_var,
                current_max, 0., self.beta, self.xi, False
            ))

        return Us

    def points_in_order(self) -> List[Point]:
        """Get all points acquired during this experiment's run in the order
        in which they were acquired"""
        init_batch, *exp_batches = self.new_points_by_epoch()

        all_points_in_order = []
        all_points_in_order.extend(init_batch.items())

        for new_points, U in zip(exp_batches, self.utilities()):
            us = np.array([U[self.d_smi_idx[smi]] for smi in new_points])

            new_points_in_order = [
                smi_score for _, smi_score in sorted(
                    zip(us, new_points.items()), reverse=True
                )
            ]
            all_points_in_order.extend(new_points_in_order)
        
        return all_points_in_order
    
    @staticmethod
    def read_scores(scores_csv: str) -> Tuple[Dict, Dict]:
        """read the scores contained in the file located at scores_csv"""
        scores = {}
        failures = {}

        with open(scores_csv) as fid:
            reader = csv.reader(fid)
            next(reader)
            for row in reader:
                try:
                    scores[row[0]] = float(row[1])
                except:
                    failures[row[0]] = None
        
        return scores, failures
    
    @staticmethod
    def read_config(config_file: str) -> Dict:
        """parse an autogenerated MolPAL config file to a dictionary"""
        with open(config_file) as fid:
            return dict(line.split(' = ') for line in fid.read().splitlines())
        