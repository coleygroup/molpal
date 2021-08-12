from collections import Counter
import csv
import heapq
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple, Union
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
        final_csv = data_dir / 'all_explored_final.csv'
        final_scores, final_failures = Experiment.read_scores(final_csv)
        self.__size = len({**final_scores, **final_failures})

        scores_csvs = [p for p in data_dir.iterdir() if 'final' not in p.stem]
        self.scores_csvs = sorted(
            scores_csvs, key=lambda p: int(p.stem.split('_')[-1])
        )

        config = Experiment.read_config(self.experiment / 'config.ini')
        self.k = int(config['top-k'])
        self.metric = config['metric']
        self.beta = float(config.get('beta', 2.))
        self.xi = float(config.get('xi', 0.001))

        print(len(self))

    # NOTE: should len(self) be the total number of points 
    # or number of iterations?
    def __len__(self) -> int:
        return self.__size

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
    
    def reward_curve(
        self, true_top_k: List[Point], reward: str = 'scores'
    ):
        """Calculate the reward curve of a molpal run

        Parameters
        ----------
        experiment : Experiment
            the data structure corresponding to the MolPAL experiment
        true_top_k : List
            the list of the true top-k molecules as tuples of their SMILES string
            and associated score
        reward : str, default='scores'
            the type of reward to calculate

        Returns
        -------
        np.ndarray
            the reward as a function of the number of molecules sampled
        """
        all_points_in_order = self.points_in_order()
        k = len(true_top_k)

        if reward == 'scores':
            _, true_scores = zip(*true_top_k)
            missed_scores = Counter(true_scores)

            all_hits_in_order = np.zeros(len(all_points_in_order), dtype=bool)
            for i, (_, score) in enumerate(all_points_in_order):
                if score not in missed_scores:
                    continue
                all_hits_in_order[i] = True
                missed_scores[score] -= 1
                if missed_scores[score] == 0:
                    del missed_scores[score]
            reward_curve = 100 * np.cumsum(all_hits_in_order) / k

        elif reward == 'smis':
            true_top_k_smis = {smi for smi, _ in true_top_k}
            all_hits_in_order = np.array([
                smi in true_top_k_smis
                for smi, _ in all_points_in_order
            ], dtype=bool)
            reward_curve = 100 * np.cumsum(all_hits_in_order) / k

        elif reward == 'top-k-ave':
            reward_curve = np.zeros(len(all_points_in_order), dtype='f8')
            heap = []

            for i, (_, score) in enumerate(all_points_in_order[:k]):
                if score is not None:
                    heapq.heappush(heap, score)
                top_k_avg = sum(heap) / k
                reward_curve[i] = top_k_avg
            reward_curve[:k] = top_k_avg

            for i, (_, score) in enumerate(all_points_in_order[k:]):
                if score is not None:
                    heapq.heappushpop(heap, score)

                top_k_avg = sum(heap) / k
                reward_curve[i+k] = top_k_avg

        elif reward == 'total-ave':
            _, all_scores_in_order = zip(*all_points_in_order)
            Y = np.array(all_scores_in_order, dtype=float)
            Y = np.nan_to_num(Y)
            N = np.arange(0, len(Y)) + 1
            reward_curve = np.cumsum(Y) / N
            
        else:
            raise ValueError

        return reward_curve

    @staticmethod
    def read_scores(scores_csv: Union[Path, str]) -> Tuple[Dict, Dict]:
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
        