from collections import Counter
import csv
import heapq
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

Point = Tuple[str, Optional[float]]  # a Point is a SMILES string and associated score


class Experiment:
    """An Experiment represents the output of a MolPAL run

    It can be queried for the progress at a given iteration, the order in which
    points were acquired, and other conveniences
    """

    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)

        chkpts_dir = self.path / "chkpts"
        try:
            self.chkpts = sorted(
                chkpts_dir.iterdir(), key=lambda p: int(p.stem.split("_")[1])
            )
            self.config = Experiment.read_config(self.path / "config.ini")

        except FileNotFoundError:
            pass

        data_dir = self.path / "data"
        try:
            self.__size = len(
                Experiment.read_scores(data_dir / "all_explored_final.csv")
            )
        except:
            self.__size = None

        scores_csvs = [p for p in data_dir.iterdir() if "final" not in p.stem]
        self.scores_csvs = sorted(scores_csvs, key=lambda p: int(p.stem.split("_")[-1]))
        self.__sizes = [len(self[i]) for i in range(self.num_iters)]

    def __len__(self) -> int:
        """the total number of inputs sampled in this experiment"""
        if self.__size is None:
            raise IncompleteExperimentError

        return self.__size

    def __getitem__(self, i: int) -> List[Point]:
        """Get the score data for iteration i, where i=0 is the initialization batch"""
        return Experiment.read_scores(self.scores_csvs[i])

    def __iter__(self) -> Iterator[List[Point]]:
        """iterate through all the score data at each iteration"""
        for scores_csv in self.scores_csvs:
            yield Experiment.read_scores(scores_csv)

    @property
    def num_iters(self) -> int:
        """the total number of iterations in this experiment, including the
        initialization batch"""
        return len(self.scores_csvs)

    @property
    def init_size(self) -> int:
        """the size of this experiment's initialization batch"""
        return self.__sizes[0]

    @property
    def num_acquired(self) -> List[int]:
        """The total number of points acquired *by* iteration i, where i=0 is the
        initialization batch"""
        return self.__sizes

    def get(self, i: int, N: int) -> List[Point]:
        """get the top-N molecules explored at iteration i"""
        return sorted(
            Experiment.read_scores(self.scores_csvs[i]),
            key=lambda xy: xy[1] or -float("inf"), reverse=True
        )[:N]

    def new_pointss(self) -> List[List[Point]]:
        """get the new points acquired at each iteration"""
        new_pointss = [self[0]]
        N = len(new_pointss[0])

        for points in self[1:]:
            new_points = [points[N:]]
            new_pointss.append(new_points)
            N = len(points)

        return new_pointss

    def predictions(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """get the predictions for exploration iteration i.

        The exploration iterations are 1-indexed to account for the initialization batch. I.e.,
        self.predictions(1) corresponds to the first exploration iteration

        Returns
        -------
        means : np.ndarray
        vars : np.ndarray

        Raises
        ------
        ValueError
            if i is less than 1
        """
        if not hasattr(self, "chkpt_dirs"):
            raise NotImplementedError("this experiment has no checkpoints!")

        if i not in range(1, self.num_iters):
            raise ValueError(f"arg: i must be in {{1..{self.num_iters}}}. got {i}")

        npz = np.load(self.chkpts[i - 1] / "preds.npz")
        return npz["Y_pred"], npz["Y_var"]

    def curve(self, true_points: List[Point], reward: str = "scores"):
        """Calculate the reward curve of a molpal run

        Parameters
        ----------
        true_points : List[Point]
            the list of the true top-N points
        reward : str, default='scores'
            the type of reward to calculate

        Returns
        -------
        np.ndarray
            the reward as a function of the number of molecules sampled
        """
        all_points_in_order = self[-1]
        k = len(true_points)

        if reward == "scores":
            _, true_scores = zip(*true_points)
            missed_scores = Counter(true_scores)
            all_hits_in_order = np.zeros(len(all_points_in_order), dtype=bool)
            
            for i, (_, score) in enumerate(all_points_in_order):
                if score not in missed_scores:
                    continue
                all_hits_in_order[i] = True
                missed_scores[score] -= 1
                if missed_scores[score] == 0:
                    del missed_scores[score]

            Y = np.cumsum(all_hits_in_order) / k

        elif reward == "smis":
            true_top_k_smis = {smi for smi, _ in true_points}
            all_hits_in_order = np.array(
                [smi in true_top_k_smis for smi, _ in all_points_in_order], dtype=bool
            )
            Y = np.cumsum(all_hits_in_order) / k

        elif reward == "top-k-ave":
            Y = np.zeros(len(all_points_in_order))
            heap = []

            for i, (_, score) in enumerate(all_points_in_order[:k]):
                if score is not None:
                    heapq.heappush(heap, score)
                top_k_avg = sum(heap) / k
                Y[i] = top_k_avg
            Y[:k] = top_k_avg

            for i, (_, score) in enumerate(all_points_in_order[k:]):
                if score is not None:
                    heapq.heappushpop(heap, score)

                top_k_avg = sum(heap) / k
                Y[i + k] = top_k_avg

        elif reward == "total-ave":
            _, ys = zip(*all_points_in_order)
            Y = np.array(ys)
            Y = np.nan_to_num(Y)
            N = np.arange(len(Y)) + 1
            Y = np.cumsum(Y) / N

        else:
            raise ValueError

        return Y

    def calculate_reward(
        self,
        i: int,
        true_points: List[Point],
        is_sorted: bool = False,
        maximize: bool = True,
        avg: bool = True,
        smis: bool = True,
        scores: bool = True,
    ) -> Tuple[float, float, float]:
        """calculate the reward for iteration i

        Parameters
        ----------
        i : int
            the iteration to calculate the reward for
        true_points : List[Point]
            the true top-N points
        is_sorted : bool, default=True
            whether the true points are sorted by objective value=
        avg : bool, default=True
            whether to calculate the average reward=
        smis : bool, default=True
            whether to calcualte the SMILES reward=
        scores : bool, default=True
            whether to calcualte the scores reward

        Returns
        -------
        f_avg : float
            the fraction of the true top-k average score
        f_smis : float
            the fraction of the true top-k SMILES identified
        f_scores : float
            the fraction of the true top-k score identified
        """
        N = len(true_points)
        if not is_sorted:
            true_points = sorted(true_points, key=lambda kv: kv[1], reverse=maximize)

        found = self.get(i, N)

        found_smis, found_scores = zip(*found)
        true_smis, true_scores = zip(*true_points)

        if avg:
            found_avg = np.mean(found_scores)
            true_avg = np.mean(true_scores)
            f_avg = found_avg / true_avg
        else:
            f_avg = None

        if smis:
            found_smis = set(found_smis)
            true_smis = set(true_smis)
            correct_smis = len(found_smis & true_smis)
            f_smis = correct_smis / len(true_smis)
        else:
            f_smis = None

        if scores:
            missed_scores = Counter(true_scores)
            missed_scores.subtract(found_scores)
            n_missed_scores = sum(max(count, 0) for count in missed_scores.values())
            f_scores = (N - n_missed_scores) / N
        else:
            f_scores = None

        return f_avg, f_smis, f_scores

    def calculate_cluster_fraction(
        self, i: int, true_clusters: Tuple[Set, Set, Set]
    ) -> Tuple[float, float, float]:
        large, mids, singletons = true_clusters
        N = len(large) + len(mids) + len(singletons)

        found = {smi for smi, _ in self.get(i, N)}

        f_large = len(found & large) / len(large)
        f_mids = len(found & mids) / len(mids)
        f_singletons = len(found & singletons) / len(singletons)

        return f_large, f_mids, f_singletons

    @staticmethod
    def read_scores(scores_csv: Union[Path, str]) -> List[Tuple]:
        """read the scores contained in the file located at scores_csv"""
        with open(scores_csv) as fid:
            reader = csv.reader(fid)
            next(reader)

            smis_scores = [
                (row[0], float(row[1])) if row[1] else (row[0], None) for row in reader
            ]

        return smis_scores

    @staticmethod
    def read_config(config_file: str) -> Dict:
        """parse an autogenerated MolPAL config file to a dictionary"""
        with open(config_file) as fid:
            return dict([line.strip().split(" = ") for line in fid])

    @staticmethod
    def boltzmann(xs: Iterable[float]) -> float:
        X = np.array(xs)
        E = np.exp(-X)
        Z = E.sum()

        return (X * E / Z).sum()


class IncompleteExperimentError(Exception):
    pass
