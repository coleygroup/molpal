from collections import Counter, defaultdict
import csv
from itertools import chain, islice
import math
from operator import itemgetter
from pathlib import Path
import pickle
import pprint
import sys
from typing import Dict, Iterable, List, Set, Tuple

from tqdm import tqdm

class PrettyPct(float):
    def __init__(self, x):
        self.x = 100*x

    def __repr__(self):
        if self.x == 0:
            return '0.000'
        elif self.x > 0.001:
            return f'{self.x:0.3f}'
        else:
            return f'{self.x:0.1e}'

def boltzmann(xs: Iterable[float]) -> float:
    Z = sum(math.exp(-x) for x in xs)
    return sum(x * math.exp(-x) / Z for x in xs)

def mean(xs: Iterable[float]) -> float:
    return sum(x for x in xs) / len(xs)

def var(xs: Iterable[float], x_mean: float) -> float:
    return sum((x-x_mean)**2 for x in xs) / len(xs)

def mean_and_sd(xs: Iterable[float]) -> Tuple[float, float]:
    x_mean = mean(xs)
    return x_mean, math.sqrt(var(xs, x_mean))

def recursive_conversion(nested_dict):
    if not isinstance(nested_dict, defaultdict):
        return nested_dict

    for k in nested_dict:
        # print(k)
        sub_dict = nested_dict[k]
        nested_dict[k] = recursive_conversion(sub_dict)
    return dict(nested_dict)

def read_data(p_data, k) -> List[Tuple]:
    with open(p_data) as fid:
        reader = csv.reader(fid); next(reader)
        # the data files are always sorted
        data = [(row[0], -float(row[1])) for row in islice(reader, 2*k)]
    
    return sorted(data, key=itemgetter(1))[:k]

def get_smis_from_data(p_data) -> Set:
    with open(p_data) as fid:
        reader = csv.reader(fid); next(reader)
        smis = {row[0] for row in reader}
    
    return smis

def compare_results(found: List[Tuple], true: List[Tuple],
                    avg: bool = True, smis: bool = True, scores: bool = True
                    ) -> Tuple[float, float, float]:
    k = len(found)
    found_smis, found_scores = zip(*found)
    true_smis, true_scores = zip(*true)

    if avg:
        found_avg = mean(found_scores)
        true_avg = mean(true_scores)
        f_avg = found_avg / true_avg
    else:
        f_avg = None

    # if boltzmann:
    #     found_boltzmann = boltzmann(found_scores)
    #     true_boltzmann = boltzmann(true_scores)
    #     f_correct_boltzmann = found_boltzmann / true_boltzmann
    # else:
    #     f_correct_boltzmann = None

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
        n_missed_scores = sum(
            count if count > 0 else 0
            for count in missed_scores.values()
        )
        f_scores = (k - n_missed_scores) / k
    else:
        f_scores = None

    return f_avg, f_smis, f_scores

def calculate_metric_results(it_rep_results):
    avg_means_sds = [[]] * len(it_rep_results)
    smis_means_sds = [[]] * len(it_rep_results)
    scores_means_sds = [[]] * len(it_rep_results)
    for i in it_rep_results:
        rep_results = zip(*it_rep_results[i].values())
        fs_avgs, fs_smis, fs_scores = rep_results

        avg_means_sds[i] = (mean_and_sd(fs_avgs))
        smis_means_sds[i] = (mean_and_sd(fs_smis))
        scores_means_sds[i] = (mean_and_sd(fs_scores))

    avg_means, avg_sds = zip(*avg_means_sds)
    avg_means = list(map(PrettyPct, avg_means))
    avg_sds = list(map(PrettyPct, avg_sds))

    smis_means, smis_sds = zip(*smis_means_sds)
    smis_means = list(map(PrettyPct, smis_means))
    smis_sds = list(map(PrettyPct, smis_sds))

    scores_means, scores_sds = zip(*scores_means_sds)
    scores_means = list(map(PrettyPct, scores_means))
    scores_sds = list(map(PrettyPct, scores_sds))

    return ((avg_means, avg_sds),
            (smis_means, smis_sds),
            (scores_means, scores_sds))

def main():
    true = pickle.load(open(sys.argv[1], 'rb'))
    parent_dir = Path(sys.argv[2])
    k = int(sys.argv[3])

    try:
        true = sorted(true.items(), key=itemgetter(1))[:k]
    except AttributeError:
        true = sorted(true, key=itemgetter(1))[:k]

    nested_dict = lambda: defaultdict(nested_dict)
    results = nested_dict()
    random = nested_dict()
    common_smis = nested_dict()

    for child in tqdm(parent_dir.iterdir(), desc='Analyzing runs', unit='run'):
        if not child.is_dir():
            continue

        fields = str(child.stem).split('_')
        _, model, metric, _, rep, *_ = fields
        rep = int(rep)
        # _, model, metric, _, _, repeat, *_ = fields
        
        p_data = child / 'data'
        for p_iter in tqdm(p_data.iterdir(), desc='Analyzing iterations',
                           leave=False):
            try:
                it = int(p_iter.stem.split('_')[-1])
            except ValueError:
                continue

            found = read_data(p_iter, k)
            if metric == 'random':
                random[it][rep] = compare_results(found, true)
            else:
                results[model][metric][it][rep] = compare_results(found, true)
                common_smis[model][metric][it][rep] = {smi for smi, _ in found}

    d_model_metric_it_rep_results = recursive_conversion(results)
    random = recursive_conversion(random)
    common_smis = recursive_conversion(common_smis)

    # pprint.pprint(d_model_metric_it_rep_results)

    d_model_results = {}
    for model in d_model_metric_it_rep_results:
        d_metric_it_rep_results = d_model_metric_it_rep_results[model]

        d_metric_avg = {}
        d_metric_smis = {}
        d_metric_scores = {}

        for metric in d_metric_it_rep_results:
            d_it_rep_results = d_metric_it_rep_results[metric]

            avg, smis, scores = calculate_metric_results(d_it_rep_results)

            d_metric_avg[metric] = avg[0], avg[1]
            d_metric_smis[metric] = smis[0], smis[1]
            d_metric_scores[metric] = scores[0], scores[1]

        d_model_results[model] = {
            'avg': d_metric_avg,
            'smis': d_metric_smis,
            'scores': d_metric_scores
        }
    pprint.pprint(d_model_results, compact=True)

    if random:
        avg, smis, scores = calculate_metric_results(random)
        random = {
            'avg': (avg[0], avg[1]),
            'smis': (smis[0], smis[1]),
            'scores': (scores[0], scores[1])
        }
        pprint.pprint(random, compact=True)

    # common_smis = nested_dict()
    # total_smis = nested_dict()

    # for model in common_smis:
    #     for metric in common_smis[model]:
    #         for it in common_smis[model][metric]:
    #             smis_sets = list(common_smis[model][metric][it].values())
    #             # print(smis_sets)[0]
    #             common_smis[model][metric][it] = len(set.intersection(
    #                 *smis_sets
    #             ))
    #             total_smis[model][metric][it] = len(set.union(
    #                 *smis_sets
    #             ))
    
    # pprint.pprint(recursive_conversion(common_smis), compact=True)
    # pprint.pprint(recursive_conversion(total_smis), compact=True)
    
def main_2():
    parent_dir = Path(sys.argv[2])
    k = int(sys.argv[3])

    nested_dict = lambda: defaultdict(nested_dict)
    common_smis = nested_dict()

    for child in tqdm(parent_dir.iterdir(), desc='Analyzing runs', unit='run'):
        if not child.is_dir():
            continue

        fields = str(child.stem).split('_')
        _, model, metric, _, rep, *_ = fields
        rep = int(rep)
        
        p_data = child / 'data'
        for p_iter in tqdm(p_data.iterdir(), desc='Analyzing iterations',
                           leave=False):
            try:
                it = int(p_iter.stem.split('_')[-1])
            except ValueError:
                continue

            found = read_data(p_iter, k)
            if metric == 'random':
                pass
                # random[it][rep] = compare_results(found, true)
            else:
                common_smis[model][metric][it][rep] = {smi for smi, _ in found}

    common_smis = recursive_conversion(common_smis)
    for model in common_smis:
        for metric in common_smis[model]:
            n_smis_by_iter = []
            for it in common_smis[model][metric]:
                smis_sets = list(common_smis[model][metric][it].values())
                n_smis_by_iter.append(len(set.intersection(*smis_sets)))
            common_smis[model][metric] = n_smis_by_iter
    pprint.pprint(common_smis, compact=True)

def main_3():
    parent_dir = Path(sys.argv[2])

    nested_dict = lambda: defaultdict(nested_dict)
    total_smis = nested_dict()

    for child in tqdm(parent_dir.iterdir(), desc='Analyzing runs', unit='run'):
        if not child.is_dir():
            continue

        fields = str(child.stem).split('_')
        _, model, metric, _, rep, *_ = fields
        rep = int(rep)
        
        p_data = child / 'data'
        for p_iter in tqdm(p_data.iterdir(), desc='Analyzing iterations',
                           leave=False):
            try:
                it = int(p_iter.stem.split('_')[-1])
            except ValueError:
                continue

            if metric == 'random':
                pass
                # random[it][rep] = compare_results(found, true)
            else:
                total_smis[model][metric][it][rep] = get_smis_from_data(p_iter)

    total_smis = recursive_conversion(total_smis)
    for model in total_smis:
        for metric in total_smis[model]:
            n_smis_by_iter = []
            for it in range(6):#total_smis[model][metric]:
                smis_sets = list(total_smis[model][metric][it].values())
                n_smis_by_iter.append(len(set.union(*smis_sets)))
            total_smis[model][metric] = n_smis_by_iter

    pprint.pprint(total_smis, compact=True)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        main()
        exit()
    if sys.argv[4] == 'intersection':
        main_2()
        exit()
    if sys.argv[4] == 'union':
        main_3()
        exit()