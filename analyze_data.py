from collections import Counter, defaultdict
import csv
from itertools import chain
import math
from operator import itemgetter
from pathlib import Path
import pickle
import pprint
import sys
from typing import Dict, Iterable, List, Tuple

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

def read_data(p_data, k):
    with open(p_data) as fid:
        reader = csv.reader(fid); next(reader)

        data = [(row[0], -float(row[1])) for row in reader]
    
    return sorted(data, key=itemgetter(1))[:k]

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

    metrics = set()
    models = set()
    iters = set()
    for child in tqdm(parent_dir.iterdir(), desc='Analyzing runs', unit='run'):
        if not child.is_dir():
            continue

        fields = str(child.stem).split('_')
        _, model, metric, _, repeat, *_ = fields
        # _, model, metric, _, _, repeat, *_ = fields
        
        models.add(model)
        metrics.add(metric)

        p_data = child / 'data'
        for p_iter in tqdm(p_data.iterdir(), desc='Analyzing iterations',
                           leave=False):
            try:
                it = int(p_iter.stem.split('_')[-1])
                iters.add(it)
            except ValueError:
                continue

            found = read_data(p_iter, k)

            results[model][metric][it][repeat] = compare_results(found, true)
            if metric == 'random':
                random[it][repeat] = results[model][metric][it][repeat]

    results = recursive_conversion(results)
    random = recursive_conversion(random)

    d_model_results = {}
    for model in models:
        d_metric_it_rep = results[model]

        d_metric_avg = {}
        d_metric_smis = {}
        d_metric_scores = {}

        for metric in metrics:
            if metric == 'random':
                # the random results are model-independent but have only
                # been run for one model, so a hack is necessary here
                d_it_rep = random
            else:
                d_it_rep = d_metric_it_rep[metric]

            avg_means_sds = []
            smis_means_sds = []
            scores_means_sds = []
            for it in range(len(iters)):
                fractions = zip(*d_it_rep[it].values())
                fs_avgs, fs_smis, fs_scores = fractions

                avg_means_sds.append(mean_and_sd(fs_avgs))
                smis_means_sds.append(mean_and_sd(fs_smis))
                scores_means_sds.append(mean_and_sd(fs_scores))

            avg_means, avg_sds = zip(*avg_means_sds)
            avg_means = list(map(PrettyPct, avg_means))
            avg_sds = list(map(PrettyPct, avg_sds))
            d_metric_avg[metric] = avg_means, avg_sds

            smis_means, smis_sds = zip(*smis_means_sds)
            smis_means = list(map(PrettyPct, smis_means))
            smis_sds = list(map(PrettyPct, smis_sds))
            d_metric_smis[metric] = smis_means, smis_sds

            scores_means, scores_sds = zip(*scores_means_sds)
            scores_means = list(map(PrettyPct, scores_means))
            scores_sds = list(map(PrettyPct, scores_sds))
            d_metric_scores[metric] = scores_means, scores_sds

        d_model_results[model] = {
            'avg': d_metric_avg,
            'smis': d_metric_smis,
            'scores': d_metric_scores
        }

    pprint.pprint(d_model_results, compact=True)

if __name__ == "__main__":
    main()
