from __future__ import annotations

from argparse import ArgumentParser, ArgumentError
from collections import defaultdict
import csv
from pathlib import Path
import tarfile
from typing import Iterable
import warnings

from tabulate import tabulate
from tqdm import tqdm


def read_top_k(parent_dir: Path, k: int) -> list[str]:
    path = parent_dir / "data" / "all_explored_final.csv"
    with open(path, "r") as fid:
        reader = csv.reader(fid)
        next(reader)
        smis = [row[0] for _, row in zip(range(k), reader) if row[1]]

    return smis


def build_name_dict(parent_dir: Path, smis: Iterable[str]) -> dict[str, list[str]]:
    nodeId2names = defaultdict(list)
    with open(parent_dir / "extended.csv") as fid:
        reader = csv.reader(fid)
        next(reader)

        for smi, name, node_id, _ in tqdm(reader, "Building name dict", unit="row", leave=False):
            if smi not in smis:
                continue

            nodeId2names[node_id].append(name)

    return nodeId2names


def add_args(parser):
    parser.add_argument("parent_dir", type=Path, help="the root directory of a molpal run")
    parser.add_argument(
        "k",
        type=int,
        help="the number of top-scoring compounds to extract. If there are not at least `k` compounds with scores, then the `<k` scored compounds will be extracted.",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="poses",
        help="the name of the directory under which all output will be placed",
    )


def main(args):
    if not args.parent_dir.is_dir():
        raise ArgumentError(f'"{args.output}" is not a directory!')

    try:
        smis = read_top_k(args.parent_dir, args.k)
        if len(smis) < args.k:
            warnings.warn(f"Only {len(smis)} molecules succeeded. Asked for top-{args.k}")
        d_nodeID_names = build_name_dict(args.parent_dir, smis)
    except FileNotFoundError as e:
        raise ArgumentError(f"arg `parent_dir` is not a properly formatted MolPAL directory! {e}")

    path = args.parent_dir / args.name

    data = []
    for node_id in tqdm(d_nodeID_names, "Searching", unit="tarfile"):
        with tarfile.open(args.parent_dir / f"{node_id}.tar.gz") as tar:
            lig_names_on_node = d_nodeID_names[node_id]
            targets = [
                filename
                for filename in tqdm(tar.getnames(), "IDing targets", unit="file", leave=False)
                if any(lig_name in filename for lig_name in lig_names_on_node)
            ]
            extracted_members = [
                tar.extract(target, path)
                for target in tqdm(targets, "Extracting", unit="file", leave=False)
            ]
            data.append((node_id, len(extracted_members)))

    print(tabulate(data, ["Node ID", "Extracted members"], "pretty", colalign=("left", "right")))


if __name__ == "__main__":
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args)
