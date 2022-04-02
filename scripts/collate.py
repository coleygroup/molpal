from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm


def collate(root, size: str):
    targets = [p.stem for p in (root / size / "full").iterdir()]
    Y_np = np.array(
        [
            np.load(root / size / "full" / f"{target}.npy")
            for target in tqdm(targets, "no pruning", leave=False)
        ]
    )
    Y_p = np.array(
        [
            np.load(root / size / "prune" / f"{target}.npy")
            for target in tqdm(targets, "pruning", leave=False)
        ]
    )
    return Y_np, Y_p, targets


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "root", metavar="ROOT", type=Path, help="the root directory containing all DOCKSTRING runs"
    )
    parser.add_argument(
        "-s", "--sizes", nargs="+", required=True, help="the batch sizes to process"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path("dockstring"),
        type=Path,
        help="the base output filename. Files will be named OUTPUT-SIZE-{Y,targets}.{npy,pkl}",
    )

    args = parser.parse_args()
    for size in tqdm(args.sizes, "sizes", leave=False):
        Y_np, Y_p, targets = collate(args.root, size)
        Y = np.stack((Y_np, Y_p))

        np.save(f"{args.output}-{size}-Y.npy", Y)
        Path(f"{args.output}-{size}-targets.pkl").write_bytes(pickle.dumps(targets))


if __name__ == "__main__":
    main()
