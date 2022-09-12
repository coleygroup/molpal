from argparse import ArgumentParser
import datetime
import os
import signal
import sys
from timeit import default_timer as time

import ray

from molpal import Explorer
from molpal.cli.args import add_args, clean_and_fix_args


def sigterm_handler(signum, frame):
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)


def main(args):
    print(
        """\
***********************************************************************
*   __    __     ______     __         ______   ______     __         *
*  /\ "-./  \   /\  __ \   /\ \       /\  == \ /\  __ \   /\ \        *
*  \ \ \-./\ \  \ \ \/\ \  \ \ \____  \ \  _-/ \ \  __ \  \ \ \____   *
*   \ \_\ \ \_\  \ \_____\  \ \_____\  \ \_\    \ \_\ \_\  \ \_____\  *
*    \/_/  \/_/   \/_____/   \/_____/   \/_/     \/_/\/_/   \/_____/  *
*                                                                     *
***********************************************************************"""
    )
    print("Welcome to MolPAL!")

    clean_and_fix_args(args)
    params = vars(args)

    print("MolPAL will be run with the following arguments:")
    for k, v in sorted(params.items()):
        print(f"  {k}: {v}")
    print(flush=True)

    try:
        if "ip_head" in os.environ:
            ray.init(os.environ["ip_head"])
        else:
            ray.init("auto")
    except ConnectionError:
        ray.init()

    print("Ray cluster online with resources:")
    print(ray.cluster_resources())
    print(flush=True)

    path = params.pop("output_dir")
    explorer = Explorer(path, **params)

    start = time()
    try:
        explorer.run()
    except BaseException:
        d_chkpts = f"{path}/chkpts"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        state_file = explorer.checkpoint(f"{d_chkpts}/iter_{explorer.iter}_{timestamp}")
        print(f'Exception raised! Intemediate state saved to "{state_file}"')
        raise
    stop = time()

    m, s = divmod(stop - start, 60)
    h, m = divmod(int(m), 60)
    d, h = divmod(h, 24)
    print(f"Total time for exploration: {d}d {h}h {m}m {s:0.2f}s")
    print("Thanks for using MolPAL!")


if __name__ == "__main__":
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args)
