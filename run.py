import signal
import sys
from timeit import default_timer as time

from molpal.args import gen_args
from molpal import Explorer

def sigterm_handler(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)

def main():
    print('''\
*********************************************************************
*  __    __     ______     __         ______   ______     __        *
* /\ "-./  \   /\  __ \   /\ \       /\  == \ /\  __ \   /\ \       *
* \ \ \-./\ \  \ \ \/\ \  \ \ \____  \ \  _-/ \ \  __ \  \ \ \____  *
*  \ \_\ \ \_\  \ \_____\  \ \_____\  \ \_\    \ \_\ \_\  \ \_____\ *
*   \/_/  \/_/   \/_____/   \/_____/   \/_/     \/_/\/_/   \/_____/ *
*********************************************************************''')
    print('Welcome to MolPAL!')
    args = gen_args()
    params = vars(args)
    print(f'MolPAL will be run with the following arguments:')
    for key, value in sorted(params.items()):
        print(f'  {key}: {value}')
    print(flush=True)
    
    explorer = Explorer(**params)

    start = time()
    try:
        explorer.run()
    except BaseException:
        # explorer.save()
        raise
    stop = time()

    elapsed = stop - start
    m, s = divmod(elapsed, 60)
    h, m = divmod(int(m), 60)
    d, h = divmod(h, 24)
    print(f'Total time for exploration: {d}d {h}h {m}m {s:0.2f}s')
    print('Thanks for using MolPAL!')

if __name__ == "__main__":
    main()