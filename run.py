import signal
import sys
from timeit import default_timer as time

# from molpal.args import gen_args
from molpal import args, Explorer

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
    params = vars(args.gen_args())
    print(f'MolPAL will be run with the following arguments:')
    for k, v in sorted(params.items()):
        print(f'  {k}: {v}')
    print(flush=True)
    
    explorer = Explorer(**params)

    start = time()
    try:
        explorer.run()
    except BaseException:
        state_file = explorer.save()
        print(f'Exception raised! Intemediate state saved to "{state_file}"')
        raise
    stop = time()

    m, s = divmod(stop-start, 60)
    h, m = divmod(int(m), 60)
    d, h = divmod(h, 24)
    print(f'Total time for exploration: {d}d {h}h {m}m {s:0.2f}s')
    print('Thanks for using MolPAL!')

if __name__ == "__main__":
    main()