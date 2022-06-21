from argparse import ArgumentParser

from molpal.cli import run


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    molpal_parser = subparsers.add_parser("run", help="Run the main MolPAL program")
    run.add_args(molpal_parser)
    molpal_parser.set_defaults(func=run.main)

    args = parser.parse_args()
    func = args.func
    del args.func

    func(args)

if __name__ == "__main__":
    main()
