from configargparse import ArgumentParser

from molpal.cli import run, extract


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser("run", help="Run the main MolPAL program")
    run.add_args(run_parser)
    run_parser.set_defaults(func=run.main)

    extract_parser = subparsers.add_parser(
        "extract", help="extract output docking files from a MolPAL run"
    )
    extract.add_args(extract_parser)
    extract_parser.set_defaults(func=extract.main)

    args = parser.parse_args()
    func = args.func
    del args.func

    func(args)


if __name__ == "__main__":
    main()
