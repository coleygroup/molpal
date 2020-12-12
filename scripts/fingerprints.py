import argparse
import gzip
from pathlib import Path

from molpal import encoder
from molpal.pools import fingerprints

parser.add_argument('--path',
                    help='the path under which to write the fingerprints file')
parser.add_argument('-nc', '--ncpu', default=1, type=int, metavar='N_CPU',
                    help='the number of cores to available to each worker/job/process/node. If performing docking, this is also the number of cores multithreaded docking programs will utilize.')
parser.add_argument('--fingerprint', default='pair',
                    choices={'morgan', 'rdkit', 'pair', 'maccs', 'map4'},
                    help='the type of encoder to use')
parser.add_argument('--radius', type=int, default=2,
                    help='the radius or path length to use for fingerprints')
parser.add_argument('--length', type=int, default=2048,
                    help='the length of the fingerprint')

parser.add_argument('--library', required=True, metavar='LIBRARY_FILEPATH',
                        help='the file containing members of the MoleculePool')
parser.add_argument('--no-title-line', action='store_true', default=False,
                    help='whether there is no title line in the library file')
parser.add_argument('--delimiter', default=',',
                    help='the column separator in the library file')
parser.add_argument('--smiles-col', default=0, type=int,
                    help='the column containing the SMILES string in the library file')
                    
def main():
    args = parser.parse_args()
    args.title_line = not args.no_title_line
    
    encoder = encoder.Encoder(fingerprint=args.fingerprint, radius=args.radius,
                              length=args.length)
    if Path(args.library).suffix == '.gz':
        open_ = partial(gzip.open, mode='rt')
    else:
        open_ = open

    print('Precalculating feature matrix ...', end=' ')
    with open_(args.library) as fid:
        reader = csv.reader(fid, delimiter=args..delimiter)

        total_size = sum(1 for _ in reader); reader.seek(0)
        if args.title_line:
            total_size -=1; next(reader)

        smis = (row[args.smiles_col] for row in reader)
        fps, invalid_lines = fingerprints.feature_matrix_hdf5(
            smis, total_size, ncpu=args.ncpu,
            encoder=encoder, name=Path(args.library).stem, path=args.path
        )

    print('Done!')
    print(f'Feature matrix was saved to "{fps}"', flush=True)

    if len(invalid_lines) == 0:
        print('Detected no invalid lines! When using this fingerprints file,',
              'you can pass the --validated flag to MolPAL to speed up pool',
              'construction.')

if __name__ == "__main__":
    pass