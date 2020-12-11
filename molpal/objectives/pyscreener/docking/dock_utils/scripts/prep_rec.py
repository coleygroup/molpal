import sys

from chimera import runCommand, openModels
from DockPrep import prep
from WriteMol2 import writeMol2

infile = sys.argv[1]
outfile = sys.argv[2]

runCommand("open %s" % infile)
models = chimera.openModels.list(modelTypes=[chimera.Molecule])
prep(models)
writeMol2(models, outfile)