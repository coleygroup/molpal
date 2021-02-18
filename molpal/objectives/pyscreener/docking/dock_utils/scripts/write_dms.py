import sys

from chimera import runCommand, openModels, MSMSModel

infile = sys.argv[1]
probe_radius = sys.argv[2]
dms_filename = sys.argv[3]

runCommand("open %s" % infile)
runCommand("surface probeRadius %s" % probe_radius)

surf = openModels.list(modelTypes=[MSMSModel])[0]
from WriteDMS import writeDMS
writeDMS(surf, dms_filename)