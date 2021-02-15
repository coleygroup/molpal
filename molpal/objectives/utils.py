import os
from pathlib import Path
import shelve
import tempfile

def get_temp_file():
    p_tmp = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    return str(p_tmp)
    # fd, filename = tempfile.mkstemp()
    # os.close(fd)

    # return filename