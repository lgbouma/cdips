import os
from cdips import __path__

DATADIR = os.path.join(os.path.dirname(__path__[0]), 'data')
RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
TESTDATADIR = os.path.join(os.path.dirname(__path__[0]), 'data', 'test_data')

LOCALDIR = os.path.join(os.path.expanduser('~'), 'local', 'cdips')
if not os.path.exists(LOCALDIR):
    # note: will fail if ~/local does not already exist.
    os.mkdir(LOCALDIR)
