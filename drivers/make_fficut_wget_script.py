from cdips.vetting.centroid_analysis import make_wget_script
import os, shutil
from glob import glob

sector = 7
tfasrdir = (
    "/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}_TFA_SR".
    format(sector)
)
outdir = (
    "/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_cutouts/sector-{}_TFA_SR".
    format(sector)
)

if not os.path.exists(outdir):
    make_wget_script(tfasrdir, xlen_px=10, ylen_px=10, tesscutvernum=0.1)
    raise Exception('need to manually run wget script')

# To execute, run
#
#     ./wget_the_TCE_cutouts.sh &> wget_log.txt &
#
# from the relevant directory. downloads about 20 cutouts per minute. for say
# 600 cutous, takes half an hour. You probably want to run a "check_wget.sh"
# type script as well.
#
# This DLs zip files. Then:

fficutpaths = glob(os.path.join(outdir, 'astrocut*All'))

for fp in fficutpaths:
    if not os.path.exists(fp+'.zip'):
        shutil.move(fp, fp+'.zip')
        print('append .zip to {}'.format(fp))

#
# Then (manually) extract everything in the directory from shell. Note there
# will be some frames from multiple sectors. This is OK.
#
