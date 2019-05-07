from cdips.vetting.centroid_analysis import make_wget_script
import os, shutil
from glob import glob

tfasrdir = "/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6_TFA_SR"
# outdir = "/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_cutouts/sector-6_TFA_SR" # FIXME
outdir = "/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_cutouts/temp/"

if not os.path.exists(outdir):
    make_wget_script(tfasrdir, xlen_px=10, ylen_px=10, tesscutvernum=0.1)

# To execute, run
#
#     ./wget_the_TCE_cutouts.sh &> wget_log.txt &
#
# from the relevant directory. downloads about 20 cutouts per minute. for say
# 600 cutous, takes half an hour.
#
# This DLs zip files. Then:

fficutpaths = glob(os.path.join(outdir, 'astrocut*All'))

for fp in fficutpaths:
    if not os.path.exists(fp+'.zip'):
        shutil.move(fp, fp+'.zip')
        print('append .zip to {}'.format(fp))

#
# Then (manually) extract everything in the directory from shell.
#
