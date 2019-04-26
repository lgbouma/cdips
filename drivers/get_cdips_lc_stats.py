"""
get simple rms vs mag stats for CDIPS LCs. plot them. assess how many all-nan
LCs there are.

NOTE: depends on pipe-trex (best run in trex_37)
"""
import aperturephot as ap
import os
from glob import glob

def get_cdips_lc_stats(
    sectornum=6,
    cdipssource_vnum=0.2,
    nworkers=32
):

    lcdirectory = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}/'.
        format(sectornum)
    )
    lcglob = 'cam?_ccd?/*_llc.fits'

    # a cut on OC_MG_FINAL_GaiaRp_lt_16_v0.2.csv to be genfromtxt readable
    catalogfile = (
        '/nfs/phtess1/ar1/TESS/PROJ/lbouma/sourceid_and_photrpmeanmag_v{}.csv'.
        format(cdipssource_vnum)
    )

    projdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips'
    statsdir = os.path.join(projdir,
                            'results',
                            'cdips_lc_stats',
                            'sector-{}'.format(sectornum))
    if not os.path.exists(statsdir):
        os.mkdir(statsdir)
    statsfile = os.path.join(statsdir,'cdips_lc_statistics.txt')

    ap.parallel_lc_statistics(lcdirectory, lcglob,
                              catalogfile, tfalcrequired=True,
                              epdlcrequired=False,
                              fitslcnottxt=True,
                              fovcatcols=(0,1), # objectid, magcol to use
                              fovcatmaglabel='GRp', outfile=statsfile,
                              nworkers=nworkers,
                              workerntasks=500, rmcols=None,
                              epcols=None, tfcols=None,
                              rfcols=None, correctioncoeffs=None,
                              sigclip=5.0, fovcathasgaiaids=True)

    ap.plot_stats_file(statsfile, statsdir, 'sector-6 cdips', binned=False,
                       logy=True, logx=False, correctmagsafter=None,
                       rangex=(5.9,16), observatory='tess', fovcathasgaiaids=True,
                       yaxisval='RMS')

def print_metadata_stats(sectornum=6):
    """
    how many LCs?
    how many all nan LCs?
    """

    statsfile = os.path.join(
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_stats',
        'sector-{}'.format(sectornum),
        'cdips_lc_statistics.txt'
    )

    stats = ap.read_stats_file(statsfile, fovcathasgaiaids=True)
    N_lcs = len(stats)

    print('CDIPS LIGHTCURVES STATS FOR SECTOR {}'.format(sectornum))
    print(42*'-')
    print('total N_lcs: {}'.format(N_lcs))

    for apn in [1,2,3]:
        N_nan = len(stats[stats['ndet_tf{}'.format(apn)]==0])
        print('for ap {}, {} ({:.1f}%) are all nan, leaving {} ok lcs'.
              format(apn, N_nan, N_nan/N_lcs*100, N_lcs-N_nan))

    print('\nsanity check: {} TF1 LCs have stdev > 0'.
          format(len(stats[stats['stdev_tf1'] > 0])))


if __name__ == "__main__":

    get_stats=1
    print_metadata=1

    if get_stats:
        get_cdips_lc_stats(
            sectornum=6,
            cdipssource_vnum=0.2,
            nworkers=32
        )
    if print_metadata:
        print_metadata_stats(
            sectornum=6
        )
