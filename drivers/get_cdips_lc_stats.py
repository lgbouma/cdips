"""
get simple rms vs mag stats for CDIPS LCs. plot them.

NOTE: depends on pipe-trex
"""
import aperturephot as ap

def get_cdips_lc_stats(
    sectornum=6,
    cdipssource_vnum=0.2
):

    lcdirectory = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}/'.
        format(sectornum)
    )
    lcglob = 'cam?_ccd?/*_llc.fits'

    # a cut on OC_MG_FINAL_GaiaRp_lt_16_v0.2.csv to be genfromtxt readable
    catalogfile = (
        '/nfs/phtess1/ar1/TESS/PROJ/lbouma/sourceid_and_photrpmeanmag_v{}.csv'.
        format(cdipssouce_vnum)
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
                              catalogfile, tfalcrequired=False,
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

if __name__ == "__main__":
    get_cdips_lc_stats()
