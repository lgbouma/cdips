"""
get simple rms vs mag stats for CDIPS LCs. plot them. assess how many all-nan
LCs there are.
supplement the statsfile by matching against Gaia DR2 and CDIPS  catalogs.

usage:

    $ (trex_37) python get_cdips_lc_stats.py |& tee /nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_stats/sector-6/stats_overview_log.txt

NOTE: depends on pipe-trex (--> run in trex_37 environment)
"""
import pandas as pd, numpy as np
import aperturephot as ap
import os, subprocess, shlex
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


def supplement_stats_file(
    cdipssource_vnum=0.2,
    sectornum=6):
    """
      add crossmatching info per line:
      * all gaia mags. also gaia extinction and parallax. (also parallax upper
        and lower bounds).
      * calculated T mag from TICv8 relations
      * all the gaia info (especially teff, rstar, etc if available. but also
        position ra,dec and x,y, for sky-map plots. rstar to then be used when
        applying rstar>5rsun cut in vetting)
      * all the CDIPS catalog info (especially the name of the damn cluster)
      * all the TIC info (the CROWDING metric, the TICID, and the Tmag)
    """

    statsfile = os.path.join(
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_stats',
        'sector-{}'.format(sectornum),
        'cdips_lc_statistics.txt'
    )
    outpath = statsfile.replace('cdips_lc_statistics',
                                'supplemented_cdips_lc_statistics')

    stats = ap.read_stats_file(statsfile, fovcathasgaiaids=True)
    df = pd.DataFrame(stats)
    del stats

    df['lcobj'].to_csv('temp.csv',index=False)

    # run the gaia2read on this list
    if not os.path.exists('temp.txt'):
        gaia2readcmd = "gaia2read --header --extra --idfile temp.csv > temp.txt"
        proc = subprocess.run(shlex.split(gaia2readcmd))

        if proc.returncode != 0:
            # NOTE: this is buggy. runs from terminal, not from script. WHY?
            print('gaia2read cmd failed!!')
            import IPython; IPython.embed()
            assert 0

    # merge statsfile against (most of) gaia dr2
    gdf = pd.read_csv('temp.txt',delim_whitespace=True)

    desiredcols = ['#Gaia-ID[1]', 'RA[deg][2]', 'Dec[deg][3]',
                   'RAError[mas][4]', 'DecError[mas][5]',
                   'Parallax[mas][6]', 'Parallax_error[mas][7]',
                   'PM_RA[mas/yr][8]', 'PM_Dec[mas/year][9]',
                   'PMRA_error[mas/yr][10]', 'PMDec_error[mas/yr][11]',
                   'Ref_Epoch[yr][12]', 'phot_g_mean_mag[20]',
                   'phot_bp_mean_mag[25]', 'phot_rp_mean_mag[30]',
                   'radial_velocity[32]', 'radial_velocity_error[33]',
                   'teff_val[35]', 'teff_percentile_lower[36]',
                   'teff_percentile_upper[37]', 'a_g_val[38]',
                   'a_g_percentile_lower[39]', 'a_g_percentile_upper[40]',
                   'e_bp_min_rp_val[41]',
                   'e_bp_min_rp_percentile_lower[42]',
                   'e_bp_min_rp_percentile_upper[43]', 'radius_val[44]',
                   'radius_percentile_lower[45]',
                   'radius_percentile_upper[46]', 'lum_val[47]',
                   'lum_percentile_lower[48]', 'lum_percentile_upper[49]']

    cgdf = gdf[desiredcols]
    df['lcobj'] = df['lcobj'].astype(np.int64)

    mdf = df.merge(cgdf, how='left', left_on='lcobj', right_on='#Gaia-ID[1]')

    del df, cgdf, gdf

    # merge against CDIPS catalog info
    catalogfile = (
        '/nfs/phtess1/ar1/TESS/PROJ/lbouma/OC_MG_FINAL_GaiaRp_lt_16_v{}.csv'.
        format(cdipssource_vnum)
    )
    cdipsdf = pd.read_csv(catalogfile, sep=';')

    dcols = 'cluster;ext_catalog_name;reference;source_id'
    dcols = dcols.split(';')
    ccdf = cdipsdf[dcols]
    ccdf['source_id'] = ccdf['source_id'].astype(np.int64)

    megadf = mdf.merge(ccdf, how='left', left_on='lcobj', right_on='source_id')

    # finally save
    megadf.to_csv(outpath, index=False, sep=';')
    print('made {}'.format(outpath))

    #os.remove('temp.txt')
    #os.remove('temp.csv')



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
        supplement_stats_file(
            cdipssource_vnum=0.2,
            sectornum=6
        )
    if print_metadata:
        print_metadata_stats(
            sectornum=6
        )
