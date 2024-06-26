"""
* get simple rms vs mag stats for CDIPS LCs
* plot them.
* assess how many all-nan LCs there are.
* move allnan light curves to a graveyard directory to collect dust
* supplement the statsfile by matching against Gaia DR2 and CDIPS catalogs.

usage:

    $ (cdips) python -u get_cdips_lc_stats.py |& tee logs/s6_stats_overview_log.txt

NOTE: depends on pipe-trex (--> run in environment with aperturephot on path)
"""
import sys

sys.path.append('/nfs/phtess1/ar1/TESS/PROJ/jhartman/202106_CDIPS/cdips-pipeline')

import pandas as pd, numpy as np
import aperturephot as ap
import os, subprocess, shlex, shutil
from glob import glob
from os.path import join

from cdips.utils import collect_cdips_lightcurves as ccl

def get_cdips_lc_stats(
    sector=6,
    cdipssource_vnum=None,
    nworkers=32,
    overwrite=0,
    filesystem='phtess2'
):

    if filesystem in ['phtess2', 'php1']:
        fs = f"/nfs/{filesystem}"
        projdir = f'{fs}/ar0/TESS/PROJ/lbouma/cdips'
        lcdirectory = f'/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{sector}/'
        catdir = '/nfs/phtess1/ar1/TESS/PROJ/lbouma/'

    elif filesystem in ['wh1', 'wh2']:
        projdir = "/ar1/PROJ/luke/proj/cdips"
        lcdirectory = f'/ar1/PROJ/luke/proj/CDIPS_LCS/sector-{sector}/'
        catdir = '/ar1/local/cdips/catalogs/'

    statsdir = join(projdir, 'results', 'cdips_lc_stats', f'sector-{sector}')
    if not os.path.exists(statsdir): os.mkdir(statsdir)
    statsfile = os.path.join(statsdir,'cdips_lc_statistics.txt')
    if os.path.exists(statsfile) and not overwrite:
        print("found statsfile and not overwrite. skip")
        return

    lcglob = 'cam?_ccd?/*_llc.fits'

    # a cut on OC_MG_FINAL_GaiaRp_lt_16_v0.4.csv to be genfromtxt readable
    catalogfile = join(catdir,
                       f'sourceid_and_photrpmeanmag_v{cdipssource_vnum}.csv' )
    if not os.path.exists(catalogfile):
        if cdipssource_vnum < 0.6:
            cfile = join(catdir,
                         f'OC_MG_FINAL_GaiaRp_lt_16_v{cdipssource_vnum}.csv')
            cdipsdf = pd.read_csv(cfile, sep=';')
        else:
            cfile = join(catdir,
                         f'cdips_targets_v{cdipssource_vnum}_gaiasources_Rplt16_orclose.csv')
            cdipsdf = pd.read_csv(cfile, sep=',')

        outdf = cdipsdf[['source_id','phot_rp_mean_mag']].dropna(axis=0, how='any')

        outdf.to_csv(catalogfile, sep=' ', index=False, header=False)

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

    ap.plot_stats_file(statsfile, statsdir,
                       f'sector-{sector} cdips',
                       binned=False, logy=True, logx=False,
                       correctmagsafter=None, rangex=(5.9,16),
                       observatory='tess', fovcathasgaiaids=True,
                       yaxisval='RMS')

    print('Finished get_cdips_lc_stats!')


def supplement_stats_file(
    cdipssource_vnum=None,
    sector=6,
    filesystem=None):
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

    if filesystem in ['phtess2', 'php1']:
        fs = f"/nfs/{filesystem}"
        projdir = f'{fs}/ar0/TESS/PROJ/lbouma/cdips'
        lcdirectory = f'/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{sector}/'
        catdir = '/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
    elif filesystem in ['wh1', 'wh2']:
        projdir = "/ar1/PROJ/luke/proj/cdips"
        lcdirectory = f'/ar1/PROJ/luke/proj/CDIPS_LCS/sector-{sector}/'
        catdir = '/ar1/local/cdips/catalogs/'

    statsdir = join(projdir, 'results', 'cdips_lc_stats', f'sector-{sector}')
    if not os.path.exists(statsdir): os.mkdir(statsdir)
    statsfile = os.path.join(statsdir,'cdips_lc_statistics.txt')

    outpath = statsfile.replace('cdips_lc_statistics',
                                'supplemented_cdips_lc_statistics')
    outdir = os.path.dirname(outpath)

    stats = ap.read_stats_file(statsfile, fovcathasgaiaids=True)
    df = pd.DataFrame(stats)
    del stats

    lcobjcsv = os.path.join(outdir, 'sector{}_lcobj.csv'.format(sector))
    lcobjtxt = os.path.join(outdir, 'sector{}_lcobj.txt'.format(sector))
    df['lcobj'].to_csv(lcobjcsv, index=False, header=False)

    # run the gaia2read on this list
    if not os.path.exists(lcobjtxt):

        gaia2readcmd = (
            "gaia2read --header --extra --idfile {} > {}".format(
                lcobjcsv, lcobjtxt
            )
        )
        returncode = os.system(gaia2readcmd)

        if returncode != 0:
            raise AssertionError('gaia2read cmd failed!!')
        else:
            print('ran {}'.format(gaia2readcmd))

    # merge statsfile against (most of) gaia dr2
    gdf = pd.read_csv(lcobjtxt, delim_whitespace=True)

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
    if np.all(pd.isnull(mdf['RA[deg][2]'])):
        errmsg = (
            'ERR! probably merging against bad temp files!! check gaia2read '
            'call, perhaps.'
        )
        raise AssertionError(errmsg)

    del df, cgdf, gdf

    # merge against CDIPS catalog info
    cdips_df = ccl.get_cdips_pub_catalog(ver=cdipssource_vnum)

    if cdipssource_vnum < 0.6:
        dcols = (
            'cluster;ext_catalog_name;reference;source_id;unique_cluster_name;logt;logt_provenance;comment'
        )
        dcols = dcols.split(';')
    else:
        dcols = (
            'source_id,ra,dec,parallax,parallax_error,pmra,pmdec,phot_g_mean_mag,phot_rp_mean_mag,phot_bp_mean_mag,cluster,age,mean_age,reference_id,reference_bibcode'
        )
        dcols = dcols.split(',')
    ccdf = cdips_df[dcols]
    ccdf['source_id'] = ccdf['source_id'].astype(np.int64)

    megadf = mdf.merge(ccdf, how='left', left_on='lcobj', right_on='source_id')

    # finally save
    megadf.to_csv(outpath, index=False, sep=';')
    print('made {}'.format(outpath))

    print('Finished supplement_stats_file!')


def print_metadata_stats(sector=6, filesystem=None):
    """
    how many LCs?
    how many all nan LCs?
    """

    assert isinstance(filesystem, str)

    if filesystem in ['phtess2', 'php1']:
        fs = f"/nfs/{filesystem}"
        projdir = f'{fs}/ar0/TESS/PROJ/lbouma/cdips'
        lcdirectory = f'/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{sector}/'
        catdir = '/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
    elif filesystem in ['wh1', 'wh2']:
        projdir = "/ar1/PROJ/luke/proj/cdips"
        lcdirectory = f'/ar1/PROJ/luke/proj/CDIPS_LCS/sector-{sector}/'
        catdir = '/ar1/local/cdips/catalogs/'

    statsdir = join(projdir, 'results', 'cdips_lc_stats', f'sector-{sector}')
    if not os.path.exists(statsdir): os.mkdir(statsdir)
    statsfile = os.path.join(statsdir,'cdips_lc_statistics.txt')

    stats = ap.read_stats_file(statsfile, fovcathasgaiaids=True)
    N_lcs = len(stats)

    print('CDIPS LIGHTCURVES STATS FOR SECTOR {}'.format(sector))
    print(42*'-')
    print('total N_lcs: {}'.format(N_lcs))

    for apn in [1,2,3]:
        N_nan = len(stats[stats['ndet_tf{}'.format(apn)]==0])
        print('for ap {}, {} ({:.1f}%) are all nan, leaving {} ok lcs'.
              format(apn, N_nan, N_nan/N_lcs*100, N_lcs-N_nan))

    print('\nsanity check: {} TF1 LCs have stdev > 0'.
          format(len(stats[stats['stdev_tf1'] > 0])))

    print('Finished print_metadata_stats!')


def move_allnan_lcs(sector=None, cdipsvnum=None, filesystem=None):

    assert isinstance(filesystem, str)

    if filesystem in ['phtess2', 'php1']:
        fs = f"/nfs/{filesystem}"
        projdir = f'{fs}/ar0/TESS/PROJ/lbouma/cdips'
        lcdirectory = f'/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{sector}/'
        catdir = '/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
    elif filesystem in ['wh1', 'wh2']:
        projdir = "/ar1/PROJ/luke/proj/cdips"
        lcdirectory = f'/ar1/PROJ/luke/proj/CDIPS_LCS/sector-{sector}/'
        catdir = '/ar1/local/cdips/catalogs/'

    statsdir = join(projdir, 'results', 'cdips_lc_stats', f'sector-{sector}')
    if not os.path.exists(statsdir): os.mkdir(statsdir)
    statsfile = os.path.join(statsdir,'cdips_lc_statistics.txt')

    stats = ap.read_stats_file(statsfile, fovcathasgaiaids=True)
    N_lcs = len(stats)

    print('CDIPS LIGHTCURVES STATS FOR SECTOR {}'.format(sector))
    print(42*'-')
    print('total N_lcs: {}'.format(N_lcs))

    for apn in [1,2,3]:
        N_nan = len(stats[stats['ndet_rm{}'.format(apn)]==0])
        print('for ap {}, {} ({:.1f}%) are all nan, leaving {} ok lcs'.
              format(apn, N_nan, N_nan/N_lcs*100, N_lcs-N_nan))

    print(42*'-')
    print('BEGINNING MOVE OF ALLNAN LIGHT CURVES')

    sel = (
        (stats['ndet_rm1']==0) &
        (stats['ndet_rm2']==0) &
        (stats['ndet_rm3']==0)
    )
    nanobjs = stats[sel]['lcobj']

    lcdirectory = join(lcdirectory, "cam?_ccd?")

    lcnames = [(
        'hlsp_cdips_tess_ffi_'
        'gaiatwo{zsourceid}-{zsector}-cam{cam}-ccd{ccd}_'
        'tess_v{zcdipsvnum}_llc.fits'
        ).format(
            cam='?',
            ccd='?',
            zsourceid=str(lcgaiaid).zfill(22),
            zsector=str(sector).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )
        for lcgaiaid in nanobjs
    ]

    lcglobs = [os.path.join(lcdirectory, lcname) for lcname in lcnames]

    lcpaths = []
    for l in lcglobs:
        try:
            lcpaths.append(glob(l)[0])
        except:
            pass

    dstpaths = [os.path.join(os.path.dirname(l),
                             'allnanlcs',
                             os.path.basename(l))
                for l in lcpaths]

    for src,dst in zip(lcpaths,dstpaths):
        dstdir = os.path.dirname(dst)
        if not os.path.exists(dstdir):
            os.mkdir(dstdir)
        try:
            shutil.move(src,dst)
            print('moved {} -> {}'.format(src,dst))
        except FileNotFoundError as e:
            if os.path.exists(dst):
                pass
            else:
                print(repr(e))
                raise FileNotFoundError


def main(sector, cdipssource_vnum, cdipsvnum, overwrite, get_stats=1,
         make_supp_stats=0, print_metadata=1, move_allnan=1, filesystem=None):

    assert isinstance(filesystem, str)

    if get_stats:
        get_cdips_lc_stats(
            sector=sector,
            cdipssource_vnum=cdipssource_vnum,
            nworkers=40,
            overwrite=overwrite,
            filesystem=filesystem
        )
    if make_supp_stats:
        supplement_stats_file(
            cdipssource_vnum=cdipssource_vnum,
            sector=sector,
            filesystem=filesystem
        )
    if print_metadata:
        print_metadata_stats(
            sector=sector,
            filesystem=filesystem
        )
    if move_allnan:
        move_allnan_lcs(
            sector=sector, cdipsvnum=cdipsvnum, filesystem=filesystem
        )


if __name__ == "__main__":

    sector=40
    cdipssource_vnum=0.6
    cdipsvnum=1
    overwrite=0
    get_stats=0
    make_supp_stats=1
    print_metadata=0
    move_allnan=1
    filesystem='wh1'

    main(sector, cdipssource_vnum, cdipsvnum, overwrite, get_stats=get_stats,
         make_supp_stats=make_supp_stats, print_metadata=print_metadata,
         move_allnan=move_allnan, filesystem=filesystem)
