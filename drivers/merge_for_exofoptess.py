"""
Usage:

$(cdips) python merge_for_exofoptess.py
"""
#############
## LOGGING ##
#############
import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############
import os, socket, subprocess
from glob import glob
import numpy as np, pandas as pd
from numpy import array as nparr
from scipy.optimize import curve_fit
from copy import deepcopy

from astrobase import imageutils as iu
from astrobase.timeutils import get_epochs_given_midtimes_and_period

from cdips.utils import today_YYYYMMDD
from cdips.utils import find_rvs as fr
from cdips.utils import get_vizier_catalogs as gvc
from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.utils.pipelineutils import save_status, load_status
from cdips.vetting.mcmc_vetting_report_page import make_mcmc_vetting_report_page

##########
# config #
##########

DEBUG = 1

hostname = socket.gethostname()
from cdips.paths import DATADIR, RESULTSDIR
fitdir = os.path.join(RESULTSDIR, 'fit_gold')
exofopdir = os.path.join(DATADIR, 'exoFOP_uploads')

FORMATDICT = {
    'period': 8,
    'period_unc': 8,
    'epoch': 8,
    'epoch_unc': 8,
    'depth': -1,
    'depth_unc': -1,
    'duration': 3,
    'duration_unc': 3,
    'inc': 1,
    'inc_unc': 1,
    'imp': 3,
    'imp_unc': 3,
    'r_planet': 5,
    'r_planet_unc': 5,
    'ar_star': 2,
    'a_rstar_unc': 2,
    'radius': 2,
    'radius_unc': 2,
    'mass': 2,
    'mass_unc': 2,
    'temp': 2,
    'temp_unc': 2,
    'insol': 2,
    'insol_unc': 2,
    'dens': 2,
    'dens_unc': 2,
    'sma': 2,
    'sma_unc': 2,
    'ecc': 2,
    'ecc_unc': 2,
    'arg_peri': 2,
    'arg_peri_unc': 2,
    'time_peri': 2,
    'time_peri_unc': 2,
    'vsa': 2,
    'vsa_unc': 2,
}

COLUMN_ORDER = ['target', 'flag', 'disp', 'period', 'period_unc', 'epoch',
                'epoch_unc', 'depth', 'depth_unc', 'duration', 'duration_unc',
                'inc', 'inc_unc', 'imp', 'imp_unc', 'r_planet', 'r_planet_unc',
                'ar_star', 'a_rstar_unc', 'radius', 'radius_unc', 'mass',
                'mass_unc', 'temp', 'temp_unc', 'insol', 'insol_unc', 'dens',
                'dens_unc', 'sma', 'sma_unc', 'ecc', 'ecc_unc', 'arg_peri',
                'arg_peri_unc', 'time_peri', 'time_peri_unc', 'vsa', 'vsa_unc',
                'tag', 'group', 'prop_period', 'notes', 'source_id']


####################
# helper functions #
####################
def linear_model(xdata, m, b):
    return m*xdata + b

def resolve_quality_metric(lgb, jh):
    # 99% of JH's quality ratings are "good"
    # so use LGB's ratings.
    out = []
    for _l, _j in zip(lgb, jh):
        if 'good' in _l:
            out.append('Qual:A')
        else:
            # maybe
            out.append('Qual:B')

    return out

def resolve_clustermember_metric(lgb, jh, reference_id, cluster):
    # LGB didn't label true CM positives
    out = []
    for _l, _j, _r, _c in zip(lgb, jh, reference_id, cluster):
        # First, identify pre-main-sequence only stars.
        pmsonly = ['Zari2018pms', 'SIMBAD_candYSO', 'Kerr2021']
        is_pmsonly = [c in pmsonly for c in _r.split(',')]
        clusternan = ['NaN', 'nan', 'SIMBAD_candYSO']
        is_clusternan = [c in clusternan for c in str(_c).split(',')]
        if np.all(is_pmsonly) and np.all(is_clusternan):
            out.append('PMS?')
        elif _c == 'Oh2017':
            out.append('Oh2017')
        # Next, classify actual cluster
        elif 'CM' in _j and not 'CM' in _l:
            out.append('CM')
        elif 'pos_Non_CM' in _l:
            out.append('NotCM?')
        elif 'Non_CM' in _l or 'Non_CM' in _j:
            out.append('NotCM')
        else:
            out.append('')
    return out

def resolve_age(mean_age):
    out = []
    for a in mean_age:
        if not np.isfinite(a) or a == 'NaN':
            out.append('')
        else:
            agestr = f'{10**a:.1e}yr'
            out.append(agestr)
    return out

def resolve_offtarget_metric(lgb, jh):
    out = []
    for _l, _j in zip(lgb, jh):
        if 'pos_offtarget' in _l or 'offtarget?' in _j:
            out.append('OffTarget?')
        else:
            out.append('')
    return out

def resolve_rotn_metrics(lgb, jh):
    out = []
    for _l, _j in zip(lgb, jh):
        if 'missingrotation' in _l and 'missingrotation' in _j:
            out.append('MissingRot')
        elif 'missingrotation' in _l or 'missingrotation' in _j:
            out.append('MissingRot?')
        else:
            out.append('')
    return out


########
# main #
########

def main(is_dayspecific_exofop_upload=1, cdipssource_vnum=0.6,
         uploadnamestr='s14_thru_s26_clear_threshold'):
    """
    First, generate the extra vetting report page from MCMC results.
    Then: put together a few useful CSV candidate summaries:

    * bulk uploads to exofop/tess

    * observer info sparse (focus on TICIDs, gaia mags, positions on sky, etc)

    * observer info full (stellar rvs for membership assessment; ephemeris
    information)

    * merge of everything (exoFOP upload, + the subset of gaia information
    useful to observers)

    ----------
    Args:

        is_dayspecific_exofop_upload: if True, reads in the manually-written (from
        google spreadsheet) comments and source_ids, and writes those to a
        special "TO_EXOFOP" csv file.

        uploadnamestr: used as unique identifying string in file names
    """

    #
    # Make MCMC vetting report pages, and append to existing vetting reports.
    #
    dirnames = glob(os.path.join( fitdir, 'Year2', 'hlsp*_llc') )
    for dirname in dirnames:
        make_mcmc_vetting_report_page(dirname)

    #
    # Read in the results from the fits
    #
    paramglob = os.path.join(fitdir, 'Year2', "hlsp_*gaiatwo*_llc/*fitparameters.csv")
    parampaths = glob(paramglob)
    statusglob = os.path.join(fitdir, 'Year2', "hlsp_*gaiatwo*_llc/*.stat")
    statuspaths = glob(statusglob)
    vetclassglob = os.path.join( RESULTSDIR, 'vetting_classifications', 'Sector14_through_Sector26', 'sector*_PCs_MERGED_RATINGS.csv' )
    vetclasspaths = glob(vetclassglob)

    statuses = [dict(load_status(f)['simpletransit']) for f in statuspaths]

    param_df = pd.concat((pd.read_csv(f, sep='|') for f in parampaths))
    vetclass_df = pd.concat((pd.read_csv(f, sep=';') for f in vetclasspaths))

    param_df['param_path'] = parampaths
    param_df['Name'] = [
        "vet_"+os.path.basename(os.path.dirname(p))+".pdf"
        for p in parampaths
    ]

    param_df['notes'] = param_df.notes.apply(
        lambda x:
        x.lstrip('CDIPS: see PDF report. ').replace("Not converged ", "Wrn: ")
    )

    # merge against existing classifications
    mdf = param_df.merge(vetclass_df, how='left', on='Name')

    # ensure all these objects do in fact have classifications
    assert not np.any(pd.isnull(mdf.average_score))

    #
    # select: "average score 2"
    #  OR
    #  * rp + σ_rp < 3.0 R_Jup
    #    AND
    #  * b + σ_b < 0.9 (non-grazing, if at all possible)
    #
    sel = (
        (
            (mdf.average_score == 2)
            &
            ~(mdf.target=='TIC13217340.01')  # mislabel
        )
        |
        (
            (mdf.radius + mdf.radius_unc < 33.62694) # <3.0 Rjup
            &
            (mdf.imp + mdf.imp_unc < 0.9)
        )
    )

    smdf = mdf[sel]

    smdf['source_id'] = smdf['param_path'].apply(
        lambda x:
        str(os.path.dirname(x).split('gaiatwo')[1].split('-')[0].lstrip('0'))
    )

    from cdips.utils.catalogs import get_cdips_catalog
    cdipsdf = get_cdips_catalog(ver=0.6)
    cdipsdf.source_id = cdipsdf.source_id.astype(str)
    smdf = smdf.merge(cdipsdf, how='left', on='source_id')

    # create the comment column.
    qual_col = resolve_quality_metric(smdf.LGB_Tags, smdf.JH_Tags)
    CM_col = resolve_clustermember_metric(smdf.LGB_Tags, smdf.JH_Tags,
                                          smdf.reference_id, smdf.cluster)
    age_col = resolve_age(smdf.mean_age)
    offtarget_col = resolve_offtarget_metric(smdf.LGB_Tags, smdf.JH_Tags)
    rotn_col = resolve_rotn_metrics(smdf.LGB_Tags, smdf.JH_Tags)

    comment_col = [ ','.join([q,c,a,o,r]) for q,c,a,o,r in
                   zip(qual_col, CM_col, age_col, offtarget_col, rotn_col)]
    comment_col = [c.replace(',,',',') for c in comment_col]
    comment_col = [c[:-1] if c.endswith(',') else c for c in comment_col]

    smdf['c0'] = comment_col
    smdf['_temp'] = smdf['c0'] + '. '+ smdf['notes']
    smdf._temp = smdf._temp.apply(
        lambda x: x.replace(',.','.').replace(',,',',')
    )
    smdf._temp = smdf._temp.apply(
        lambda x: x.replace('<R>','mean(R)')
    )
    assert np.all(smdf._temp.apply(lambda x: len(x)) <= 119)

    smdf['notes'] = smdf['_temp']

    outpath = os.path.join(
        fitdir, f"all_{today_YYYYMMDD()}_{uploadnamestr}_mergedfitparams.csv"
    )
    mdf.to_csv(outpath, index=False, sep='|')
    LOGINFO(f'made {outpath}')

    outpath = os.path.join(
        fitdir, f"sel_{today_YYYYMMDD()}_{uploadnamestr}_mergedfitparams.csv"
    )
    smdf.to_csv(outpath, index=False, sep='|')
    LOGINFO(f'made {outpath}')

    status_df = pd.DataFrame(statuses)
    status_df['statuspath'] = statuspaths
    status_df['source_id'] = status_df['statuspath'].apply(
        lambda x:
        str(os.path.dirname(x).split('gaiatwo')[1].split('-')[0].lstrip('0'))
    )

    # NOTE: THIS PROCEDURE WILL GET THE EPOCHS ON MULTI-SECTOR PLANETS LESS
    # PRECISE THAN THEY CAN BE.  THIS IS OK, FOR NOW.  A MULTI-SECTOR SEARCH
    # AND BEST-POSSIBLE EPHEMERIS DERIVATION WOULD BE A GOOD IDEA THOUGH.

    #     #
    #     # Duplicate entries in "to_exofop_df" are multi-sector. Average their
    #     # parameters (really will end up just being durations) across sectors,
    #     # and then remove the duplicate multi-sector rows using the "groupby"
    #     # aggregator. This removes the string-based columns, which we can
    #     # reclaim by a "drop_duplicates" call, since they don't have
    #     # sector-specific information.  Then, assign comments and format as
    #     # appropriate for ExoFop-TESS. Unique tag for the entire upload.
    #     #

    #     to_exofop_df['source_id'] = to_exofop_df['source_id'].astype(str)

    #     mean_val_to_exofop_df = to_exofop_df.groupby('target').mean().reset_index()

    #     string_cols = ['target', 'flag', 'disp', 'tag', 'group', 'notes',
    #                    'source_id']
    #     dup_dropped_str_df = (
    #         to_exofop_df.drop_duplicates(
    #             subset=['target'], keep='first', inplace=False
    #             )[string_cols]
    #     )

    #     out_df = mean_val_to_exofop_df.merge(
    #         dup_dropped_str_df, how='left', on='target'
    #     )

    #     #
    #     # The above procedure got the epochs on multisector planets wrong.
    #     # Determine (t0,P) by fitting a line to entries with >=3 sectors
    #     # instead. For the two-sector case, due to bad covariance matrices,
    #     # just use the newest ephemeris.
    #     #
    #     multisector_df = (
    #         to_exofop_df[to_exofop_df.target.groupby(to_exofop_df.target).
    #                      transform('value_counts') > 1]
    #     )
    #     u_multisector_df = out_df[out_df.target.isin(multisector_df.target)]

    #     # temporarily drop the multisector rows from out_df (they will be
    #     # re-merged)
    #     out_df = out_df.drop(
    #         np.argwhere(out_df.target.isin(multisector_df.target)).flatten(),
    #         axis=0
    #     )

    #     ephem_d = {}
    #     for ix, t in enumerate(np.unique(multisector_df.target)):
    #         sel = (multisector_df.target == t)
    #         tmid = nparr(multisector_df[sel].epoch)
    #         tmid_err = nparr(multisector_df[sel].epoch_unc)
    #         init_period = nparr(multisector_df[sel].period.mean())

    #         E, init_t0 = get_epochs_given_midtimes_and_period(
    #             tmid, init_period, verbose=False
    #         )

    #         popt, pcov = curve_fit(
    #             linear_model, E, tmid, p0=(init_period, init_t0), sigma=tmid_err
    #         )

    #         if np.all(np.isinf(pcov)):
    #             # if least-squares doesn't give good error (i.e., just two
    #             # epochs), take the most recent epoch.
    #             s = np.argmax(tmid)
    #             use_t0 = tmid[s]
    #             use_t0_err = tmid_err[s]
    #             use_period = nparr(multisector_df[sel].period)[s]
    #             use_period_err = nparr(multisector_df[sel].period_unc)[s]

    #         else:
    #             use_t0 = popt[1]
    #             use_t0_err = pcov[1,1]**0.5
    #             use_period = popt[0]
    #             use_period_err = pcov[0,0]**0.5

    #         if DEBUG:
    #             LOGINFO(
    #                 'init tmid {}, tmiderr {}\nperiod {}, perioderr {}'.
    #                 format(tmid, tmid_err, nparr(multisector_df[sel].period),
    #                        nparr(multisector_df[sel].period_unc))
    #             )
    #             LOGINFO(
    #                 'use tmid {}, tmiderr {}\nperiod {}, perioderr {}'.
    #                 format(use_t0, use_t0_err, use_period, use_period_err)
    #             )
    #             LOGINFO(10*'-')

    #         ephem_d[ix] = {
    #             'target': t, 'epoch': use_t0, 'epoch_unc': use_t0_err,
    #             'period': use_period, 'period_unc': use_period_err
    #         }

    #     ephem_df = pd.DataFrame(ephem_d).T

    #     mdf = ephem_df.merge(u_multisector_df, how='left', on='target',
    #                          suffixes=('','_DEPRECATED'))
    #     mdf = mdf.drop([c for c in mdf.columns if 'DEPRECATED' in c],
    #                    axis=1, inplace=False)


    #     temp_df = out_df.append(mdf, ignore_index=True, sort=False)
    #     out_df = temp_df

    #    to_exofop_df = out_df[COLUMN_ORDER]

    #    # to_exofop_df = mdf[COLUMN_ORDER] # special behavior for 2020/02/07 fix
    #    # to_exofop_df['flag'] = 'newparams'

    #    _df = manual_comment_df[
    #        manual_comment_df.source_id.isin(to_exofop_df.source_id)
    #    ]

    #    comments = list(_df['comment'])
    #    # comments = 'Fixed ephemeris bug. (Old epoch was erroneous).' # #2020/02/07

    #    for c in comments:
    #        assert len(c)<=119

    to_exofop_df = deepcopy(smdf)

    to_exofop_df = to_exofop_df.sort_values(by="source_id")

    to_exofop_df['tag'] = (
        '{}_bouma_cdips-v01_00001'.format(today_YYYYMMDD())
    )

    istoi = ~to_exofop_df['target'].astype(str).str.startswith('TIC')
    if np.any(istoi):
        newtargetname = 'TOI'+to_exofop_df[istoi].target.astype(str)
        to_exofop_df.loc[istoi, 'target'] = newtargetname

    outpath = os.path.join(
        exofopdir, "{}_{}_w_sourceid.csv".
        format(today_YYYYMMDD(), uploadnamestr)
    )

    to_exofop_df = to_exofop_df[COLUMN_ORDER]

    to_exofop_df.to_csv(outpath, index=False, sep='|')
    LOGINFO('made {}'.format(outpath))

    to_exofop_df = to_exofop_df.drop(['source_id'], axis=1)

    outpath = os.path.join(
        exofopdir, "params_planet_{}_001.txt".
        format(today_YYYYMMDD())
    )
    for c in ['epoch','epoch_unc','period','period_unc']:
        to_exofop_df[c] = to_exofop_df[c].astype(float)
    to_exofop_df = to_exofop_df.round(FORMATDICT)
    to_exofop_df.to_csv(outpath, index=False, sep='|', header=False)
    LOGINFO('made {}'.format(outpath))

    # manually check these...
    LOGINFO('\n'+42*'='+'\n')
    LOGINFO('\nPeriod uncertainties [minutes]')
    LOGINFO(to_exofop_df['period_unc']*24*60)
    LOGINFO('\nEpoch uncertainties [minutes]')
    LOGINFO(to_exofop_df['epoch_unc']*24*60)
    LOGINFO('\nPlanet radii [Rearth]')
    LOGINFO(smdf[['source_id','radius','radius_unc','notes']])
    LOGINFO('\n'+42*'='+'\n')
    LOGINFO(smdf[['target','source_id','radius','radius_unc', 'notes']][smdf.radius>34])
    LOGINFO('\n'+42*'='+'\n')

    # print summary stats
    N = len(np.unique(to_exofop_df.target))
    LOGINFO(f"New knowledge for {N} targets")
    N = len(np.unique(to_exofop_df[to_exofop_df.flag=='newctoi'].target))
    LOGINFO(f"{N} of the targets are new CTOIs")
    N = len(np.unique(to_exofop_df[(to_exofop_df.flag=='newctoi') &
                                   (to_exofop_df.notes.str.contains('Qual:A'))].target))
    LOGINFO(f"{N} of the targets are new CTOIs and 'A-tier' quality")

    # NOTE: DEPRECATED!  THIS STUFF GOES INTO THE CANDIDATE DATABASE ANYWAY

    # #
    # # above is the format exofop-TESS wants. however it's not particularly
    # # useful for followup. for that, we want: gaia IDs, magnitudes, ra, dec.
    # #
    # gaiaids = list(map(
    #     lambda x: int(
    #         os.path.basename(x).split('gaiatwo')[1].split('-')[0].lstrip('0')
    #     ), parampaths
    # ))

    # lcnames = list(map(
    #     lambda x: os.path.basename(x).replace('_fitparameters.csv','.fits'),
    #     parampaths
    # ))

    # #FIXME FIXME FIXME TODO TODO THIS NEED TO BE REFACTORED...
    # lcdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-*/cam?_ccd?/'
    # lcpaths = [ glob(os.path.join(lcdir,lcn))[0] for lcn in lcnames ]

    # # now get the header values
    # kwlist = ['RA_OBJ','DEC_OBJ','CDIPSREF','CDCLSTER','phot_g_mean_mag',
    #           'phot_bp_mean_mag','phot_rp_mean_mag',
    #           'TESSMAG','Gaia-ID','TICID','TICTEFF','TICRAD','TICMASS']

    # for k in kwlist:
    #     thislist = []
    #     for l in lcpaths:
    #         thislist.append(iu.get_header_keyword(l, k, ext=0))
    #     param_df[k] = np.array(thislist)

    # # now search for stellar RV xmatch
    # res = [fr.get_rv_xmatch(ra, dec, G_mag=gmag, dr2_sourceid=s)
    #        for ra, dec, gmag, s in
    #        zip(list(param_df['RA_OBJ']), list(param_df['DEC_OBJ']),
    #            list(param_df['phot_g_mean_mag']), list(param_df['Gaia-ID']))
    #       ]

    # res = np.array(res)
    # param_df['stellar_rv'] = res[:,0]
    # param_df['stellar_rv_unc'] = res[:,1]
    # param_df['stellar_rv_provenance'] = res[:,2]

    # # make column showing whether there are ESO spectra available
    # res = [fr.wrangle_eso_for_rv_availability(ra, dec)
    #        for ra, dec in
    #        zip(list(param_df['RA_OBJ']), list(param_df['DEC_OBJ']))
    #       ]
    # param_df['eso_rv_availability'] = nparr(res)[:,2]

    # #
    # # try to get cluster RV. first from Soubiran, then from Kharchenko.
    # # to do this, load in CDIPS target catalog. merging the CDCLSTER name
    # # (comma-delimited string) against the target catalog on source identifiers
    # # allows unique cluster name identification, since I already did that,
    # # earlier.
    # #
    # cdips_df = ccl.get_cdips_pub_catalog(ver=cdipssource_vnum)
    # dcols = 'cluster;reference;source_id;unique_cluster_name'
    # ccdf = cdips_df[dcols.split(';')]
    # ccdf['source_id'] = ccdf['source_id'].astype(np.int64)
    # mdf = param_df.merge(ccdf, how='left', left_on='source_id', right_on='source_id')
    # param_df['unique_cluster_name'] = nparr(mdf['unique_cluster_name'])

    # s19 = gvc.get_soubiran_19_rv_table()
    # k13_param = gvc.get_k13_param_table()

    # c_rvs, c_err_rvs, c_rv_nstar, c_rv_prov = [], [], [], []
    # for ix, row in param_df.iterrows():

    #     if row['unique_cluster_name'] in nparr(s19['ID']):
    #         sel = (s19['ID'] == row['unique_cluster_name'])
    #         c_rvs.append(float(s19[sel]['RV'].iloc[0]))
    #         c_err_rvs.append(float(s19[sel]['e_RV'].iloc[0]))
    #         c_rv_nstar.append(int(s19[sel]['Nsele'].iloc[0]))
    #         c_rv_prov.append('Soubiran+19')
    #         continue

    #     elif row['unique_cluster_name'] in nparr(k13_param['Name']):
    #         sel = (k13_param['Name'] == row['unique_cluster_name'])
    #         c_rvs.append(float(k13_param[sel]['RV'].iloc[0]))
    #         c_err_rvs.append(float(k13_param[sel]['e_RV'].iloc[0]))
    #         c_rv_nstar.append(int(k13_param[sel]['o_RV'].iloc[0]))
    #         c_rv_prov.append('Kharchenko+13')
    #         continue

    #     else:
    #         c_rvs.append(np.nan)
    #         c_err_rvs.append(np.nan)
    #         c_rv_nstar.append(np.nan)
    #         c_rv_prov.append('')

    # param_df['cluster_rv'] = c_rvs
    # param_df['cluster_err_rv'] = c_err_rvs
    # param_df['cluster_rv_nstar'] = c_rv_nstar
    # param_df['cluster_rv_provenance'] = c_rv_prov

    # #
    # # finally, begin writing the output
    # #

    # outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
    #            "{}_{}_fitparams_plus_observer_info.csv".
    #            format(today_YYYYMMDD(), uploadnamestr))
    # param_df.to_csv(outpath, index=False, sep='|')
    # LOGINFO('made {}'.format(outpath))

    # #
    # # sparse observer info cut
    # #
    # scols = ['target', 'flag', 'disp','tag', 'group', 'RA_OBJ', 'DEC_OBJ',
    #          'CDIPSREF', 'CDCLSTER', 'phot_g_mean_mag', 'phot_bp_mean_mag',
    #          'phot_rp_mean_mag', 'TICID', 'TESSMAG', 'TICTEFF', 'TICRAD',
    #          'TICMASS', 'Gaia-ID'
    #         ]
    # sparam_df = param_df[scols]

    # outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
    #            "{}_{}_observer_info_sparse.csv".
    #            format(today_YYYYMMDD(), uploadnamestr))
    # sparam_df.to_csv(outpath, index=False, sep='|')
    # LOGINFO('made {}'.format(outpath))

    # #
    # # full observer info cut
    # #
    # scols = ['target', 'flag', 'disp','tag', 'group', 'RA_OBJ', 'DEC_OBJ',
    #          'CDIPSREF', 'CDCLSTER', 'phot_g_mean_mag', 'phot_bp_mean_mag',
    #          'phot_rp_mean_mag', 'TICID', 'TESSMAG', 'TICTEFF', 'TICRAD',
    #          'TICMASS', 'Gaia-ID',
    #          'period', 'period_unc', 'epoch', 'epoch_unc', 'depth',
    #          'depth_unc', 'duration', 'duration_unc', 'radius', 'radius_unc',
    #          'stellar_rv', 'stellar_rv_unc', 'stellar_rv_provenance',
    #          'eso_rv_availability', 'cluster_rv', 'cluster_err_rv',
    #          'cluster_rv_nstar', 'cluster_rv_provenance'
    #         ]
    # sparam_df = param_df[scols]

    # outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
    #            "{}_{}_observer_info_full.csv".
    #            format(today_YYYYMMDD(), uploadnamestr))
    # sparam_df.to_csv(outpath, index=False, sep='|')
    # LOGINFO('made {}'.format(outpath))


if __name__=="__main__":
    main()
