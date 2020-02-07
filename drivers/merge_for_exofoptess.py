"""
Usage:

$(cdips) python merge_for_exofoptess.py
"""
import os, socket
from glob import glob
import numpy as np, pandas as pd
from numpy import array as nparr
from scipy.optimize import curve_fit

from astrobase import imageutils as iu
from astrobase.timeutils import get_epochs_given_midtimes_and_period

from cdips.utils import today_YYYYMMDD
from cdips.utils import find_rvs as fr
from cdips.utils import get_vizier_catalogs as gvc
from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.utils.pipelineutils import save_status, load_status

##########
# config #
##########

DEBUG = 1

hostname = socket.gethostname()
if 'phtess1' in hostname or 'phtess2' in hostname:
    fitdir = "/home/lbouma/proj/cdips/results/fit_gold"
    exofopdir = "/home/lbouma/proj/cdips/data/exoFOP_uploads"
elif 'brik' in hostname:
    fitdir = "/home/luke/Dropbox/proj/cdips/results/fit_gold"
    exofopdir = "/home/luke/Dropbox/proj/cdips/data/exoFOP_uploads"
else:
    raise ValueError('where is fit_gold directory on {}?'.format(hostname))

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

########
# main #
########

def main(is_dayspecific_exofop_upload=1, cdipssource_vnum=0.4,
         uploadnamestr='sectors_8_thru_11_clear_threshold'):
    """
    Put together a few useful CSV candidate summaries:

    * bulk uploads to exofop/tess (targets, planet and stellar parameters and
    uncertainties)

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
    # Read in the results from the fits
    #
    paramglob = os.path.join(
        fitdir, "sector-*_CLEAR_THRESHOLD/fitresults/hlsp_*gaiatwo*_llc/*fitparameters.csv"
    )
    parampaths = glob(paramglob)
    statusglob = os.path.join(
        fitdir, "sector-*_CLEAR_THRESHOLD/fitresults/hlsp_*gaiatwo*_llc/*.stat"
    )
    statuspaths = glob(statusglob)

    statuses = [dict(load_status(f)['fivetransitparam_fit'])
                for f in statuspaths]

    param_df = pd.concat((pd.read_csv(f, sep='|') for f in parampaths))

    outpath = os.path.join(
        fitdir, "{}_{}_mergedfitparams.csv".
        format(today_YYYYMMDD(), uploadnamestr)
    )
    param_df['param_path'] = parampaths
    param_df.to_csv(outpath, index=False, sep='|')
    print('made {}'.format(outpath))

    status_df = pd.DataFrame(statuses)

    status_df['statuspath'] = statuspaths

    status_gaiaids = list(map(
        lambda x: int(
            os.path.dirname(x).split('gaiatwo')[1].split('-')[0].lstrip('0')
        ), statuspaths
    ))

    status_df['source_id'] = status_gaiaids

    if is_dayspecific_exofop_upload:

        #
        # Manually commented candidates are the only ones we're uploading.
        #
        manual_comment_df = pd.read_csv(
            '../data/exoFOP_uploads/{}_cdips_candidate_upload.csv'.
            format(today_YYYYMMDD()), sep=","
        )
        common = status_df.merge(manual_comment_df,
                                 on='source_id',
                                 how='inner')
        sel_status_df = status_df[status_df.source_id.isin(common.source_id)]

        #
        # WARN: the MCMC fits should have converged before uploading.
        # (20190918 had two exceptions, where the fit looked fine.)
        #
        if len(sel_status_df[sel_status_df['is_converged']=='False'])>0:

            print('\nWRN! THE FOLLOWING CANDIDATES ARE NOT CONVERGED')
            print(sel_status_df[sel_status_df['is_converged']=='False'])

        param_gaiaids = list(map(
            lambda x: int(
                os.path.basename(x).split('gaiatwo')[1].split('-')[0].lstrip('0')
            ), parampaths
        ))
        param_df['source_id'] = param_gaiaids

        #
        # Require that you actually have a parameter file (...).
        #
        _df = sel_status_df.merge(param_df, on='source_id', how='inner')

        to_exofop_df = param_df[param_df.source_id.isin(_df.source_id)]

        if len(to_exofop_df) != len(manual_comment_df):

            print('\nWRN! {} CANDIDATES DID NOT HAVE PARAMETERS'.format(
                len(manual_comment_df) - len(to_exofop_df)
            ))
            print('They are...')
            print(
                manual_comment_df[
                    ~manual_comment_df.source_id.isin(to_exofop_df.source_id)
                ]
            )
            print('\n')

        #
        # Duplicate entries in "to_exofop_df" are multi-sector. Average their
        # parameters (really will end up just being durations) across sectors,
        # and then remove the duplicate multi-sector rows using the "groupby"
        # aggregator. This removes the string-based columns, which we can
        # reclaim by a "drop_duplicates" call, since they don't have
        # sector-specific information.  Then, assign comments and format as
        # appropriate for ExoFop-TESS. Unique tag for the entire upload.
        #

        to_exofop_df['source_id'] = to_exofop_df['source_id'].astype(str)

        mean_val_to_exofop_df = to_exofop_df.groupby('target').mean().reset_index()

        string_cols = ['target', 'flag', 'disp', 'tag', 'group', 'notes',
                       'source_id']
        dup_dropped_str_df = (
            to_exofop_df.drop_duplicates(
                subset=['target'], keep='first', inplace=False
                )[string_cols]
        )

        out_df = mean_val_to_exofop_df.merge(
            dup_dropped_str_df, how='left', on='target'
        )

        #
        # The above procedure got the epochs on multisector planets wrong.
        # Determine (t0,P) by fitting a line to entries with >=3 sectors
        # instead. For the two-sector case, due to bad covariance matrices,
        # just use the newest ephemeris.
        #
        multisector_df = (
            to_exofop_df[to_exofop_df.target.groupby(to_exofop_df.target).
                         transform('value_counts') > 1]
        )
        u_multisector_df = out_df[out_df.target.isin(multisector_df.target)]

        # temporarily drop the multisector rows from out_df (they will be
        # re-merged)
        out_df = out_df.drop(
            np.argwhere(out_df.target.isin(multisector_df.target)).flatten(),
            axis=0
        )

        ephem_d = {}
        for ix, t in enumerate(np.unique(multisector_df.target)):
            sel = (multisector_df.target == t)
            tmid = nparr(multisector_df[sel].epoch)
            tmid_err = nparr(multisector_df[sel].epoch_unc)
            init_period = nparr(multisector_df[sel].period.mean())

            E, init_t0 = get_epochs_given_midtimes_and_period(
                tmid, init_period, verbose=False
            )

            popt, pcov = curve_fit(
                linear_model, E, tmid, p0=(init_period, init_t0), sigma=tmid_err
            )

            if np.all(np.isinf(pcov)):
                # if least-squares doesn't give good error (i.e., just two
                # epochs), take the most recent epoch.
                s = np.argmax(tmid)
                use_t0 = tmid[s]
                use_t0_err = tmid_err[s]
                use_period = nparr(multisector_df[sel].period)[s]
                use_period_err = nparr(multisector_df[sel].period_unc)[s]

            else:
                use_t0 = popt[1]
                use_t0_err = pcov[1,1]**0.5
                use_period = popt[0]
                use_period_err = pcov[0,0]**0.5

            if DEBUG:
                print(
                    'init tmid {}, tmiderr {}\nperiod {}, perioderr {}'.
                    format(tmid, tmid_err, nparr(multisector_df[sel].period),
                           nparr(multisector_df[sel].period_unc))
                )
                print(
                    'use tmid {}, tmiderr {}\nperiod {}, perioderr {}'.
                    format(use_t0, use_t0_err, use_period, use_period_err)
                )
                print(10*'-')

            ephem_d[ix] = {
                'target': t, 'epoch': use_t0, 'epoch_unc': use_t0_err,
                'period': use_period, 'period_unc': use_period_err
            }

        ephem_df = pd.DataFrame(ephem_d).T

        mdf = ephem_df.merge(u_multisector_df, how='left', on='target',
                             suffixes=('','_DEPRECATED'))
        mdf = mdf.drop([c for c in mdf.columns if 'DEPRECATED' in c],
                       axis=1, inplace=False)


        temp_df = out_df.append(mdf, ignore_index=True, sort=False)
        out_df = temp_df

        to_exofop_df = out_df[COLUMN_ORDER]

        # to_exofop_df = mdf[COLUMN_ORDER] # special behavior for 2020/02/07 fix
        # to_exofop_df['flag'] = 'newparams'

        _df = manual_comment_df[
            manual_comment_df.source_id.isin(to_exofop_df.source_id)
        ]

        comments = list(_df['comment'])
        # comments = 'Fixed ephemeris bug. (Old epoch was erroneous).' # #2020/02/07

        for c in comments:
            assert len(c)<=119

        to_exofop_df = to_exofop_df.sort_values(by="source_id")
        _df = _df.sort_values(by="source_id")

        to_exofop_df['notes'] = comments
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
        to_exofop_df.to_csv(outpath, index=False, sep='|')
        print('made {}'.format(outpath))

        to_exofop_df = to_exofop_df.drop(['source_id'], axis=1)

        outpath = os.path.join(
            exofopdir, "params_planet_{}_001.txt".
            format(today_YYYYMMDD())
        )
        for c in ['epoch','epoch_unc','period','period_unc']:
            to_exofop_df[c] = to_exofop_df[c].astype(float)
        to_exofop_df = to_exofop_df.round(FORMATDICT)
        to_exofop_df['depth'] = to_exofop_df['depth'].astype(int)
        to_exofop_df['depth_unc'] = to_exofop_df['depth_unc'].astype(int)
        to_exofop_df.to_csv(outpath, index=False, sep='|', header=False)
        print('made {}'.format(outpath))

        # manually check these...
        print('\n'+42*'='+'\n')
        print('\nPeriod uncertainties [minutes]')
        print(to_exofop_df['period_unc']*24*60)
        print('\nEpoch uncertainties [minutes]')
        print(to_exofop_df['epoch_unc']*24*60)
        print('\nPlanet radii [Rearth]')
        print(to_exofop_df[['radius','radius_unc','notes']])
        print('\n'+42*'='+'\n')

    #
    # above is the format exofop-TESS wants. however it's not particularly
    # useful for followup. for that, we want: gaia IDs, magnitudes, ra, dec.
    #
    gaiaids = list(map(
        lambda x: int(
            os.path.basename(x).split('gaiatwo')[1].split('-')[0].lstrip('0')
        ), parampaths
    ))

    lcnames = list(map(
        lambda x: os.path.basename(x).replace('_fitparameters.csv','.fits'),
        parampaths
    ))

    lcdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-*/cam?_ccd?/'
    lcpaths = [ glob(os.path.join(lcdir,lcn))[0] for lcn in lcnames ]

    # now get the header values
    kwlist = ['RA_OBJ','DEC_OBJ','CDIPSREF','CDCLSTER','phot_g_mean_mag',
              'phot_bp_mean_mag','phot_rp_mean_mag',
              'TESSMAG','Gaia-ID','TICID','TICTEFF','TICRAD','TICMASS']

    for k in kwlist:
        thislist = []
        for l in lcpaths:
            thislist.append(iu.get_header_keyword(l, k, ext=0))
        param_df[k] = np.array(thislist)

    # now search for stellar RV xmatch
    res = [fr.get_rv_xmatch(ra, dec, G_mag=gmag, dr2_sourceid=s)
           for ra, dec, gmag, s in
           zip(list(param_df['RA_OBJ']), list(param_df['DEC_OBJ']),
               list(param_df['phot_g_mean_mag']), list(param_df['Gaia-ID']))
          ]

    res = np.array(res)
    param_df['stellar_rv'] = res[:,0]
    param_df['stellar_rv_unc'] = res[:,1]
    param_df['stellar_rv_provenance'] = res[:,2]

    # make column showing whether there are ESO spectra available
    res = [fr.wrangle_eso_for_rv_availability(ra, dec)
           for ra, dec in
           zip(list(param_df['RA_OBJ']), list(param_df['DEC_OBJ']))
          ]
    param_df['eso_rv_availability'] = nparr(res)[:,2]

    #
    # try to get cluster RV. first from Soubiran, then from Kharchenko.
    # to do this, load in CDIPS target catalog. merging the CDCLSTER name
    # (comma-delimited string) against the target catalog on source identifiers
    # allows unique cluster name identification, since I already did that,
    # earlier.
    #
    cdips_df = ccl.get_cdips_pub_catalog(ver=cdipssource_vnum)
    dcols = 'cluster;reference;source_id;unique_cluster_name'
    ccdf = cdips_df[dcols.split(';')]
    ccdf['source_id'] = ccdf['source_id'].astype(np.int64)
    mdf = param_df.merge(ccdf, how='left', left_on='source_id', right_on='source_id')
    param_df['unique_cluster_name'] = nparr(mdf['unique_cluster_name'])

    s19 = gvc.get_soubiran_19_rv_table()
    k13_param = gvc.get_k13_param_table()

    c_rvs, c_err_rvs, c_rv_nstar, c_rv_prov = [], [], [], []
    for ix, row in param_df.iterrows():

        if row['unique_cluster_name'] in nparr(s19['ID']):
            sel = (s19['ID'] == row['unique_cluster_name'])
            c_rvs.append(float(s19[sel]['RV'].iloc[0]))
            c_err_rvs.append(float(s19[sel]['e_RV'].iloc[0]))
            c_rv_nstar.append(int(s19[sel]['Nsele'].iloc[0]))
            c_rv_prov.append('Soubiran+19')
            continue

        elif row['unique_cluster_name'] in nparr(k13_param['Name']):
            sel = (k13_param['Name'] == row['unique_cluster_name'])
            c_rvs.append(float(k13_param[sel]['RV'].iloc[0]))
            c_err_rvs.append(float(k13_param[sel]['e_RV'].iloc[0]))
            c_rv_nstar.append(int(k13_param[sel]['o_RV'].iloc[0]))
            c_rv_prov.append('Kharchenko+13')
            continue

        else:
            c_rvs.append(np.nan)
            c_err_rvs.append(np.nan)
            c_rv_nstar.append(np.nan)
            c_rv_prov.append('')

    param_df['cluster_rv'] = c_rvs
    param_df['cluster_err_rv'] = c_err_rvs
    param_df['cluster_rv_nstar'] = c_rv_nstar
    param_df['cluster_rv_provenance'] = c_rv_prov

    #
    # finally, begin writing the output
    #

    outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
               "{}_{}_fitparams_plus_observer_info.csv".
               format(today_YYYYMMDD(), uploadnamestr))
    param_df.to_csv(outpath, index=False, sep='|')
    print('made {}'.format(outpath))

    #
    # sparse observer info cut
    #
    scols = ['target', 'flag', 'disp','tag', 'group', 'RA_OBJ', 'DEC_OBJ',
             'CDIPSREF', 'CDCLSTER', 'phot_g_mean_mag', 'phot_bp_mean_mag',
             'phot_rp_mean_mag', 'TICID', 'TESSMAG', 'TICTEFF', 'TICRAD',
             'TICMASS', 'Gaia-ID'
            ]
    sparam_df = param_df[scols]

    outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
               "{}_{}_observer_info_sparse.csv".
               format(today_YYYYMMDD(), uploadnamestr))
    sparam_df.to_csv(outpath, index=False, sep='|')
    print('made {}'.format(outpath))

    #
    # full observer info cut
    #
    scols = ['target', 'flag', 'disp','tag', 'group', 'RA_OBJ', 'DEC_OBJ',
             'CDIPSREF', 'CDCLSTER', 'phot_g_mean_mag', 'phot_bp_mean_mag',
             'phot_rp_mean_mag', 'TICID', 'TESSMAG', 'TICTEFF', 'TICRAD',
             'TICMASS', 'Gaia-ID',
             'period', 'period_unc', 'epoch', 'epoch_unc', 'depth',
             'depth_unc', 'duration', 'duration_unc', 'radius', 'radius_unc',
             'stellar_rv', 'stellar_rv_unc', 'stellar_rv_provenance',
             'eso_rv_availability', 'cluster_rv', 'cluster_err_rv',
             'cluster_rv_nstar', 'cluster_rv_provenance'
            ]
    sparam_df = param_df[scols]

    outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
               "{}_{}_observer_info_full.csv".
               format(today_YYYYMMDD(), uploadnamestr))
    sparam_df.to_csv(outpath, index=False, sep='|')
    print('made {}'.format(outpath))


if __name__=="__main__":
    main()
