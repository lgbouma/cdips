"""
Usage:

$(cdips) python merge_for_exofoptess.py
"""
import os, socket
from glob import glob
import numpy as np, pandas as pd
from numpy import array as nparr
from cdips.utils import today_YYYYMMDD

from astrobase import imageutils as iu

from cdips.utils import find_rvs as fr
from cdips.utils import get_vizier_catalogs as gvc
from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.utils.pipelineutils import save_status, load_status

hostname = socket.gethostname()
if 'phtess1' in hostname or 'phtess2' in hostname:
    fitdir = "/home/lbouma/proj/cdips/results/fit_gold"
    exofopdir = "/home/lbouma/proj/cdips/data/exoFOP_uploads"
elif 'brik' in hostname:
    fitdir = "/home/luke/Dropbox/proj/cdips/results/fit_gold"
    exofopdir = "/home/luke/Dropbox/proj/cdips/data/exoFOP_uploads"
else:
    raise ValueError('where is fit_gold directory on {}?'.format(hostname))

def main(is_dayspecific_exofop_upload=1, cdipssource_vnum=0.4):
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
    """

    #FIXME FIXME
    # 1) need to update your "tag" scheme, s.t. each CTOI in the same upload
    # gets the same tag. (I guess this makes sense)
    # 2) need to fix your float truncation length. round to reasonable
    # numbers!!
    #FIXME FIXME
    # raise AssertionError('fix above items')

    #
    # read in the results from the fits
    #
    paramglob = os.path.join(
        fitdir, "sector-?/fitresults/hlsp_*gaiatwo*_llc/*fitparameters.csv"
    )
    parampaths = glob(paramglob)
    statusglob = os.path.join(
        fitdir, "sector-*/fitresults/hlsp_*gaiatwo*_llc/*.stat"
    )
    statuspaths = glob(statusglob)

    statuses = [dict(load_status(f)['fivetransitparam_fit'])
                for f in statuspaths]

    param_df = pd.concat((pd.read_csv(f, sep='|') for f in parampaths))

    outpath = os.path.join(
        fitdir, "{}_sector6_and_sector7_gold_mergedfitparams.csv".
        format(today_YYYYMMDD())
    )
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
        # These manually commented candidates are the only ones we're
        # uploading.
        #
        manual_comment_df = pd.read_csv(
            '../data/exoFOP_uploads/{}_cdips_candidate_upload.csv'.
            format(today_YYYYMMDD()), sep="|"
        )
        common = status_df.merge(manual_comment_df,
                                 on='source_id',
                                 how='inner')
        sel_status_df = status_df[status_df.source_id.isin(common.source_id)]

        #
        # WARN: the MCMC fits should have converged before uploading.
        # (20190918 had two exceptions, 5618515825371166464, and
        # 5715454237977779968, where the fits look OK.)
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
        # Just require that you actually have a parameter file (...).
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

        _df = manual_comment_df[
            manual_comment_df.source_id.isin(to_exofop_df.source_id)
        ]

        comments = list(_df['comment'])

        for c in comments:
            assert len(c)<=119

        to_exofop_df = to_exofop_df.sort_values(by="source_id")
        _df = _df.sort_values(by="source_id")

        to_exofop_df['notes'] = comments

        outpath = os.path.join(
            exofopdir, "{}_s6_and_s7_w_sourceid.csv".
            format(today_YYYYMMDD())
        )
        to_exofop_df.to_csv(outpath, index=False, sep='|')
        print('made {}'.format(outpath))

        to_exofop_df = to_exofop_df.drop(['source_id'], axis=1)
        outpath = os.path.join(
            exofopdir, "params_planet_{}_001.txt".
            format(today_YYYYMMDD())
        )
        to_exofop_df.to_csv(outpath, index=False, sep='|', header=False)
        print('made {}'.format(outpath))

        # manually check these...
        print('\nPeriod uncertainties [minutes]')
        print(to_exofop_df['period_unc']*24*60)
        print('\nEpoch uncertainties [minutes]')
        print(to_exofop_df['epoch_unc']*24*60)
        print('\nPlanet radii [Rearth]')
        print(to_exofop_df[['radius','radius_unc','notes']])

    #
    # above is the format exofop-TESS wants. however it's not particularly
    # useful for followup. for that, we want: gaia IDs, magnitudes, ra, dec.
    #
    assert 0 #FIXME : remove once phtess2 is up!
    gaiaids = list(map(
        lambda x: int(
            os.path.basename(x).split('gaiatwo')[1].split('-')[0].lstrip('0')
        ), parampaths
    ))

    lcnames = list(map(
        lambda x: os.path.basename(x).replace('_fitparameters.csv','.fits'),
        parampaths
    ))

    lcdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-?/cam?_ccd?/'
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
    mdf = df.merge(ccdf, how='left', left_on='Gaia-ID', right_on='source_id')
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
               "{}_sector6_and_sector7_gold_fitparams_plus_observer_info.csv".
               format(today_YYYYMMDD()))
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
               "{}_sector6_and_sector7_gold_observer_info_sparse.csv".
               format(today_YYYYMMDD()))
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
               "{}_sector6_and_sector7_gold_observer_info_full.csv".
               format(today_YYYYMMDD()))
    sparam_df.to_csv(outpath, index=False, sep='|')
    print('made {}'.format(outpath))


if __name__=="__main__":
    main()
