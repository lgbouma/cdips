"""
Put together a few useful CSV candidate summaries:

* bulk uploads to exofop/tess
    target|flag|disp|period|period_unc|epoch|epoch_unc|depth|depth_unc|
    duration|duration_unc|inc|inc_unc|imp|imp_unc|r_planet|r_planet_unc|
    ar_star|a_rstar_unc|radius|radius_unc|mass|mass_unc|temp|temp_unc|
    insol|insol_unc|dens|dens_unc|sma|sma_unc|ecc|ecc_unc|arg_peri|
    arg_peri_unc|time_peri|time_peri_unc|vsa|vsa_unc|tag|group|prop_period|
    notes

* observer info sparse
    target|flag|disp|tag|group|RA_OBJ|DEC_OBJ|CDIPSREF|phot_g_mean_mag|
    phot_bp_mean_mag|phot_rp_mean_mag|TICID|TESSMAG|Gaia-ID

* observer info full
    target|flag|disp|tag|group|RA_OBJ|DEC_OBJ|CDIPSREF|phot_g_mean_mag|
    phot_bp_mean_mag|phot_rp_mean_mag|TICID|TESSMAG|Gaia-ID
    period|period_unc|epoch|epoch_unc|depth|depth_unc|duration|duration_unc|
    r_planet|r_planet_unc|radius|radius_unc|mass|mass_unc|temp|temp_unc|
    stellar_rv|stellar_rv_unc|stellar_rv_provenance|eso_rv_availability

* merge of everything
    target|flag|disp|period|period_unc|epoch|epoch_unc|depth|depth_unc|
    duration|duration_unc|inc|inc_unc|imp|imp_unc|r_planet|r_planet_unc|
    ar_star|a_rstar_unc|radius|radius_unc|mass|mass_unc|temp|temp_unc|
    insol|insol_unc|dens|dens_unc|sma|sma_unc|ecc|ecc_unc|arg_peri|
    arg_peri_unc|time_peri|time_peri_unc|vsa|vsa_unc|tag|group|prop_period|
    notes|RA_OBJ|DEC_OBJ|CDIPSREF|phot_g_mean_mag|phot_bp_mean_mag|
    phot_rp_mean_mag|TICID|TESSMAG|Gaia-ID
"""
import os
from glob import glob
import numpy as np, pandas as pd
from numpy import array as nparr
from cdips.utils import today_YYYYMMDD

import imageutils as iu

from cdips.utils import find_rvs as fr
from cdips.utils import get_vizier_catalogs as gvc
from cdips.utils import collect_cdips_lightcurves as ccl

def main(cdipssource_vnum=0.3):

    paramglob = ("/home/lbouma/proj/cdips/results/fit_gold"
                 "/sector-*/fitresults/hlsp_*gaiatwo*_llc/*fitparameters.csv")
    parampaths = glob(paramglob)

    df = pd.concat((pd.read_csv(f, sep='|') for f in parampaths))

    outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
               "{}_sector6_and_sector7_gold_mergedfitparams.csv".
               format(today_YYYYMMDD()))
    df.to_csv(outpath, index=False, sep='|')
    print('made {}'.format(outpath))

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
        df[k] = np.array(thislist)

    # now search for stellar RV xmatch
    res = [fr.get_rv_xmatch(ra, dec, G_mag=gmag, dr2_sourceid=s)
           for ra, dec, gmag, s in
           zip(list(df['RA_OBJ']), list(df['DEC_OBJ']),
               list(df['phot_g_mean_mag']), list(df['Gaia-ID']))
          ]

    res = np.array(res)
    df['stellar_rv'] = res[:,0]
    df['stellar_rv_unc'] = res[:,1]
    df['stellar_rv_provenance'] = res[:,2]

    # make column showing whether there are ESO spectra available
    res = [fr.wrangle_eso_for_rv_availability(ra, dec)
           for ra, dec in
           zip(list(df['RA_OBJ']), list(df['DEC_OBJ']))
          ]
    df['eso_rv_availability'] = nparr(res)[:,2]

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
    df['unique_cluster_name'] = nparr(mdf['unique_cluster_name'])

    s19 = gvc.get_soubiran_19_rv_table()
    k13_param = gvc.get_k13_param_table()

    c_rvs, c_err_rvs, c_rv_nstar, c_rv_prov = [], [], [], []
    for ix, row in df.iterrows():

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

    df['cluster_rv'] = c_rvs
    df['cluster_err_rv'] = c_err_rvs
    df['cluster_rv_nstar'] = c_rv_nstar
    df['cluster_rv_provenance'] = c_rv_prov

    #
    # finally, begin writing the output
    #

    outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
               "{}_sector6_and_sector7_gold_fitparams_plus_observer_info.csv".
               format(today_YYYYMMDD()))
    df.to_csv(outpath, index=False, sep='|')
    print('made {}'.format(outpath))

    #
    # sparse observer info cut
    #
    scols = ['target', 'flag', 'disp','tag', 'group', 'RA_OBJ', 'DEC_OBJ',
             'CDIPSREF', 'phot_g_mean_mag', 'phot_bp_mean_mag',
             'phot_rp_mean_mag', 'TICID', 'TESSMAG', 'TICTEFF', 'TICRAD',
             'TICMASS', 'Gaia-ID'
            ]
    sdf = df[scols]

    outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
               "{}_sector6_and_sector7_gold_observer_info_sparse.csv".
               format(today_YYYYMMDD()))
    sdf.to_csv(outpath, index=False, sep='|')
    print('made {}'.format(outpath))

    #
    # full observer info cut
    #
    scols = ['target', 'flag', 'disp','tag', 'group', 'RA_OBJ', 'DEC_OBJ',
             'CDIPSREF', 'phot_g_mean_mag', 'phot_bp_mean_mag',
             'phot_rp_mean_mag', 'TICID', 'TESSMAG', 'TICTEFF', 'TICRAD',
             'TICMASS', 'Gaia-ID',
             'period', 'period_unc', 'epoch', 'epoch_unc', 'depth',
             'depth_unc', 'duration', 'duration_unc', 'radius', 'radius_unc',
             'stellar_rv', 'stellar_rv_unc', 'stellar_rv_provenance',
             'eso_rv_availability', 'cluster_rv', 'cluster_err_rv',
             'cluster_rv_nstar', 'cluster_rv_provenance'
            ]
    sdf = df[scols]

    outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
               "{}_sector6_and_sector7_gold_observer_info_full.csv".
               format(today_YYYYMMDD()))
    sdf.to_csv(outpath, index=False, sep='|')
    print('made {}'.format(outpath))


if __name__=="__main__":
    main()
