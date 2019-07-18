import os
from glob import glob
import numpy as np, pandas as pd
from cdips.utils import today_YYYYMMDD

import imageutils as iu

def main():

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
    kwlist = ['RA_OBJ','DEC_OBJ','CDIPSREF','phot_g_mean_mag',
              'phot_bp_mean_mag','phot_rp_mean_mag','TICID',
              'TESSMAG','Gaia-ID']

    for k in kwlist:
        thislist = []
        for l in lcpaths:
            thislist.append(iu.get_header_keyword(l, k, ext=0))
        df[k] = np.array(thislist)

    outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
               "{}_sector6_and_sector7_gold_fitparams_plus_observer_info.csv".
               format(today_YYYYMMDD()))
    df.to_csv(outpath, index=False, sep='|')
    print('made {}'.format(outpath))

if __name__=="__main__":
    main()
