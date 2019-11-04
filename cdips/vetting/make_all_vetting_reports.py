import os
import numpy as np, pandas as pd
from astropy.io import fits
from cdips.vetting.make_vetting_multipg_pdf import make_vetting_multipg_pdf

def _get_supprow(sourceid, supplementstatsdf):

    mdf = supplementstatsdf.loc[supplementstatsdf['lcobj']==sourceid]

    return mdf


def make_all_vetting_reports(tfa_sr_paths, lcbasedir, resultsdir, cdips_df,
                             supplementstatsdf, pfdf, toidf, k13_notes_df,
                             sector=6, cdipsvnum=1):

    for tfa_sr_path in tfa_sr_paths:

        sourceid = int(tfa_sr_path.split('gaiatwo')[1].split('-')[0].lstrip('0'))
        mdf = cdips_df[cdips_df['source_id']==sourceid]
        if len(mdf) != 1:
            errmsg = 'expected exactly 1 source match in CDIPS cat'
            raise AssertionError(errmsg)

        hdul = fits.open(tfa_sr_path)
        hdr = hdul[0].header
        cam, ccd = hdr['CAMERA'], hdr['CCD']
        hdul.close()

        lcname = (
            'hlsp_cdips_tess_ffi_'
            'gaiatwo{zsourceid}-{zsector}-cam{cam}-ccd{ccd}_'
            'tess_v{zcdipsvnum}_llc.fits'
        ).format(
            cam=cam,
            ccd=ccd,
            zsourceid=str(sourceid).zfill(22),
            zsector=str(sector).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )

        lcpath = os.path.join(
            lcbasedir,
            'cam{}_ccd{}'.format(cam, ccd),
            lcname
        )

        # logic: even if you know it's nottransit, it's a TCE. therefore, the
        # pdf will be made. i just dont want to have to look at it. put it in a
        # separate directory.
        outpath = os.path.join(
            resultsdir,'pdfs',
            'vet_'+os.path.basename(tfa_sr_path).replace('.fits','.pdf')
        )
        nottransitpath = os.path.join(
            resultsdir,'nottransitpdfs',
            'vet_'+os.path.basename(tfa_sr_path).replace('.fits','.pdf')
        )

        supprow = _get_supprow(sourceid, supplementstatsdf)
        suppfulldf = supplementstatsdf

        pfrow = pfdf.loc[pfdf['source_id']==sourceid]
        if len(pfrow) != 1:
            errmsg = 'expected exactly 1 source match in period find df'
            raise AssertionError(errmsg)

        if not os.path.exists(outpath) and not os.path.exists(nottransitpath):
            make_vetting_multipg_pdf(tfa_sr_path, lcpath, outpath, mdf,
                                     sourceid, supprow, suppfulldf, pfdf,
                                     pfrow, toidf, sector, k13_notes_df)
        else:
            print('found {}, continue'.format(outpath))
