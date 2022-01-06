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
import os
import multiprocessing as mp
import numpy as np, pandas as pd
from astropy.io import fits
from cdips.vetting.make_vetting_multipg_pdf import make_vetting_multipg_pdf
from cdips.testing import check_dependencies

# NOTE (PERFORMANCE WARNING): possible slowdown here -- the multithreading is
# happening over the periodfinding and similar tasks within each vetting
# report.  rather than giving each vetting report one thread, and then
# multithreading over vetting reports.
nworkers = mp.cpu_count()

def make_all_vetting_reports(lcpaths, lcbasedir, resultsdir, cdips_df,
                             supplementstatsdf, pfdf, toidf,
                             show_rvs=True,
                             sector=None, cdipsvnum=1):

    check_dependencies()

    for lcpath in lcpaths:

        source_id = int(lcpath.split('gaiatwo')[1].split('-')[0].lstrip('0'))
        mdf = cdips_df[cdips_df['source_id']==source_id]
        if len(mdf) != 1:
            errmsg = 'expected exactly 1 source match in CDIPS cat'
            raise AssertionError(errmsg)

        try:
            hdul = fits.open(lcpath)
        except FileNotFoundError:
            LOGERROR(f'ERROR! Failed to find {lcpath}. skipping.')
            continue

        hdr = hdul[0].header
        cam, ccd = hdr['CAMERA'], hdr['CCD']
        hdul.close()

        lcname = (
            'hlsp_cdips_tess_ffi_'
            'gaiatwo{zsource_id}-{zsector}-cam{cam}-ccd{ccd}_'
            'tess_v{zcdipsvnum}_llc.fits'
        ).format(
            cam=cam,
            ccd=ccd,
            zsource_id=str(source_id).zfill(22),
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
            'vet_'+os.path.basename(lcpath).replace('.fits','.pdf')
        )
        nottransitpath = os.path.join(
            resultsdir,'nottransitpdfs',
            'vet_'+os.path.basename(lcpath).replace('.fits','.pdf')
        )

        supprow = _get_supprow(source_id, supplementstatsdf)
        suppfulldf = supplementstatsdf

        # don't make a report if the membership claim is insufficient
        if 'reference' in supprow:
            reference = str(supprow['reference'].iloc[0])
        elif 'reference_id' in supprow:
            reference = str(supprow['reference_id'].iloc[0])
        referencesplt = reference.split(',')
        INSUFFICIENT = ['Zari_2018_UMS', 'Zari2018ums']
        is_insufficient = [c in INSUFFICIENT for c in referencesplt]
        if np.all(is_insufficient):
            msg = (
                'Found {} had membership only in {}: was {}. Skip.'.
                format(source_id, repr(INSUFFICIENT), repr(referencesplt))
            )
            LOGINFO(msg)
            continue

        pfrow = pfdf.loc[pfdf['source_id']==source_id]
        if len(pfrow) != 1:
            if len(pfrow) == 0:
                errmsg = (
                    '{} expected 1 source match in period find df, got {}.'
                    .format(source_id, len(pfrow))
                )
                raise AssertionError(errmsg)
            else:
                # If you have multiple hits for a given source (e.g., because
                # of geometric camera overlap), take the max SDE.
                pfrow = pd.DataFrame(pfrow.loc[pfrow.tls_sde.idxmax()]).T

        DEPTH_CUTOFF = 0.75
        if float(pfrow.tls_depth) < DEPTH_CUTOFF:
            msg = (
                'Found {} had TLS depth {}. Too low. Skip.'.
                format(source_id, float(pfrow.tls_depth))
            )
            LOGINFO(msg)
            continue

        if not os.path.exists(outpath) and not os.path.exists(nottransitpath):
            if DEBUG:
                make_vetting_multipg_pdf(lcpath, outpath, mdf,
                                         source_id, supprow, suppfulldf, pfdf,
                                         pfrow, toidf, sector,
                                         mask_orbit_edges=True,
                                         nworkers=nworkers, show_rvs=show_rvs)
            else:
                try:
                    make_vetting_multipg_pdf(lcpath, outpath, mdf,
                                             source_id, supprow, suppfulldf, pfdf,
                                             pfrow, toidf, sector,
                                             mask_orbit_edges=True,
                                             nworkers=nworkers, show_rvs=show_rvs)
                except Exception as e:
                    LOGWARNING('WRN! {} continue.'.format(repr(e)))
        else:
            if os.path.exists(outpath):
                LOGINFO(f'Found {outpath}, continue')
            elif os.path.exists(nottransitpath):
                LOGINFO(f'Found {nottransitpath}, continue')

    LOGINFO('Completed make_all_vetting_reports!')


def _get_supprow(source_id, supplementstatsdf):

    mdf = supplementstatsdf.loc[supplementstatsdf['lcobj']==source_id]

    if len(mdf) > 1:
        LOGINFO('WRN! Got multiple supplementstatsdf entries for {}'.
              format(source_id))
        # Case: multiple matches. Take whichever has the least NaNs. Maintain
        # it as a ~160 column, 1 row dataframe.
        mdf = pd.DataFrame(
            mdf.iloc[mdf.isnull().sum(axis=1).values.argmin()]
        ).T

    return mdf
