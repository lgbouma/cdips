"""
do_allvariable_report_making.py - Luke Bouma (bouma.luke@gmail) - Nov 2020

Given a list of Gaia source_ids, make systematics-corrected multi-sector light
curves, run periodograms for general variability classification (not just
planet-finding), and make an associated report.

Debugging:
$ python -u do_allvariable_report_making.py &> logs/ic2602_allvariable.log &

Run-time:
    Use the run_allvar_driver.sh script. (Because my implementation has a
    memory leak)
"""

import pickle, os, gc
import numpy as np, pandas as pd
from glob import glob
from datetime import datetime

import cdips.utils.lcutils as lcu
import cdips.utils.pipelineutils as ppu
import cdips.lcproc.detrend as dtr
import cdips.lcproc.mask_orbit_edges as moe
from cdips.plotting.allvar_report import make_allvar_report
from cdips.utils import str2bool

from wotan.slide_clipper import slide_clip

RUNID_EXTINCTION_DICT = {
    'IC_2602': 0.0799,  # avg E(B-V) from Randich+18, *1.31 per Stassun+2019
    'NGC_2516': 0.1343, # 0.25 for KC19
    'CrA': 0.06389, # KC19 ratio used
    'kc19group_113': 0.1386, # 0.258 from KC19 -- take ratio
    'Orion': 0.1074, # again KC19 ratio used
    'VelaOB2': 0.2686, # KC19 ratio
    'ScoOB2': 0.161, # KC19 ratio
}

def main():

    runid = 'kc19group_113' # CG18+KC19, sorted by color
    # runid = 'CrA' # CG18+KC19, sorted by color
    # runid = 'IC_2602' # CG18+KC19, sorted by color

    E_BpmRp = RUNID_EXTINCTION_DICT[runid]
    sourcelist_path = (
        f'/home/lbouma/proj/cdips/data/cluster_data/cdips_catalog_split/OC_MG_FINAL_v0.4_publishable_CUT_{runid}.csv'
    )

    # runid = 'NGC2516_core_plus_halo' # CG18+KC19, sorted by color
    # E_BpmRp = 0.1343 # apply extinction in summary plots
    # sourcelist_path = (
    #     f'/home/lbouma/proj/cdips/data/cluster_data/{runid}.csv'
    # )

    # runid = 'NGC2516_demo' # CG18 trimmed to first 15
    # E_BpmRp = 0.1343 # apply extinction in summary plots
    # sourcelist_path = (
    #     f'/home/lbouma/proj/cdips/data/cluster_data/{runid}_source_ids.csv'
    # )

    # runid = 'ic2602_examples'
    # sourcelist_path = (
    #     f'/home/lbouma/proj/cdips/tests/data/test_pca_{runid}.csv'
    # )
    # df = pd.read_csv(sourcelist_path, comment='#', names=['source_id'])

    ##########################################

    # the plot and linked pickles go here
    outdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/allvariability_reports'
    outdir = os.path.join(outdir, runid)
    for d in [outdir,
              os.path.join(outdir, 'data'),
              os.path.join(outdir, 'reports'),
              os.path.join(outdir, 'logs')
    ]:
        if not os.path.exists(d):
            os.mkdir(d)

    df = pd.read_csv(sourcelist_path)

    n_sources = len(df)
    n_logs = len(glob(os.path.join(outdir, 'logs', '*_status.log')))

    if n_logs < n_sources:

        max_per_run = 10

        cnt = 0
        for s in list(df.source_id):
            print('-'*42)
            print(cnt)
            res = do_allvariable_report_making(
                s, outdir=outdir, apply_extinction=E_BpmRp
            )
            cnt += res

            if cnt >= max_per_run:
                break

    else:
        print('-'*42)
        print('Found all logs! Taking a week-long break to avoid log-flooding.')
        print('Escape through a manual `killall python`.')
        print('-'*42)
        import time
        time.sleep(60*60*24*7)






def do_allvariable_report_making(source_id, outdir=None, overwrite=False,
                                 apply_extinction=None):

    print(42*'=')

    picklepath = os.path.join(outdir, 'data', f'{source_id}_allvar.pkl')
    statuspath = os.path.join(outdir, 'logs', f'{source_id}_status.log')

    if not os.path.exists(statuspath):
        # initialize status file
        lc_info = {'n_sectors': None, 'lcpaths': None,
                   'detrending_completed': None }
        ppu.save_status(statuspath, 'lc_info', lc_info)
        report_info = {
            'report_completed': None,
            'ls_period': None,
            'bestlspval': None,
            'nbestperiods': None,
            'nbestlspvals': None,
            'n_dict': None
        }
        ppu.save_status(statuspath, 'report_info', report_info)

    s = ppu.load_status(statuspath)
    if not overwrite:
        if str2bool(s['report_info']['report_completed']):
            print(f'Found {source_id} report_completed')
            return 0
        if s['lc_info']['n_sectors'] == '0':
            print(f'Found {source_id} n_sectors = 0')
            return 0
        if s['lc_info']['detrending_completed'] == 'False':
            print(f'Found {source_id} not detrending_completed')
            return 0

    # get the data needed to make the report if it hasn't already been made.
    if not os.path.exists(picklepath):

        thetime = datetime.utcnow().isoformat()
        print(f'{thetime}: Beginning {source_id} detrending.')

        #
        # get the light curves
        #

        lcpaths = lcu.find_cdips_lc_paths(source_id, raise_error=False)

        if lcpaths is None:
            lc_info = {'n_sectors': 0, 'lcpaths': None,
                       'detrending_completed': False}
            ppu.save_status(statuspath, 'lc_info', lc_info)
            return 0

        #
        # detrend systematics. each light curve yields tuples of:
        #   primaryhdr, data, ap, dtrvecs, eigenvecs, smooth_eigenvecs
        #
        dtr_infos = []
        for lcpath in lcpaths:
            dtr_info = dtr.detrend_systematics(lcpath)
            dtr_infos.append(dtr_info)

        #
        # stitch all available light curves
        #
        ap = dtr_infos[0][2]
        timelist = [d[1]['TMID_BJD'] for d in dtr_infos]
        maglist = [d[1][f'PCA{ap}'] for d in dtr_infos]
        magerrlist = [d[1][f'IRE{ap}'] for d in dtr_infos]

        extravecdict = {}
        extravecdict[f'IRM{ap}'] = [d[1][f'IRM{ap}'] for d in dtr_infos]
        for i in range(0,7):
            extravecdict[f'CBV{i}'] = [d[3][i, :] for d in dtr_infos]

        try:
            time, flux, fluxerr, vec_dict = lcu.stitch_light_curves(
                timelist, maglist, magerrlist, extravecdict
            )
        except ValueError:
            lc_info = {'n_sectors': len(lcpaths), 'lcpaths': lcpaths,
                       'detrending_completed': False}
            ppu.save_status(statuspath, 'lc_info', lc_info)
            return 0


        #
        # mask orbit edges
        #
        s_time, s_flux, inds = moe.mask_orbit_start_and_end(
            time, flux, raise_expectation_error=False, orbitgap=0.7,
            return_inds=True
        )
        s_fluxerr = fluxerr[inds]

        #
        # remove outliers with windowed stdevn removal
        #
        window_length = 1.5 # days
        s_flux = slide_clip(s_time, s_flux, window_length, low=3, high=2,
                            method='mad', center='median')

        #
        # fix any "zero" values in s_flux to be NaN
        #
        s_flux[s_flux == 0] = np.nan

        ap = dtr_infos[0][2]
        allvardict = {
            'source_id': source_id,
            'E_BpmRp': apply_extinction,
            'ap': ap,
            'TMID_BJD': time,
            f'PCA{ap}': flux,
            f'IRE{ap}': fluxerr,
            'STIME': s_time,
            f'SPCA{ap}': s_flux,
            f'SPCAE{ap}': s_fluxerr,
            'dtr_infos': dtr_infos,
            'vec_dict': vec_dict
        }

        with open(picklepath , 'wb') as f:
            pickle.dump(allvardict, f)

        #
        # sanity check that PCA / detrending worked
        #
        limit_fraction = 0.75
        if len(flux[pd.isnull(flux)])/len(flux) > limit_fraction:
            lc_info = {'n_sectors': len(lcpaths), 'lcpaths': lcpaths,
                       'detrending_completed': False}
            ppu.save_status(statuspath, 'lc_info', lc_info)
            return 0

        #
        # update status that detrending worked.
        #
        lc_info = {'n_sectors': len(lcpaths), 'lcpaths': lcpaths,
                   'detrending_completed': True}
        ppu.save_status(statuspath, 'lc_info', lc_info)

    s = ppu.load_status(statuspath)
    if str2bool(s['lc_info']['detrending_completed']):
        with open(picklepath, 'rb') as f:
            allvardict = pickle.load(f)
    else:
        return 0

    #
    # make summary plots.
    #

    if not str2bool(s['report_info']['report_completed']):

        thetime = datetime.utcnow().isoformat()
        print(f'{thetime}: Beginning {source_id} allvar report.')

        plotdir = os.path.join(outdir, 'reports')
        outd = make_allvar_report(allvardict, plotdir)

    #
    # save their output (most crucially, including the bestperiods)
    #
    outpicklepath = os.path.join(outdir, 'data', f'{source_id}_reportinfo.pkl')
    with open(outpicklepath , 'wb') as f:
        pickle.dump(outd, f)

    report_info = {
        'report_completed': True,
        'ls_period': outd['lsp']['bestperiod'],
        'bestlspval': outd['lsp']['bestlspval'],
        'nbestperiods': outd['lsp']['nbestperiods'],
        'nbestlspvals': outd['lsp']['nbestlspvals'],
        'n_dict': outd['n_dict']
    }
    ppu.save_status(statuspath, 'report_info', report_info)

    return 1


if __name__ == "__main__":
    main()
