import os, argparse, pickle, h5py, json, shutil
from glob import glob
from parse import search

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('text', usetex=False)

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy.io import fits
from astropy import units as u, constants as const

from astrobase.varbase import lcfit
from astrobase import astrotess as at
from astrobase.periodbase import kbls
from astrobase.varbase.trends import smooth_magseries_ndimage_medfilt
from astrobase import lcmath
from astrobase.services.tic import tic_single_object_crossmatch
from astrobase.varbase.transits import get_snr_of_dip
from astrobase.varbase.transits import estimate_achievable_tmid_precision
from astrobase.plotbase import plot_phased_magseries

from astrobase.varbase.transits import get_transit_times

from numpy.polynomial.legendre import Legendre

from astropy import units as u
from astropy.coordinates import SkyCoord

"""
MCMC fit Mandel-Agol transits to gold above -- these parameters are used for
paper and CTOIs.

(pulls from tessorbitaldecay/measure_transit_times_from_lightcurve.py repo)
"""

#TODO: need to use moe & related detrending to re-get the maximally detrended
#LC... or else just save it somewhere to stop re-doing it in code (!!)

def read_gold(sector):
    pass

def main():
    #FIXME
    #FIXME
    # implement!! probably add the tessorbitaldecay thing to ur path, bc it
    # makes sense to just continue developing those functions...
    #FIXME
    #FIXME

    # run bls to get initial parameters.
    endp = 1.05*(np.nanmax(time) - np.nanmin(time))/2
    blsdict = kbls.bls_parallel_pfind(time, flux, err_flux, magsarefluxes=True,
                                      startp=0.1, endp=endp,
                                      maxtransitduration=0.3, nworkers=8,
                                      sigclip=None)
    fitd = kbls.bls_stats_singleperiod(time, flux, err_flux,
                                       blsdict['bestperiod'],
                                       magsarefluxes=True, sigclip=None,
                                       perioddeltapercent=5)

    bls_period = fitd['period']
    #  plot the BLS model.
    from astrobase.lcfit.utils import make_fit_plot
    make_fit_plot(fitd['phases'], fitd['phasedmags'], None, fitd['blsmodel'],
                  fitd['period'], fitd['epoch'], fitd['epoch'], blsfit_savfile,
                  magsarefluxes=True)

    ingduration_guess = fitd['transitduration']*0.2
    transitparams = [fitd['period'], fitd['epoch'], fitd['transitdepth'],
                     fitd['transitduration'], ingduration_guess
                    ]

    # fit a trapezoidal transit model; plot the resulting phased LC.
    trapfit = lcfit.traptransit_fit_magseries(time, flux, err_flux,
                                              transitparams,
                                              magsarefluxes=True, sigclip=None,
                                              plotfit=trapfit_savfile)

    period = trapfit['fitinfo']['finalparams'][0]
    t0 = trapfit['fitinfo']['fitepoch']
    transitduration_phase = trapfit['fitinfo']['finalparams'][3]
    tdur = period * transitduration_phase

    # isolate each transit to within +/- n_transit_durations
    tmids, t_starts, t_ends = (
        get_transit_times(fitd, time, n_transit_durations, trapd=trapfit)
    )

    rp = np.sqrt(trapfit['fitinfo']['finalparams'][2])

    if read_literature_params:

        litdir = "../data/literature_physicalparams/{:d}/".format(ticid)
        if not os.path.exists(litdir):
            os.mkdir(litdir)
        litpath = os.path.join(litdir, 'params.csv')

        if not os.path.exists(litpath):
            # attempt to get physical parameters of planet -- period, a/Rstar, and
            # inclination -- for the initial guesses.
            from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
            eatab = NasaExoplanetArchive.get_confirmed_planets_table()

            pl_coords = eatab['sky_coord']
            tcoord = SkyCoord(targetcoordstr, frame='icrs', unit=(u.deg, u.deg))

            print('got match w/ separation {}'.format(
                np.min(tcoord.separation(pl_coords).to(u.arcsec))))
            pl_row = eatab[np.argmin(tcoord.separation(pl_coords).to(u.arcsec))]

            # all dimensionful
            period = pl_row['pl_orbper'].value
            incl = pl_row['pl_orbincl'].value
            semimaj_au = pl_row['pl_orbsmax']
            rstar = pl_row['st_rad']
            a_by_rstar = (semimaj_au / rstar).cgs.value

            litdf = pd.DataFrame(
                {'period_day':period,
                 'a_by_rstar':a_by_rstar,
                 'inclination_deg':incl
                }, index=[0]
            )
            # get the fixed physical parameters from the data. period_day,
            # a_by_rstar, and inclination_deg are comma-separated in this file.
            litdf.to_csv(litpath, index=False, header=True, sep=',')
            litdf = pd.read_csv(litpath, sep=',')
        else:
            litdf = pd.read_csv(litpath, sep=',')

        # NOTE: only period is used, for now
        litparams = tuple(map(float,
            [litdf['period_day'],litdf['a_by_rstar'],litdf['inclination_deg']])
        )
    else:
        raise AssertionError

    # fit the phased transit, within N durations of the transit itself, to
    # determine a/Rstar, inclination, and quadratric terms for fixed
    # parameters. only period from literature.
    fit_abyrstar, fit_incl, fit_ulinear, fit_uquad = (
        fit_phased_transit_mandelagol_and_line(
            sectornum,
            t_starts, t_ends, time, flux, err_flux, lcfile, fitd, trapfit,
            bls_period, litparams, ticid, fit_savdir, chain_savdir, nworkers,
            n_phase_mcmc_steps, overwriteexistingsamples, mcmcprogressbar)
    )


