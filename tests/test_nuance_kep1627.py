import os
from os.path import join
import numpy as np, pandas as pd, lightkurve as lk
from cdips.lcproc.nuance_planet_search import run_nuance, run_iterative_nuance

star_id = 'Kepler_1627'
cachedir = "./temp/"

csvcachepath = join(cachedir, f"{star_id}_lc.csv")
if not os.path.exists(csvcachepath):
    # Get long cadence light curves for all quarters. Median normalize all
    # quarters, remove nans, and run a 5-sigma outlier clipping.
    lcf = lk.search_lightcurve(
        "6184894", mission="Kepler", author="Kepler", cadence="long"
    ).download_all()
    lc = lcf.stitch().remove_nans().remove_outliers()

    # Require non-zero quality flags, since we have an abundance of data.
    lc = lc[lc.quality == 0]

    # Make sure that the data type is consistent
    time = np.ascontiguousarray(lc.time.value, dtype=np.float64)
    flux = np.ascontiguousarray(lc.flux, dtype=np.float64)
    flux_err = np.ascontiguousarray(lc.flux_err, dtype=np.float64)

    df = pd.DataFrame({'time':time, 'flux':flux, 'flux_err': flux_err})
    df.to_csv(csvcachepath, index=False)

df = pd.read_csv(csvcachepath)
time, flux, flux_err = np.array(df.time), np.array(df.flux), np.array(df.flux_err)
med_flux = np.nanmedian(flux)
flux /= med_flux
flux_err /= med_flux

time, flux, flux_err = time, flux, flux_err

search_params = {
        'period_min': 1,
        'period_max': 20,
        'oversample': 50, # 50 not bad
}

# DROP HALF THE DATA TO IMPROVE RUNTIME
N = len(time)
time = time[:int(N/2)]
flux = flux[:int(N/2)]
flux_err = flux_err[:int(N/2)]

outdict = run_iterative_nuance(
    time, flux, flux_err,
    star_id, cachedir,
    cleaning_type='iterativegp',
    n_cpus=10,
    search_params=search_params
)
