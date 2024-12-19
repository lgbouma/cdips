import os
from os.path import join
import numpy as np, pandas as pd, lightkurve as lk
from cdips.lcproc.nuance_planet_search import run_nuance, run_iterative_nuance

def main():
    star_id = 'TOI_837'
    #star_id = 'TOI_451'
    cachedir = "./temp/"

    # TOI-837: TESS-SPOC 1800 sec sector 11 data
    csvcachepath = join(cachedir, f"{star_id}_lc.csv")
    if not os.path.exists(csvcachepath):
        lc = lk.search_lightcurve(star_id.replace("_", " "), author="TESS-SPOC",
                                  exptime=1800)[1].download()
        time = lc.time.to_value("btjd")
        flux = lc.pdcsap_flux.to_value().filled(np.nan)
        df = pd.DataFrame({'time':time, 'flux':flux})
        df.to_csv(csvcachepath, index=False)

    df = pd.read_csv(csvcachepath)
    time, flux = np.array(df.time), np.array(df.flux)
    flux /= np.nanmedian(flux)

    from cdips.utils.lcutils import p2p_rms
    flux_err = np.ones_like(flux) * p2p_rms(flux)

    search_params = {
            'period_min': 1,
            'period_max': 12,
            'oversample': 200, # 50 not bad
    }

    run_iterative = 1
    if run_iterative:
        outdict = run_iterative_nuance(
            time, flux, flux_err,
            star_id, cachedir,
            cleaning_type='iterativegp',
            n_cpus=10,
            search_params=search_params
        )
    else:
        outdict = run_nuance(
            time, flux, flux_err,
            star_id, cachedir,
            cleaning_type='iterativegp',
            n_cpus=10,
            search_params=search_params
        )

if __name__ == "__main__":
    main()
