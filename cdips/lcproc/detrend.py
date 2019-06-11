from wotan import flatten

def detrend_flux(time, flux):

    # matched detrending to do_initial_period_finding

    break_tolerance = 0.5
    flat_flux, trend_flux = flatten(time, flux,
                                    method='pspline',
                                    return_trend=True,
                                    break_tolerance=break_tolerance)

    return flat_flux, trend_flux
