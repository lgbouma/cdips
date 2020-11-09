import numpy as np
from wotan import flatten
import matplotlib.pyplot as plt
from matplotlib import rcParams; rcParams["figure.dpi"] = 150

def test_wotan_fake_data(t0):
    # t0: time system offset

    delta_t = 0.5/24 # TESS FFI cadence
    time = t0+np.arange(0, 28.5, delta_t)
    rotation_period = 2
    rotation_semiamplitude = 10 # units: part per thousdand. 10 <-> 1%
    flux = 1 + ((
        30*np.sin(2*np.pi*(time-t0)/rotation_period)
        + (time-t0) / 10
        + (time-t0)**1.5 / 100) / 1000
    )

    noise = np.random.normal(0, 0.0001, len(time))

    flux += noise
    for i in range(len(time)):
        if i % 75 == 0:
            flux[i:i+5] -= 5e-3  # Add some transits
            flux[i+50:i+52] += 5e-3  # and flares
    flux[300:400] = np.nan  # and a gap

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='biweight',
        window_length=0.5,    # The length of the filter window in units of ``time``
        edge_cutoff=0.5,      # length (in units of time) to be cut off each edge.
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        cval=6              # Tuning parameter for the robust estimators
        )

    f,ax = plt.subplots(nrows=2, ncols=1, figsize=(16,8))
    ax[0].scatter(time, flux, color='black', s=3, zorder=5)
    ax[0].plot(time, trend_lc, color='red', lw=1, zorder=3)

    ax[1].scatter(time, flatten_lc, s=1, color='black')
    f.savefig('test_wotan_plots/test_wotan_fake_data_t0_{}.png'.format(t0), bbox_inches='tight')


if __name__ == "__main__":
    test_wotan_fake_data(t0=2457000)
    test_wotan_fake_data(t0=0)
