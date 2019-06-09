"""
Randomly select N=100 (or 1000) TFA CDIPS light curves.
(Optionally select them to be only the LCs with strong LS FAPs already?)
Then inject a 0.5 RJup central-transit planet with periods in the range 1–15
d.  Try recovering via TLS (a) without detrending, (b) with detrending.
(For each light curve, do say 10 experiments).

For case (b), first-pass try using
the robust Huber spline (with w = 0.3 d) and also the sliding
biweight (w = 0.25 d). For a sliding biweight, if w/T14 > 2.2, most (> 98%)
of the flux integral is preserved. So anything between w=0.25 days to w=0.5
days should be good...
"""

def main():
    pass

def get_light_curves():
    pass

def inject_signals():
    pass

def detrend_lightcurves():
    pass

def find_dips():
    pass

def assess_statistics():
    pass
