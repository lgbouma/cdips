"""
test_stitchroutine.py - Luke Bouma (bouma.luke@gmail) - Nov 2020

Run tests for:
* Finding all available light curves given a GAIA DR2 source_id.
* Running a long-term trend correction over each orbit
* Stitching them together.
"""

from cdips.utils.lcutils import (
    find_cdips_lc_paths, get_lc_data, get_best_ap_number_given_lcpath
)
from cdips.testing import (
    assert_lsperiod_is_approx, assert_spdmperiod_is_approx
)


def test_stitchroutine():

    source_id = '5237036108585124096'

    lcpaths = test_get_cdips_lightcurves(source_id)
    assert len(lcpaths) >= 2

    #FIXME: implement this "longterm" correction.
    corrected_lightcurves = test_longterm_correction(lcpaths)
    import IPython; IPython.embed()

    stitched_lightcurve = test_stitch(corrected_lightcurves)


def test_get_cdips_lightcurves(source_id):

    lcpaths = find_cdips_lc_paths(source_id)

    return lcpaths


def test_longterm_correction(lcpaths):

    for lcpath in lcpaths:

        ap = get_best_ap_number_given_lcpath(lcpath)

        # tuple d:
        # (source_id, tfa_time, ap_mag, xcc, ycc, ra, dec, tmag, other_mag)
        #FIXME: i think you want the background time-series here too...
        d = get_lc_data(lcpath, mag_aperture=f'IRM{ap}',
                        tfa_aperture=f'PCA{ap}')

        assert len(d[1]) == len(d[2]) == len(d[8])

        # plan:
        # see research_notebook.txt, 2020/11/09
        #FIXME FIXME FIXME 

        import IPython; IPython.embed()


        # TODO: remove a linear, or maybe quadratic, trend.
        # ...At the orbit level.
        # ...Maybe via BIC? 
        pass

    assert X
    assert Y

    return corrected_lightcurves


def test_stitch(corrected_lightcurves):

    stitch(corrected_lightcurves)

    assert X
    assert Y

    return stitched_lightcurve


if __name__ == "__main__":
    test_stitchroutine()
