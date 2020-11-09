"""
test_stitchroutine.py - Luke Bouma (bouma.luke@gmail) - Nov 2020

Run tests for:
* Finding all available light curves given a GAIA DR2 source_id.
* Running a long-term trend correction over each orbit
* Stitching them together.
"""

def test_stitchroutine():

    source_id = '5237036108585124096'

    lcpaths = test_get_cdips_lightcurves(source_id)
    import IPython; IPython.embed()

    corrected_lightcurves = test_longterm_correction(lcpaths)

    stitched_lightcurve = test_stitch(corrected_lightcurves)


def test_get_cdips_lightcurves(source_id):

    from cdips.utils.lcutils import find_cdips_lc_paths

    lcpaths = find_cdips_lc_paths(source_id)

    assert 0 #FIXME

    assert X
    assert Y

    return lcpaths


def test_longterm_correction(lcpaths):

    for lcpath in lcpaths:
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

