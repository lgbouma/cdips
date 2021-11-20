import os
from glob import glob

from cdips.vetting import (
    initialize_vetting_report_information as ivri,
    make_all_vetting_reports as mavp
)

def test_vetting_report(sector=None, cdips_cat_vnum=None):

    outdir = 'test_vetting_reports'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    (lc_paths, lcbasedir, resultsdir, cdips_df, supplementstatsdf, pfdf, toidf,
     sector) = (
            ivri.initialize_vetting_report_information(sector, cdips_cat_vnum)
    )


    if sector == 6:
        ids_to_test = [
            '5579734916388215808' # PC
        ]

    if sector == 7:
        ids_to_test = [
            '3050033749239975552', # PC for vetting report description doc
            # '5596735638203997824' # PC
        ]

    if sector == 8:
        ids_to_test = [
            '5514749866474194816'
            # '5423913124931271424', # EB w/ rot + dips
            # '5290781443841554432',
            #'5510676828723793920' # PC
        ]

    if sector == 9:
        ids_to_test = [
            '5514749866474194816'
            # '5246508676134913024',
            # '5489726768531119616', # TIC 268
            # '5326491313765089792' # detrending debugger
            # '5256717749007641344', # PC near TOI 684
            # '5523104093269203712', # PC
            # '5489726768531119616', # PC
            # '5432321060287733888', # PC
            # '5256502966294604800', # PC
            # '5524965566445719168', # EB offtarget
            # '4666329826478795008', # contact binary
            # '5436252604630719488', # non-CM from parallax
        ]

    if sector == 10:
        ids_to_test = [
            '5246508676134913024',
            # '5334408965769940608', # detrending debugger
            # '5290781443841554432',
            #'5251470948229949568' # TOI-837
        ]

    if sector == 11:
        ids_to_test = [
            '5246508676134913024',
            #'5339389268061191040',
            #'5838450865699668736',
            #'5245968236116294016',
            #'6113920619134019456'
        ]

    if sector == 12:
        ids_to_test = [
            '5989314684373466624',
            # '5938237284044738944',
            # '4054631994268958208', # dias nonCM
            # '5832122489147732352', # kharchenko nonCM
            # '5851066108057126912', # dias
            # '5851519283006720000', # ditto  and zari ums
            # '6035621578633168768', # zari ums
            # '5887003203290910592',
            # '5910255056670150656',
            # '5934234168384721664',
            # '5997809579916196608',
            # '6024737031987685632',
            # '5899727920059348352',
            # '5981521449025600256',
            # '6011518462675791872',
        ]

    if sector == 14:
        ids_to_test = [
            '2028660942571841152',
            #'2069071980566847232', # verify group matching is working
            #'2030626834970610688', # should be autoclassed nottransit
            #'2038548472649717120', # centroid page all wonky
            #'1048109222955337600',
            # '4516549232971506560', # PC; not auto-classified
            # '1821207356364973568', # "notransit"
            # '2021549198278632320', # iterative search through neighborhood
        ]

    for id_to_test in ids_to_test:

        existing_report_files = glob(
            'test_vetting_reports/*/*/*{}*pdf'.format(id_to_test)
        )
        if len(existing_report_files) > 0:
            for f in existing_report_files:
                os.remove(f)
                print('removing {} to make for test_vetting_report'.format(f))

        paths = [f for f in lc_paths if id_to_test in f]

        mavp.make_all_vetting_reports(
            paths, lcbasedir, resultsdir, cdips_df, supplementstatsdf, pfdf,
            toidf, show_rvs=True, sector=sector
        )


if __name__ == "__main__":
    test_vetting_report(sector=14, cdips_cat_vnum=0.6)
