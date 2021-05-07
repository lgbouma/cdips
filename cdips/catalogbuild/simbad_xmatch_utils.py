"""
Directly parse the SIMBAD database for known stars of various types.

Contents
--------

run_SIMBAD_to_csv: get defaults known stars for CDIPS v0.5. TT*, Y*O, Y*?, TT?,
    Pl, pMS*

SIMBAD_otype_to_GaiaDR2_csv: given an otype[s], get all SIMBAD matches.

SIMBAD_longotype_to_GaiaDR2_csv: given an otype[s], segment sky into 20 chunks,
    get SIMBAD matches in each chunk, and merge them.  This works provided
    there are less than 20,000 of the objects per chunk (i.e., 400k objects in
    the SIMBAD database).

SIMBAD_bibcode_to_GaiaDR2_csv: given a Bibcode, get all SIMBAD matches.
"""

import os
import numpy as np, pandas as pd

from astropy.table import Table
from cdips.paths import DATADIR
clusterdir = os.path.join(DATADIR, 'cluster_data')

from cdips.utils import today_YYYYMMDD

def run_SIMBAD_to_csv(
    otype_list="TT*,Y*O,Y*?,TT?,Pl,pMS*",
    otypes_list="EB*",
    get_longtypes=0
):
    """
    Get the simbad object classes specified in otype and otypes (comma
    separated lists).

    Per https://github.com/astropy/astroquery/issues/1230, there is a 20,000
    hard row limit imposed at the CDS side. So e.g., "Y*O", "Y*?" classes hit
    this limit. "EB*" does too. This can be avoided by segmenting over sky
    regions.

    Details:

    Object types build a hierarchy defined here:
        https://simbad.u-strasbg.fr/simbad/sim-display?data=otypes

    Each astronomical object has a main type defined, and several other types
    generally infered from its identifiers.

    Therefore, there are four different object type criterions:

        `maintype` and `otype` looks for objects having exactly the requested type,
        either as the main type or any type.
        `maintypes` and `otypes` retrieves all objects having the given type or any
        type underneath the given one in its hierarchy sub-tree.

    """
    otype_list = otype_list.split(',')
    otypes_list = otypes_list.split(',')
    outdir = os.path.join(clusterdir, 'v05')

    # these types break the "20k per row" rule when tiling the sky in 10, or
    # even 20 pieces.
    LONG_TYPES = ["Y*O", "Y*?", "EB*"]

    if get_longtypes:

        for otype in ["Y*O", "Y*?"]:
            SIMBAD_longotype_to_GaiaDR2_csv(otype, outdir)

        for otypes in ["EB*"]:
            SIMBAD_longotype_to_GaiaDR2_csv(otypes, outdir, is_otypes=1)

    else:

        for otype in otype_list:
            if otype not in LONG_TYPES:
                SIMBAD_otype_to_GaiaDR2_csv(otype, outdir)

        for otypes in otypes_list:
            if otypes not in LONG_TYPES:
                SIMBAD_otype_to_GaiaDR2_csv(otypes, outdir, is_otypes=1)


def SIMBAD_longotype_to_GaiaDR2_csv(otype, outdir, is_otypes=0):
    """
    Given an otype[s], segment sky into 40 chunks, get SIMBAD matches in each
    chunk, and merge them.

    This works provided there are less than 20,000 of the objects per chunk
    (i.e., 800k objects in the SIMBAD database).
    """

    from astroquery.simbad import Simbad

    ROW_LIMIT = int(1e5)

    TIMEOUT_S = 300
    Simbad.ROW_LIMIT = ROW_LIMIT
    Simbad.TIMEOUT = TIMEOUT_S

    # return all the ids
    Simbad.add_votable_fields('typed_id')
    Simbad.add_votable_fields('ids')

    ra_left = np.arange(0, 360, 9)
    ra_right = np.arange(9, 360+9, 9)

    outname = (
        'SIMBAD_' + otype.replace("*", "S") + "_" + today_YYYYMMDD() +
        f"_longMerged.csv"
    )
    mergedoutpath = os.path.join(outdir, outname)

    if not os.path.exists(mergedoutpath):

        outpaths = []
        ix = 0
        for _l, _r in zip(ra_left, ra_right):

            querystr = f"ra >= {_l} & ra < {_r}"
            print(79*'-')
            print(f'{ix} Beginning SIMBAD query for {querystr} otype[s]={otype}...')

            outname = (
                'SIMBAD_' + otype.replace("*", "S") + "_" + today_YYYYMMDD() +
                f"_longchunk_ra{_l}-{_r}.csv"
            )
            outpath = os.path.join(outdir, 'longchunks', outname)

            if not is_otypes:
                result = Simbad.query_criteria(querystr, otype=otype)
            else:
                result = Simbad.query_criteria(querystr, otypes=otype)
            assert len(result) < ROW_LIMIT
            assert len(result) != 20000

            df = Table(result).to_pandas()

            sel = df.IDS.str.contains("Gaia DR2")

            print(f'Got {len(df)} {otype} results from SIMBAD')
            sdf = df[sel]
            print(f'...and {len(sdf)} {otype} of them have Gaia DR2 identifiers')

            sdf["source_id"] = sdf.IDS.str.extract("Gaia\ DR2\ (\d*)")

            sdf.to_csv(outpath, index=False)
            print(f'Wrote {outpath}')

            outdf = sdf["source_id"]
            outpath = outpath.replace('.csv', '_cut_source.csv')
            outdf.to_csv(outpath, index=False)
            print(f'Wrote {outpath}')
            outpaths.append(outpath)

            ix +=1

        df = pd.concat([pd.read_csv(f) for f in outpaths])
        df.to_csv(mergedoutpath, index=False)


    else:
        print(f'Found {mergedoutpath}')



def SIMBAD_otype_to_GaiaDR2_csv(otype, outdir, is_otypes=0):
    """
    Given an otype[s], get all SIMBAD matches.
    """

    from astroquery.simbad import Simbad

    ROW_LIMIT = int(1e5)

    TIMEOUT_S = 300
    Simbad.ROW_LIMIT = ROW_LIMIT
    Simbad.TIMEOUT = TIMEOUT_S

    # return all the ids
    Simbad.add_votable_fields('typed_id')
    Simbad.add_votable_fields('ids')

    outname = 'SIMBAD_'+otype.replace("*", "S")+"_"+today_YYYYMMDD()+".csv"
    outpath = os.path.join(outdir, outname)

    if not os.path.exists(outpath):

        if not is_otypes:
            result = Simbad.query_criteria(otype=otype)
        else:
            result = Simbad.query_criteria(otypes=otype)
        assert len(result) < ROW_LIMIT
        assert len(result) != 20000

        df = Table(result).to_pandas()

        sel = df.IDS.str.contains("Gaia DR2")

        print(f'Got {len(df)} {otype} results from SIMBAD')
        sdf = df[sel]
        print(f'...and {len(sdf)} {otype} of them have Gaia DR2 identifiers')

        sdf["source_id"] = sdf.IDS.str.extract("Gaia\ DR2\ (\d*)")

        sdf.to_csv(outpath, index=False)
        print(f'Wrote {outpath}')

        outdf = sdf["source_id"]
        outname = (
            'SIMBAD_' +otype.replace("*", "S")+"_"
            +today_YYYYMMDD()+"_cut_source.csv"
        )
        outpath = os.path.join(outdir, outname)

        outdf.to_csv(outpath, index=False)
        print(f'Wrote {outpath}')

    else:
        print(f'Found {outpath}')


def SIMBAD_bibcode_to_GaiaDR2_csv(bibcode, outdir):
    """
    Given a bibcode, get all SIMBAD matches with Gaia DR2 source_id's.
    e.g., "2016ApJS..225...15C"
    """

    from astroquery.simbad import Simbad

    ROW_LIMIT = int(3e4)

    TIMEOUT_S = 300
    Simbad.ROW_LIMIT = ROW_LIMIT
    Simbad.TIMEOUT = TIMEOUT_S

    # return all the ids
    Simbad.add_votable_fields('typed_id')
    Simbad.add_votable_fields('ids')

    outname = 'SIMBAD_'+bibcode+".csv"
    outpath = os.path.join(outdir, outname)

    if not os.path.exists(outpath):

        result = Simbad.query_bibobj(bibcode)

        assert len(result) < ROW_LIMIT
        assert len(result) != 20000

        df = Table(result).to_pandas()

        sel = df.IDS.str.contains("Gaia DR2")

        print(f'Got {len(df)} {bibcode} results from SIMBAD')
        sdf = df[sel]
        print(f'...and {len(sdf)} {bibcode} of them have Gaia DR2 identifiers')

        sdf["source_id"] = sdf.IDS.str.extract("Gaia\ DR2\ (\d*)")

        sdf.to_csv(outpath, index=False)
        print(f'Wrote {outpath}')

        outdf = sdf["source_id"]
        outname = (
            'SIMBAD_' + bibcode + "_" + "_cut_source.csv"
        )
        outpath = os.path.join(outdir, outname)

        outdf.to_csv(outpath, index=False)
        print(f'Wrote {outpath}')

    else:
        print(f'Found {outpath}')
