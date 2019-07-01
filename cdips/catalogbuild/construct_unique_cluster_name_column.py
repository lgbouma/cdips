"""
we need some idea of the cluster names.
"""

#TODO write ... and generalize to match between pg 5 and this function...

from glob import glob
import os, textwrap, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as const

from itertools import repeat

from cdips.utils import collect_cdips_lightcurves as ccl
from astropy.io.votable import from_table, writeto, parse

clusterdir = "/home/lbouma/proj/cdips/data/cluster_data"

DEBUG = True

def main(cdips_cat_vnum=0.3):

    # cluster;dec;dist;ext_catalog_name;parallax;phot_bp_mean_mag;phot_g_mean_mag;phot_rp_mean_mag;pmdec;pmra;ra;reference;source_id
    cdips_df = ccl.get_cdips_catalog(ver=cdips_cat_vnum)

    # Name,Type,MWSC,ra,dec, etc.
    k13 = get_k13_df()

    sourceid = nparr(cdips_df['source_id'])
    clusterarr = nparr(cdips_df['cluster'])
    ras = nparr(cdips_df['ra'])
    decs = nparr(cdips_df['dec'])
    referencearr = nparr(cdips_df['reference'])

    if DEBUG:
        np.random.seed(49)

        #inds = np.random.choice(np.arange(0, len(sourceid)), size=5000, replace=False)
        uarr, inds = np.unique(clusterarr, return_index=True)

        sourceid = sourceid[inds]
        clusterarr = clusterarr[inds]
        ras = ras[inds]
        decs = decs[inds]
        referencearr = referencearr[inds]

    res = list( map(get_k13_name_match,
                    zip(clusterarr, ras, decs, referencearr, repeat(k13)))
              )

    resdf = pd.DataFrame(res, columns=['k13_name_match', 'how_match', 'have_name_match',
                                       'have_mwsc_id_match',
                                       'is_known_asterism', 'not_in_k13',
                                       'why_not_in_k13'])
    resdf['source_id'] = sourceid

    outpath = (
        '/nfs/phtess1/ar1/TESS/PROJ/lbouma/OC_MG_FINAL_v{}_with_K13_name_match.csv'.
        format(cdips_cat_vnum)
    )
    outdf = resdf.merge(cdips_df, how='left', on='source_id')
    outdf.to_csv(outpath, index=False, sep=';')
    print('made {}'.format(outpath))

    # FIXME
    # TODO: still need to construct MY unique cluster name. as much as possible,
    # this is the K13 name. however for any Gulliver objects, or things for
    # which it's not in K13 but is known to be missing (Hyades, RSG1/7/8, ..), we
    # still want to list names.



def get_k13_df():

    getfile = os.path.join(clusterdir,'Kharchenko_2013_MWSC.vot')
    vot = parse(getfile)
    tab = vot.get_first_table().to_table()
    k13 = tab.to_pandas()
    del tab
    k13['Name'] = k13['Name'].str.decode('utf-8')
    k13['Type'] = k13['Type'].str.decode('utf-8')
    k13['MWSC'] = k13['MWSC'].str.decode('utf-8')

    return k13


def get_k13_name_match(task):
    """
    call in for loop, or in map, once you have cdips_df['cluster'] type column

    args:
        originalname : name from arbitrary reference of cluster
        target_ra : in decimal deg
        target_dec
        reference: source of cluster membership
        k13: dataframe of Kharchenko+2013 result

    returns:
        (matchname, have_name_match, have_mwsc_id_match, is_known_asterism,
        not_in_k13, why_not_in_k13)
    """

    originalname, target_ra, target_dec, reference, k13 = task

    cluster = originalname

    have_name_match = False
    have_mwsc_id_match = False
    is_known_asterism = False
    how_match = np.nan
    why_not_in_k13 = ''

    if pd.isnull(originalname):
        return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                is_known_asterism, True, 'no_cluster_name')

    if reference in ['Zari_2018_PMS','Zari_2018_UMS']:

        not_in_k13 = True
        is_zari_ums = False

        if 'Zari_2018_UMS' in reference:
            is_zari_ums = True
        is_zari_pms = False
        if 'Zari_2018_PMS' in reference:
            is_zari_pms = True

        if is_zari_pms:
            why_not_in_k13 = 'is_zari_pms'
        elif is_zari_ums:
            why_not_in_k13 = 'is_zari_ums'

        return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                is_known_asterism, not_in_k13, why_not_in_k13)

    is_gaia_member = False
    if not pd.isnull(cluster):
        if ( ('Gulliver' in cluster and 'CantatGaudin_2018'
              in reference)
            or
             ('Teutsch_12' in cluster and
              'CantatGaudin_2018' in reference)
           ):
            is_gaia_member = True
    if is_gaia_member:
        not_in_k13 = True
        why_not_in_k13 = 'is_gaia_member'
        return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                is_known_asterism, not_in_k13, why_not_in_k13)

    # Hyades: I checked in a distance sort that it's not in K13. Probably b/c
    #  it's very extended.
    # RSG7,8: Soubiran+2018 note "RSG 7 and RSG 8, two OCs recently found by
    #  Röser et al. (2016), are confirmed as two separate OCs very close in
    #  space and motion." So they were found after K13.
    known_missing_clusters = ['Hyades', 'RSG_7', 'RSG_8', 'RSG_1', "RSG_5"]

    for c in cluster.split(','):
        if c in known_missing_clusters:
            not_in_k13 = True
            why_not_in_k13 = 'known missing from K13'
            how_match = 'manual_check'
            return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                    is_known_asterism, not_in_k13, why_not_in_k13)

    #
    # special formatting cases:
    # * ESO clusters like "ESO_129-32" (K13 format) are read from Dias as
    #   "ESO_129_32".
    # * Pozzo 1 is Vel OB2
    # * Majaess50 is 10' from FSR0721, from D. Majaess (2012).
    # * Majaess58 is 6' from FSR0775, from D. Majaess (2012).
    # * Ditto for a few from CG2018...
    # * weird "-" to "_" for Alessi-Teutsch 10.
    # * "Collinder 34" is a sub-cluster of the larger Collinder 33 nebula, not
    #   appreciably different from SAI-24 either. (Saurin+ 2015,
    #   https://arxiv.org/pdf/1502.00373.pdf)
    # * Some catalogs, eg., Cantat-Gaudin's, replaced "vdBergh-Hagen" with "BH"
    # * Coma Star Cluster in Coma Berenices is Melotte 111
    #
    if cluster.startswith('ESO_'):
        if '-' not in cluster:
            _cid = cluster.split('_')[1] + '-' + cluster.split('_')[2]
            cluster = 'ESO_'+_cid
    if 'Pozzo_1' in cluster:
        cluster = 'Vel_OB2'
        how_match = 'manual_override'
    if 'Majaess 50' in cluster:
        cluster = 'FSR_0721'
        how_match = 'manual_override'
    if 'Majaess 58' in cluster:
        cluster = 'FSR_0775'
        how_match = 'manual_override'
    if 'Alessi Teutsch' in cluster:
        # e.g., 'Alessi Teutsch 10'
        cluster = 'Alessi-Teutsch_' + cluster.split(' ')[-1]
        how_match = 'string_match'
    if 'Alessi_Teutsch_5' in cluster:
        # underscores mess up the duplicate name list
        cluster = "ASCC 118"
        how_match = 'string_match'
    #FIXME FIXME: GENERALIZE BELOW TO ALL "CHARACTER CHARACTER NUMBER" TYPE
    # STRINGS. ALSO DO 'Aveni_Hunter_1,nan,Aveni_Hunter_1,nan' style match.
    if 'Aveni Hunter 1' in cluster:
        cluster = "Aveni-Hunter_1"
        how_match = 'string_match'
    if 'Collinder 34' in cluster:
        cluster = 'Collinder_33'
        how_match = 'manual_override'
    if 'BH' in cluster:
        cluster = cluster.replace("BH","vdBergh-Hagen")
    if 'ComaBer' in cluster:
        cluster = cluster.replace('ComaBer','Melotte_111')
        how_match = 'manual_override'
    if 'ScoOB2' in cluster:
        cluster = cluster.replace('ScoOB2','Sco_OB2')
        how_match = 'string_match'
    if "BDSB" in cluster:
        if ' ' not in cluster and "_" not in cluster:
            cluster = cluster.replace("BDSB","BDSB_")
            how_match = 'string_match'

    #
    # initial normal match: try matching against replacing spaces with
    # underscores
    #
    clustersplt = cluster.split(',')
    trystrs = []
    for c in clustersplt:
        trystrs.append(c)
        trystrs.append(c.replace(' ','_'))

    for trystr in trystrs:
        if trystr in nparr(k13['Name']):
            have_name_match=True
            name_match = trystr
            how_match = 'string_match'
            break

    #
    # try if SIMBAD's name matcher has anything.
    #
    if not have_name_match:
        for c in clustersplt:
            res = Simbad.query_objectids(c)
            try:
                resdf = res.to_pandas()
            except AttributeError:
                print('{}: simbad no matches'.format(c))
                continue

            resdf['ID'] = resdf['ID'].str.decode('utf-8')
            smatches = nparr(resdf['ID'])

            # some names have format 'Name M 42'
            clean_smatches = [s.lstrip('NAME ') for s in smatches]
            # some names have format '[KPS2012] MWSC 0531'
            for ix, s in enumerate(clean_smatches):
                strm = re.search("\[.*\]\ ", s)
                if strm is not None:
                    clean_smatches[ix] = s.lstrip(strm.group())
            # some names have variable length whitespace... e.g., 'NGC  2224'

            # first set of attempts: everything in clean matches (irrespective if
            # MWSC number exists)
            trystrs = []
            for _c in clean_smatches:
                trystrs.append(_c)
                trystrs.append(_c.replace(' ','_'))

            for trystr in trystrs:
                if trystr in nparr(k13['Name']):
                    have_name_match = True
                    how_match = 'SIMBAD_name_match'
                    name_match = trystr
                    break

            # only then: check if you have MWSC identifier.
            inds = ['MWSC' in _c for _c in clean_smatches]
            mwsc_match = nparr(clean_smatches)[inds]
            if len(mwsc_match) > 1:
                pass
            if len(mwsc_match) == 1:
                have_mwsc_id_match = True
                mwsc_id_match = mwsc_match[0].replace('MWSC ','')
            if len(mwsc_match) == 0:
                pass

            if have_mwsc_id_match:
                break

    #
    # if you got mwsc id above, use the index table to convert to a name
    #
    if have_mwsc_id_match and not have_name_match:

        Vizier.ROW_LIMIT = -1
        catalog_list = Vizier.find_catalogs('J/A+A/558/A53')
        catalogs = Vizier.get_catalogs(catalog_list.keys())
        t = catalogs[1]
        df_index = t.to_pandas()
        del t

        # columns are MWSC,Name,flag,Type,n_Type,Src,SType,N
        for c in df_index.columns:
            if c != 'N':
                df_index[c] = df_index[c].str.decode('utf-8')

        mapd = {
            "=": "cluster parameters are determined",
            ":": ("possibly this is a cluster, but parameters are not "
                  "determined"),
            "-": "this is not a cluster",
            "&": "duplicated/coincides with other cluster"
        }

        _k13 = df_index[df_index['MWSC'] == mwsc_id_match]

        have_name_match=True
        how_match = 'SIMBAD_MWSCID_match'
        name_match = _k13['Name'].iloc[0]

        flag = str(_k13['flag'].iloc[0])
        if flag in [":","-","&"]:
            is_in_index = True
            why_not_in_k13 = "K13index: "+mapd[flag]
        else:
            pass

    #
    # check against the larger kharchenko index, which includes the clusters
    # that they found dubious, or didn't report parameters for.
    #
    is_in_index = False
    if not have_name_match and not have_mwsc_id_match:

        Vizier.ROW_LIMIT = -1
        catalog_list = Vizier.find_catalogs('J/A+A/558/A53')
        catalogs = Vizier.get_catalogs(catalog_list.keys())
        t = catalogs[1]
        df_index = t.to_pandas()
        del t

        # columns are MWSC,Name,flag,Type,n_Type,Src,SType,N
        for c in df_index.columns:
            if c != 'N':
                df_index[c] = df_index[c].str.decode('utf-8')

        mapd = {
            "=": "cluster parameters are determined",
            ":": ("possibly this is a cluster, but parameters are not "
                  "determined"),
            "-": "this is not a cluster",
            "&": "duplicated/coincides with other cluster"
        }

        clustersplt = cluster.split(',')
        trystrs = []
        for c in clustersplt:
            trystrs.append(c)
            trystrs.append(c.replace(' ','_'))
        for trystr in trystrs:
            if trystr in nparr(df_index['Name']):
                have_name_match=True
                how_match = 'K13_index_table_string_match'
                name_match = trystr
                is_in_index = True

                _k13 = df_index.loc[df_index['Name'] == name_match]
                flag = str(_k13['flag'].iloc[0])

                if flag not in [":","-","&"]:
                    raise NotImplementedError('why do u match here, not '
                                              'earlier?')
                else:
                    why_not_in_k13 = "K13index: "+mapd[flag]

                break


    #
    # try searching K13 within circles of 5,10 arcminutes of the quoted
    # position. if more than 1 match, omit (to not get false name matches).  if
    # only 1 match, use the name.  (this introduces some false matches,
    # probably...)
    #
    if not have_name_match and not have_mwsc_id_match:

        ra,dec = target_ra, target_dec

        c = SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs')

        k13_c = SkyCoord(nparr(k13['RAJ2000']), nparr(k13['DEJ2000']),
                         frame='icrs', unit=(u.deg,u.deg))

        seps = k13_c.separation(c)

        CUTOFFS = [5*u.arcmin, 10*u.arcmin]

        for CUTOFF in CUTOFFS:
            cseps = seps < CUTOFF

            if len(cseps[cseps]) == 1:
                have_name_match=True
                how_match = 'K13_spatial_match_lt_{}arcmin'.format(CUTOFF.value)
                name_match = k13.loc[cseps, 'Name'].iloc[0]

            elif len(cseps[cseps]) > 1:
                print('got too many matches within {} arcminutes!'.
                      format(CUTOFF))
                pass

            else:
                pass

            if have_name_match:
                break

    #
    # if after all this, no match, check double cluster names list, and recall
    # this function
    #
    if not have_name_match and not have_mwsc_id_match:
        ddf = pd.read_csv(
            os.path.join(clusterdir, 'double_names_WEBDA_20190606.csv'),
            sep=';')
        for c in clustersplt:
            if len(ddf[ddf['cluster_name'] == c.replace('_',' ')]) == 1:
                adopted_name = (
                    ddf[ddf['cluster_name'] ==
                        c.replace('_',' ')].iloc[0]['adopted_name']
                )
                return get_k13_name_match((adopted_name, target_ra, target_dec,
                                           reference, k13))

    #if cluster == 'Alessi 40,Alessi 40':
    #    import IPython; IPython.embed()

    #
    # Check against known asterisms
    #
    is_known_asterism = False
    for c in clustersplt:
        # Baumgardt 1998.
        if c in ['Collinder 399', 'Upgren 1', 'NGC 1252', 'Melotte 227']:
            is_known_asterism = True
            break

        if 'NGC' in c:
            getfile = '/nfs/phn12/ar0/H/PROJ/lbouma/cdips/data/cluster_data/Sulentic_1973_NGC_known_asterisms.vot'
            vot = parse(getfile)
            tab = vot.get_first_table().to_table()
            ddf = tab.to_pandas()
            del tab

            ngc_asterisms = nparr(ddf['NGC'])

            if c.startswith('NGC '):
                c = c.lstrip('NGC ')
            elif c.startswith('NGC_'):
                c = c.lstrip('NGC_')
            elif c.startswith('NGC'):
                c = c.lstrip('NGC')

            # https://en.wikipedia.org/wiki/NGC_2451
            if c.endswith('A'):
                c = c.rstrip('A')
            if c.endswith('B'):
                c = c.rstrip('B')

            if int(c) in ngc_asterisms:
                is_known_asterism = True
                break

            # NGC 1252 was also by Baumgardt 1998.
            # identified by Kos+2018, MNRAS 480 5242-5259 as asterisms
            if int(c) in [1252, 6994, 7772, 7826]:
                is_known_asterism = True
                break

    is_gagne_mg = False
    if 'Gagne' in reference:
        is_gagne_mg = True
        why_not_in_k13 = 'is_gagne_mg'

    is_oh_mg = False
    if 'Oh' in reference:
        is_oh_mg = True
        why_not_in_k13 = 'is_oh_mg'

    is_rizzuto_mg = False
    if 'Rizzuto' in reference:
        is_rizzuto_mg = True
        why_not_in_k13 = 'is_rizzuto_mg'

    not_in_k13 = False
    if (
        (is_gagne_mg or is_oh_mg or is_rizzuto_mg or is_in_index)
        and
        not have_name_match
    ):
        not_in_k13 = True
        name_match = np.nan

    #
    # finally, if we failed to get matches above, (e.g., for some of the IR
    # Majaess clusters), skip
    #
    for c in cluster.split(','):
        if "Majaess" in c:
            not_in_k13 = True
            why_not_in_k13 = 'Majaess IR cluster match missing in K13'
            how_match = 'majaess_flag'
            return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                    is_known_asterism, not_in_k13, why_not_in_k13)



    try:
        return (name_match, how_match, have_name_match, have_mwsc_id_match,
                is_known_asterism, not_in_k13, why_not_in_k13)
    except UnboundLocalError:
        print('got UnboundLocalError for {} ({})'.format(
            repr(cluster), repr(reference)))
        import IPython; IPython.embed()

if __name__ == "__main__":
    main()
