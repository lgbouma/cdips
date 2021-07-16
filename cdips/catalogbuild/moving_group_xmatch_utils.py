# -*- coding: utf-8 -*-
"""
functions to crossmatch moving group catalogs with Gaia-DR2.

called from `homogenize_cluster_lists.py`

includes:

    make_vizier_GaiaDR2_crossmatch
        make_Kraus14_GaiaDR2_crossmatch
        make_Roser11_GaiaDR2_crossmatch
        make_Luhman12_GaiaDR2_crossmatch
    make_votable_given_cols
    make_votable_given_full_cols
    Tian2020_to_csv
    Kerr2021_to_csv
    Gagne2020_to_csv

DEPRECATED IN >=V0.5 (in favor of SIMBAD_bibcode_to_GaiaDR2_csv):
    make_Gagne18_BANYAN_any_DR2_crossmatch
        make_Gagne18_BANYAN_XIII_GaiaDR2_crossmatch
        make_Gagne18_BANYAN_XII_GaiaDR2_crossmatch
        make_Gagne18_BANYAN_XI_GaiaDR2_crossmatch
    make_Rizzuto11_GaiaDR2_crossmatch
    make_Oh17_GaiaDR2_crossmatch
    make_Rizzuto15_GaiaDR2_crossmatch
    make_Preibisch01_GaiaDR2_crossmatch
    make_Casagrande_11_GaiaDR2_crossmatch
    make_Bell17_GaiaDR2_crossmatch
"""
from __future__ import division, print_function

import numpy as np, pandas as pd

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.io.votable import from_table, writeto, parse

from astroquery.gaia import Gaia

from astroquery.vizier import Vizier

import sys, os, re, itertools, subprocess
from glob import glob
from numpy import array as arr


from cdips.paths import DATADIR
clusterdatadir = os.path.join(DATADIR, 'cluster_data')

datadir = os.path.join('moving_groups')

def make_vizier_GaiaDR2_crossmatch(vizier_search_str, ra_str, dec_str,
                                   pmra_str, pmdec_str, name_str, assoc_name,
                                   table_num=0, outname='', maxsep=10,
                                   outdir=datadir,
                                   homedir='/home/luke/' ):
    '''
    Spatially crossmatch catalog of <~100,000 members w/ coords and PMs against
    Gaia DR2.  This assumes that the catalog is on vizier.

    make_Kraus14_GaiaDR2_crossmatch is an example of a call.
    '''

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs(vizier_search_str)
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    tab = catalogs[table_num]
    print(42*'-')
    print('{}'.format(outname))
    print('initial number of members: {}'.format(len(tab)))

    # gaia xmatch need these two column names
    RA = tab[ra_str]
    dec = tab[dec_str]
    pm_RA = tab[pmra_str]
    pm_dec = tab[pmdec_str]
    name = tab[name_str]

    assoc = np.repeat(assoc_name, len(RA))

    assert tab[ra_str].unit == u.deg
    assert tab[pmra_str].unit == u.mas/u.year

    xmatchoutpath = outname.replace('.csv','_MATCHES_GaiaDR2.csv')
    outfile = outname.replace('.csv','_GOTMATCHES_GaiaDR2.xml')
    xmltouploadpath = outname.replace('.csv','_TOUPLOAD_GaiaDR2.xml')

    # do the spatial crossmatch...
    if os.path.exists(outfile):
        os.remove(outfile)
    if not os.path.exists(outfile):
        _ = make_votable_given_cols(name, assoc, RA, dec, pm_RA, pm_dec,
                                    outpath=xmltouploadpath)

        Gaia.login(credentials_file=os.path.join(homedir, '.gaia_credentials'))

        # separated less than 10 arcsec.
        jobstr = (
        '''
        SELECT TOP {ncut:d} u.name, u.assoc, u.ra, u.dec, u.pm_ra, u.pm_dec,
        g.source_id, DISTANCE(
          POINT('ICRS', u.ra, u.dec),
          POINT('ICRS', g.ra,g.dec)) AS dist,
          g.phot_g_mean_mag as gaia_gmag,
          g.pmra AS gaia_pmra,
          g.pmdec AS gaia_pmdec
        FROM tap_upload.foobar as u, gaiadr2.gaia_source AS g
        WHERE 1=CONTAINS(
          POINT('ICRS', u.ra, u.dec),
          CIRCLE('ICRS', g.ra, g.dec, {sep:.8f})
        )
        '''
        )
        maxncut = int(5*len(name)) # to avoid query timeout
        maxsep = (maxsep*u.arcsec).to(u.deg).value
        query = jobstr.format(sep=maxsep, ncut=maxncut)

        if not os.path.exists(outfile):
            # might do async if this times out. but it doesn't.
            j = Gaia.launch_job(query=query,
                                upload_resource=xmltouploadpath,
                                upload_table_name="foobar", verbose=True,
                                dump_to_file=True, output_file=outfile)

        Gaia.logout()

    vot = parse(outfile)
    tab = vot.get_first_table().to_table()

    if maxncut - len(tab) < 10:
        errmsg = 'ERROR! too many matches'
        raise AssertionError(errmsg)

    print('number of members after gaia 10 arcsec search: {}'.format(len(tab)))

    # if nonzero and finite proper motion, require Gaia pm match to sign
    # of stated PMs.
    df = tab.to_pandas()

    print('\n'+42*'-')

    sel = (df['gaia_gmag'] < 18)
    print('{} stars in sep < 10 arcsec, G<18, xmatch'.format(len(df[sel])))

    sel &= (
        (   (df['pm_ra'] != 0 ) & (df['pm_dec'] != 0 ) &
            ( np.sign(df['pm_ra']) == np.sign(df['gaia_pmra']) ) &
            ( np.sign(df['pm_dec']) == np.sign(df['gaia_pmdec']) )
        )
        |
        (
            (df['pm_ra'] == 0 ) & (df['pm_dec'] == 0 )
        )
    )
    df = df[sel]
    print('{} stars in sep < 10 as xmatch, G<18, after pm cut (xor zero pm)'.
          format(len(df)))

    # make multiplicity column. then sort by name, then by distance. then drop
    # name duplicates, keeping the first (you have nearest neighbor saved!)
    _, inv, cnts = np.unique(df['name'], return_inverse=True,
                             return_counts=True)

    df['n_in_nbhd'] = cnts[inv]

    df['name'] = df['name'].str.decode('utf-8')
    df['assoc'] = df['assoc'].str.decode('utf-8')

    df = df.sort_values(['name','dist'])

    df = df.drop_duplicates(subset='name', keep='first')

    df['source_id'] = df['source_id'].astype('int64')

    print('{} stars after above cuts + chosing nearest nbhr by spatial sep'.
          format(len(df)))

    df.to_csv(xmatchoutpath, index=False)
    print('made {}'.format(xmatchoutpath))
    print(79*'=')


def make_votable_given_cols(name, assoc, RA, dec, pm_RA, pm_dec, outpath=None):

    t = Table()
    t['name'] = name.astype(str)
    t['assoc'] = assoc.astype(str)
    t['ra'] = RA*u.deg
    t['dec'] = dec*u.deg
    t['pm_ra'] = pm_RA*(u.mas/u.year)
    t['pm_dec'] = pm_dec*(u.mas/u.year)

    votable = from_table(t)

    writeto(votable, outpath)
    print('made {}'.format(outpath))

    return outpath

def make_votable_given_full_cols(name, assoc, RA, dec, pm_RA, pm_dec, u_pm_RA,
                            u_pm_dec, outpath=None):

    t = Table()
    t['name'] = name.astype(str)
    t['assoc'] = assoc.astype(str)
    t['ra'] = RA*u.deg
    t['dec'] = dec*u.deg
    t['pm_ra'] = pm_RA*(u.mas/u.year)
    t['pm_dec'] = pm_dec*(u.mas/u.year)
    t['err_pm_ra'] = u_pm_RA*(u.mas/u.year)
    t['err_pm_dec'] = u_pm_dec*(u.mas/u.year)

    votable = from_table(t)

    writeto(votable, outpath)
    print('made {}'.format(outpath))

    return outpath


def make_Gagne18_BANYAN_XIII_GaiaDR2_crossmatch():

    # Downloaded direct from
    # http://iopscience.iop.org/0004-637X/862/2/138/suppdata/apjaaca2et2_mrt.txt
    # God I wish vizier were a thing.
    tablepath = os.path.join(datadir,
                             'Gagne_2018_BANYAN_XIII_apjaaca2et2_mrt.txt')

    make_Gagne18_BANYAN_any_DR2_crossmatch(
        tablepath, namestr='Gagne_2018_BANYAN_XIII_GaiaDR2_crossmatched',
        maxsep=10)


def make_Gagne18_BANYAN_XII_GaiaDR2_crossmatch():

    # Downloaded direct from
    # http://iopscience.iop.org/0004-637X/860/1/43/suppdata/apjaac2b8t4_mrt.txt
    tablepath = os.path.join(datadir,
                             'Gagne_2018_BANYAN_XII_apjaac2b8t4_mrt.txt')

    make_Gagne18_BANYAN_any_DR2_crossmatch(
        tablepath, namestr='Gagne_2018_BANYAN_XII_GaiaDR2_crossmatched',
        maxsep=10)


def make_Gagne18_BANYAN_XI_GaiaDR2_crossmatch():

    # Downloaded direct from
    # http://iopscience.iop.org/0004-637X/856/1/23/suppdata/apjaaae09t5_mrt.txt
    tablepath = os.path.join(datadir, 'Gagne_2018_apjaaae09t5_mrt.txt')

    make_Gagne18_BANYAN_any_DR2_crossmatch(
        tablepath, namestr='Gagne_2018_BANYAN_XI_GaiaDR2_crossmatched',
        maxsep=10)

def make_Gagne18_BANYAN_any_DR2_crossmatch(
        tablepath,
        namestr=None,
        maxsep=10,
        outdir=datadir,
        homedir='/home/luke/'):
    """
    J. Gagne's tables have a particular format that requires some wrangling.
    Also, since so many of the stars are high PM, the spatial cross-matches
    will be crap unless we also include PM information in the matching.

    Do the matching via astroquery's Gaia API.
    """
    assert type(namestr) == str
    t = Table.read(tablepath, format='ascii.cds')

    RAh, RAm, RAs = arr(t['RAh']), arr(t['RAm']), arr(t['RAs'])

    RA_hms =  [str(rah).zfill(2)+'h'+
               str(ram).zfill(2)+'m'+
               str(ras).zfill(2)+'s'
               for rah,ram,ras in zip(RAh, RAm, RAs)]

    DEd, DEm, DEs = arr(t['DEd']),arr(t['DEm']),arr(t['DEs'])
    DEsign = arr(t['DE-'])
    DEsign[DEsign != '-'] = '+'

    DE_dms = [str(desgn)+
              str(ded).zfill(2)+'d'+
              str(dem).zfill(2)+'m'+
              str(des).zfill(2)+'s'
              for desgn,ded,dem,des in zip(DEsign, DEd, DEm, DEs)]

    coords = SkyCoord(ra=RA_hms, dec=DE_dms, frame='icrs')

    RA = coords.ra.value
    dec = coords.dec.value
    pm_RA, pm_dec = arr(t['pmRA']), arr(t['pmDE'])
    u_pm_RA, u_pm_dec = arr(t['e_pmRA']), arr(t['e_pmDE'])

    maxsep = (maxsep*u.arcsec).to(u.deg).value

    name = t['Main'] if 'XI_' in namestr else t['Name']
    assoc = t['Assoc']

    outfile = os.path.join(outdir,'gotmatches_{}.xml.gz'.format(namestr))
    xmltouploadpath = os.path.join(outdir,'toupload_{}.xml'.format(namestr))

    if os.path.exists(outfile):
        os.remove(outfile) # NOTE if it's fast, can just do this to overwrite
    if not os.path.exists(outfile):
        _ = make_votable_given_full_cols(name, assoc, RA, dec, pm_RA, pm_dec,
                                    u_pm_RA, u_pm_dec,
                                    outpath=xmltouploadpath)

        Gaia.login(credentials_file=os.path.join(homedir, '.gaia_credentials'))

        # separated less than 10 arcsec.
        jobstr = (
        '''
        SELECT TOP {ncut:d} u.name, u.assoc, u.ra, u.dec, u.pm_ra, u.pm_dec,
        u.err_pm_ra, u.err_pm_dec,
        g.source_id, DISTANCE(
          POINT('ICRS', u.ra, u.dec),
          POINT('ICRS', g.ra,g.dec)) AS dist,
          g.phot_g_mean_mag as gaia_gmag,
          g.pmra AS gaia_pmra,
          g.pmdec AS gaia_pmdec
        FROM tap_upload.foobar as u, gaiadr2.gaia_source AS g
        WHERE 1=CONTAINS(
          POINT('ICRS', u.ra, u.dec),
          CIRCLE('ICRS', g.ra, g.dec, {sep:.8f})
        )
        '''
        )
        maxncut = int(5*len(name)) # to avoid query timeout
        query = jobstr.format(sep=maxsep, ncut=maxncut)

        if not os.path.exists(outfile):
            # might do async if this times out. but it doesn't.
            j = Gaia.launch_job(query=query,
                                upload_resource=xmltouploadpath,
                                upload_table_name="foobar", verbose=True,
                                dump_to_file=True, output_file=outfile)

        Gaia.logout()

    vot = parse(outfile)
    tab = vot.get_first_table().to_table()

    if maxncut - len(tab) < 10:
        errmsg = 'ERROR! too many matches'
        raise AssertionError(errmsg)

    # if nonzero and finite proper motion, require Gaia pm match to sign
    # of stated Gagne PMs.
    df = tab.to_pandas()

    print('\n'+42*'-')
    print('{} stars in original Gagne table'.format(len(t)))
    print('{} stars in sep < 10 arcsec xmatch'.format(len(df)))

    sel = (df['gaia_gmag'] < 18)
    print('{} stars in sep < 10 arcsec, G<18, xmatch'.format(len(df[sel])))

    sel &= (
        (   (df['pm_ra'] != 0 ) & (df['pm_dec'] != 0 ) &
            ( np.sign(df['pm_ra']) == np.sign(df['gaia_pmra']) ) &
            ( np.sign(df['pm_dec']) == np.sign(df['gaia_pmdec']) )
        )
        |
        (
            (df['pm_ra'] == 0 ) & (df['pm_dec'] == 0 )
        )
    )
    df = df[sel]
    print('{} stars in sep < 10 as xmatch, G<18, after pm cut (xor zero pm)'.
          format(len(df)))

    # make multiplicity column. then sort by name, then by distance. then drop
    # name duplicates, keeping the first (you have nearest neighbor saved!)
    _, inv, cnts = np.unique(df['name'], return_inverse=True,
                             return_counts=True)

    df['n_in_nbhd'] = cnts[inv]

    df['name'] = df['name'].str.decode('utf-8')
    df['assoc'] = df['assoc'].str.decode('utf-8')

    df = df.sort_values(['name','dist'])

    df = df.drop_duplicates(subset='name', keep='first')

    df['source_id'] = df['source_id'].astype('int64')

    print('{} stars after above cuts + chosing nearest nbhr by spatial sep'.
          format(len(df)))

    outpath = os.path.join(outdir,'MATCHED_{}.csv'.format(namestr))
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))
    print(79*'=')


def make_Rizzuto11_GaiaDR2_crossmatch(
    outdir=datadir, homedir='/home/luke/'):
    '''
    Aaron Rizzuto et al (2011) gave a list of 436 Sco OB2 members.
    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/MNRAS/416/3108
    '''
    vizier_search_str = 'J/MNRAS/416/3108'
    table_num = 0
    ra_str = '_RA'
    dec_str = '_DE'
    outname = os.path.join(datadir,'Rizzuto_11_table_1_ScoOB2_members.csv')
    assoc_name = 'ScoOB2'
    namestr = 'Rizzuto_11_table_1_ScoOB2_members'

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs(vizier_search_str)
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    tab = catalogs[table_num]
    print(42*'-')
    print('{}'.format(outname))
    print('initial number of members: {}'.format(len(tab)))

    # gaia xmatch need these two column names
    RA = tab[ra_str]
    dec = tab[dec_str]
    assoc = np.repeat(assoc_name, len(RA))

    # match "HIP" number to hipparcos table...
    outfile = os.path.join(outdir,'gotmatches_{}.xml.gz'.format(namestr))
    xmltouploadpath = os.path.join(outdir,'toupload_{}.xml'.format(namestr))

    if os.path.exists(outfile):
        os.remove(outfile)
    if not os.path.exists(outfile):

        votable = from_table(tab)
        writeto(votable, xmltouploadpath)
        print('made {}'.format(xmltouploadpath))

        Gaia.login(credentials_file=os.path.join(homedir, '.gaia_credentials'))

        # https://gea.esac.esa.int/archive/documentation/GDR2/...
        # Gaia_archive/chap_datamodel/sec_dm_auxiliary_tables/...
        # ssec_dm_dr1_neighbourhood.html
        jobstr = (
        '''
        SELECT TOP {ncut:d} u.hip, u.memb, u.col_ra, u.col_de,
        n.source_id, n.angular_distance as dist, n.number_of_neighbours
        FROM tap_upload.foobar as u, gaiadr2.hipparcos2_best_neighbour AS n
        WHERE u.hip = n.original_ext_source_id
        '''
        )
        maxncut = int(5*len(tab)) # to avoid query timeout
        query = jobstr.format(ncut=maxncut)

        if not os.path.exists(outfile):
            # might do async if this times out. but it doesn't.
            j = Gaia.launch_job(query=query,
                                upload_resource=xmltouploadpath,
                                upload_table_name="foobar", verbose=True,
                                dump_to_file=True, output_file=outfile)

        Gaia.logout()

    vot = parse(outfile)
    tab = vot.get_first_table().to_table()
    print('number after hipparcos xmatch: {}'.format(len(tab)))

    if maxncut - len(tab) < 10:
        errmsg = 'ERROR! too many matches'
        raise AssertionError(errmsg)

    df = tab.to_pandas()
    df['source_id'] = df['source_id'].astype('int64')

    df['name'] = df['hip'].astype('int64')

    outpath = os.path.join(outdir,'MATCHED_{}.csv'.format(namestr))
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))
    print(79*'=')


def make_Oh17_GaiaDR2_crossmatch(
    namestr='Oh_2017_clustering_GaiaDR2_crossmatched',
    outdir=datadir, homedir='/home/luke/'):
    '''
    Semyeong Oh et al (2017) discovered 10.6k stars with separations <10pc that
    are in likely comoving pairs / groups.

    see
    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/153/257
    '''

    # Download Oh's tables of stars, pairs, and groups.
    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/AJ/153/257')
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    stars = catalogs[0]
    pairs = catalogs[1]
    groups = catalogs[2]

    # IMPOSE GROUP SIZE >= 3.
    stars = stars[stars['Size'] >= 3]

    # use the gaia dr1 ids, to match the dr1_neighbourhood table.

    outfile = os.path.join(outdir,'gotmatches_{}.xml.gz'.format(namestr))
    xmltouploadpath = os.path.join(outdir,'toupload_{}.xml'.format(namestr))

    if os.path.exists(outfile):
        os.remove(outfile)
    if not os.path.exists(outfile):

        t = Table()
        t['name'] = arr(stars['Star']).astype(str)
        # note "Group" is a bad thing to name a column b/c it is a SQL
        # keyword...
        t['groupname'] = arr(stars['Group']).astype(str)
        t['ra'] = arr(stars['RAJ2000'])*u.deg
        t['dec'] = arr(stars['DEJ2000'])*u.deg
        t['gaia'] = arr(stars['Gaia']).astype(int)
        t['gmag'] = arr(stars['Gmag'])*u.mag
        t['size'] = arr(stars['Size']).astype(int)

        votable = from_table(t)
        writeto(votable, xmltouploadpath)
        print('made {}'.format(xmltouploadpath))

        Gaia.login(credentials_file=os.path.join(homedir, '.gaia_credentials'))

        # https://gea.esac.esa.int/archive/documentation/GDR2/...
        # Gaia_archive/chap_datamodel/sec_dm_auxiliary_tables/...
        # ssec_dm_dr1_neighbourhood.html
        jobstr = (
        '''
        SELECT TOP {ncut:d} u.name, u.gaia, u.ra, u.dec,
        u.groupname as assoc, u.size, u.gmag,
        n.dr2_source_id as source_id, n.angular_distance as dist, n.rank,
        n.magnitude_difference
        FROM tap_upload.foobar as u, gaiadr2.dr1_neighbourhood AS n
        WHERE u.gaia = n.dr1_source_id
        '''
        )
        maxncut = int(5*len(stars)) # to avoid query timeout
        query = jobstr.format(ncut=maxncut)

        if not os.path.exists(outfile):
            # might do async if this times out. but it doesn't.
            j = Gaia.launch_job(query=query,
                                upload_resource=xmltouploadpath,
                                upload_table_name="foobar", verbose=True,
                                dump_to_file=True, output_file=outfile)

        Gaia.logout()

    vot = parse(outfile)
    tab = vot.get_first_table().to_table()

    if maxncut - len(tab) < 10:
        errmsg = 'ERROR! too many matches'
        raise AssertionError(errmsg)

    # if nonzero and finite proper motion, require Gaia pm match to sign
    # of stated Gagne PMs.
    df = tab.to_pandas()

    print('\n'+42*'-')
    print('{} stars in original Oh table (size>=3)'.format(len(stars)))
    print('{} stars in gaia nbhd match'.format(len(df)))

    # we want DR2 mags. "gmag" is the uploaded DR1 mag.
    df['gaia_gmag'] = df['gmag'] + df['magnitude_difference']
    df.drop(['gmag'], axis=1, inplace=True)

    _, inv, cnts = np.unique(df['name'], return_inverse=True,
                             return_counts=True)
    df['n_in_nbhd'] = cnts[inv]

    df['name'] = df['name'].str.decode('utf-8')
    df['assoc'] = df['assoc'].str.decode('utf-8')

    df = df.sort_values(['name','dist'])

    df = df.drop_duplicates(subset='name', keep='first')

    df['gaia'] = df['gaia'].astype('int64')
    df = df.rename(index=str, columns={"gaia":"gaia_dr1_source_id"})

    df['size'] = df['size'].astype('int')

    df['source_id'] = df['source_id'].astype('int64')

    print('{} stars after above cuts + chosing nearest nbhr by spatial sep'.
          format(len(df)))

    outpath = os.path.join(outdir,'MATCHED_{}.csv'.format(namestr))
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))
    print(79*'=')


def make_Rizzuto15_GaiaDR2_crossmatch():
    '''
    Aaron Rizzuto et al (2015) picked out ~400 candidate USco members, then
    surveyed them for Li absorption. Led to 237 spectroscopically confirmed
    members.

    see
    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/MNRAS/448/2737
    '''

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/MNRAS/448/2737')
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    cands = catalogs[0]
    usco_pms = catalogs[1] # pre-MS stars in USco
    usco_disk = catalogs[2] # members of USco w/ circumstellar disk

    c = SkyCoord([ra.replace(' ',':')
                    for ra in list(map(str,usco_pms['RAJ2000']))],
                 [de.replace(' ',':')
                    for de in list(map(str,usco_pms['DEJ2000']))],
                 unit=(u.hourangle, u.deg))
    usco_pms['RA'] = c.ra.value
    usco_pms['DEC'] = c.dec.value
    usco_pms.remove_column('RAJ2000')
    usco_pms.remove_column('DEJ2000')
    foo = usco_pms.to_pandas()
    outname = '../data/cluster_data/moving_groups/Rizzuto_15_table_2_USco_PMS.csv'

    foo.to_csv(outname,index=False)
    print('saved {:s}'.format(outname))

    c = SkyCoord([ra.replace(' ',':')
                    for ra in list(map(str,usco_disk['RAJ2000']))],
                 [de.replace(' ',':')
                    for de in list(map(str,usco_disk['DEJ2000']))],
                 unit=(u.hourangle, u.deg))
    usco_disk['RA'] = c.ra.value
    usco_disk['DEC'] = c.dec.value
    usco_disk.remove_column('RAJ2000')
    usco_disk.remove_column('DEJ2000')
    foo = usco_disk.to_pandas()
    outname = os.path.join(datadir, 'Rizzuto_15_table_3_USco_hosts_disk.csv')
    foo.to_csv(outname,index=False)
    print('saved {:s}'.format(outname))

    print(
    ''' I then uploaded these lists to MAST, and used their spatial
        cross-matching with a 3 arcsecond cap, following
            https://archive.stsci.edu/tess/tutorials/upload_list.html

        This crossmatch is the output that I then saved to
            '../results/Rizzuto_15_table_2_USco_PMS_GaiaDR2_crossmatched_2arcsec_MAST.csv'
            '../results/Rizzuto_15_table_3_USco_disks_GaiaDR2_crossmatched_2arcsec_MAST.csv'
    '''
    )


def make_Preibisch01_GaiaDR2_crossmatch():
    '''
    Thomas Preibisch did a spectroscopic survey for low-mass members of USco.
    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/121/1040
    '''
    vizier_search_str = 'J/AJ/121/1040'
    table_num = 0
    ra_str = '_RA'
    dec_str = '_DE'
    outname = os.path.join(datadir,
                           'Preibisch_01_table_1_USco_LiRich_members.csv')

    make_vizier_GaiaDR2_crossmatch(vizier_search_str, ra_str, dec_str,
                               table_num=table_num, outname=outname)


def make_Casagrande_11_GaiaDR2_crossmatch():
    '''
    Casagrande et al (2011) re-analyzed the Geneva-Copenhagen survey, and got a
    kinematically unbiased sample of solar neighborhood stars with kinematics,
    metallicites, and ages.

    If Casagrande's reported max likelihood ages (from Padova isochrones) are
    below 1 Gyr, then that's interesting enough to look into.

    NOTE that this age cut introduces serious biases into the stellar mass
    distribution -- see Casagrande+2011, Figure 14.

    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/A+A/530/A138
    '''
    vizier_search_str = 'J/A+A/530/A138'
    table_num = 0
    ra_str = 'RAJ2000'
    dec_str = 'DEJ2000'
    outname = os.path.join(datadir,
                           'Casagrande_2011_table_1_GCS_ages_lt_1Gyr.csv')

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs(vizier_search_str)
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    tab = catalogs[0]

    # ageMLP: max-likelihood padova isochrone ages
    # ageMLB: max-likelihood BASTI isochrone ages. not queried.
    sel = (tab['ageMLP'] > 0)
    sel &= (tab['ageMLP'] < 1)

    coords = SkyCoord(ra=tab[ra_str], dec=tab[dec_str], frame='icrs',
                      unit=(u.hourangle, u.deg))
    # MAST uploads need these two column names
    tab['RA'] = coords.ra.value
    tab['DEC'] = coords.dec.value
    tab.remove_column('RAJ2000')
    tab.remove_column('DEJ2000')

    foo = tab[sel].to_pandas()
    foo.to_csv(outname,index=False)
    print('saved {:s}'.format(outname))

    print(
    ''' I then uploaded these lists to MAST, and used their spatial
        cross-matching with a 3 arcsecond cap, following
            https://archive.stsci.edu/tess/tutorials/upload_list.html

        This crossmatch is the output that I then saved to
            {:s}
    '''.format(outname.replace('data','results').replace('.csv','_GaiaDR2_3arcsec_crossmatch_MAST.csv'))
    )


def make_Kraus14_GaiaDR2_crossmatch():
    '''
    Adam Kraus et al (2014) did spectroscopy of members of the
    Tucana-Horologium moving group, looking at RVs, Halpha emission, and Li
    absoprtion.

    WARNING: only ~70% of the rows in this table turned out to be members.

    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/147/146
    '''
    vizier_search_str = 'J/AJ/147/146'
    table_num = 0
    ra_str = '_RA'
    dec_str = '_DE'
    pmra_str = 'pmRA'
    pmdec_str = 'pmDE'

    name_str = '_2MASS'
    assoc_name = 'Tuc-Hor'

    outname = os.path.join(datadir,
                           'Kraus_14_table_2_TucanaHorologiumMG_members.csv')

    make_vizier_GaiaDR2_crossmatch(vizier_search_str, ra_str, dec_str,
                                   pmra_str, pmdec_str, name_str, assoc_name,
                                   table_num=table_num, outname=outname)

def make_Roser11_GaiaDR2_crossmatch():
    '''
    Roser et al (2011) used PPMXL (positions, propoer motions, and photometry
    for 9e8 stars) to report Hyades members down to r<17.

    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/A+A/531/A92
    '''
    vizier_search_str = 'J/A+A/531/A92'
    table_num = 0
    ra_str = 'RAJ2000'
    dec_str = 'DEJ2000'
    pmra_str = 'pmRA'
    pmdec_str = 'pmDE'

    name_str = 'Seq'
    assoc_name = 'Hyades'

    outname = os.path.join(datadir, 'Roser11_table_1_Hyades_members.csv')

    make_vizier_GaiaDR2_crossmatch(vizier_search_str, ra_str, dec_str,
                                   pmra_str, pmdec_str, name_str, assoc_name,
                                   table_num=table_num, outname=outname)


def make_Luhman12_GaiaDR2_crossmatch():
    '''
    Luhman and Mamajek 2012 combined Spitzer & WISE photometry for all known
    USco members. Found IR excesses.
    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/ApJ/758/31
    '''
    vizier_search_str = 'J/ApJ/758/31'
    table_num = 0
    ra_str = '_RA'
    dec_str = '_DE'
    outname = os.path.join(datadir, 'Luhman_12_table_1_USco_IR_excess.csv')

    make_vizier_GaiaDR2_crossmatch(vizier_search_str, ra_str, dec_str,
                                   table_num=table_num, outname=outname)



def make_Bell17_GaiaDR2_crossmatch(maxsep=10,
                                   outdir=datadir,
                                   homedir='/home/luke/' ):

    with open(os.path.join(datadir,'Bell_2017_32Ori_table_3.txt')) as f:
        lines = f.readlines()

    lines = [l.replace('\n','') for l in lines if not l.startswith('#') and
             len(l) > 200]

    twomass_id_strs, pm_ra_strs, pm_dec_strs = [], [], []
    for l in lines:
        try:
            # regex finds the 2mass id
            twomass_id_strs.append(
                re.search('[0-9]{8}.[0-9]{7}', l).group(0)
            )
            ix = 0
            # regex finds floats in order, with the \pm appended. first is
            # always pm_RA, second is always pm_DEC.
            for m in re.finditer('[+-]?([0-9]*[.])?[0-9]+\\\\pm', l):
                if ix >= 2:
                    continue
                if ix == 0:
                    pm_ra_strs.append(float(m.group(0).rstrip('\\pm')))
                elif ix == 1:
                    pm_dec_strs.append(float(m.group(0).rstrip('\\pm')))
                ix += 1

        except:
            print('skipping')
            print(l)
            continue

    RA = [t[0:2]+'h'+t[2:4]+'m'+t[4:6]+'.'+t[6:8]
              for t in twomass_id_strs
         ]

    DE = [t[8]+t[9:11]+'d'+t[11:13]+'m'+t[13:15]+'.'+t[15]
              for t in twomass_id_strs
         ]

    c = SkyCoord(RA, DE, frame='icrs')

    RA = arr(c.ra.value)
    dec = arr(c.dec.value)
    pm_RA = arr(pm_ra_strs)
    pm_dec = arr(pm_dec_strs)
    name = arr(twomass_id_strs)
    assoc_name = '32Ori'
    assoc = np.repeat(assoc_name, len(RA))

    print(42*'-')
    outname = os.path.join(datadir, 'Bell17_table_32Ori.csv')
    print('{}'.format(outname))
    print('initial number of members: {}'.format(len(RA)))

    xmatchoutpath = outname.replace('.csv','_MATCHES_GaiaDR2.csv')
    outfile = outname.replace('.csv','_GOTMATCHES_GaiaDR2.xml')
    xmltouploadpath = outname.replace('.csv','_TOUPLOAD_GaiaDR2.xml')

    # do the spatial crossmatch...
    if os.path.exists(outfile):
        os.remove(outfile)
    if not os.path.exists(outfile):
        _ = make_votable_given_cols(name, assoc, RA, dec, pm_RA, pm_dec,
                                    outpath=xmltouploadpath)

        Gaia.login(credentials_file=os.path.join(homedir, '.gaia_credentials'))

        # separated less than 10 arcsec.
        jobstr = (
        '''
        SELECT TOP {ncut:d} u.name, u.assoc, u.ra, u.dec, u.pm_ra, u.pm_dec,
        g.source_id, DISTANCE(
          POINT('ICRS', u.ra, u.dec),
          POINT('ICRS', g.ra,g.dec)) AS dist,
          g.phot_g_mean_mag as gaia_gmag,
          g.pmra AS gaia_pmra,
          g.pmdec AS gaia_pmdec
        FROM tap_upload.foobar as u, gaiadr2.gaia_source AS g
        WHERE 1=CONTAINS(
          POINT('ICRS', u.ra, u.dec),
          CIRCLE('ICRS', g.ra, g.dec, {sep:.8f})
        )
        '''
        )
        maxncut = int(5*len(name)) # to avoid query timeout
        maxsep = (maxsep*u.arcsec).to(u.deg).value
        query = jobstr.format(sep=maxsep, ncut=maxncut)

        if not os.path.exists(outfile):
            # might do async if this times out. but it doesn't.
            j = Gaia.launch_job(query=query,
                                upload_resource=xmltouploadpath,
                                upload_table_name="foobar", verbose=True,
                                dump_to_file=True, output_file=outfile)

        Gaia.logout()

    vot = parse(outfile)
    tab = vot.get_first_table().to_table()

    if maxncut - len(tab) < 10:
        errmsg = 'ERROR! too many matches'
        raise AssertionError(errmsg)

    print('number of members after gaia 10 arcsec search: {}'.format(len(tab)))

    # if nonzero and finite proper motion, require Gaia pm match to sign
    # of stated PMs.
    df = tab.to_pandas()

    print('\n'+42*'-')

    sel = (df['gaia_gmag'] < 18)
    print('{} stars in sep < 10 arcsec, G<18, xmatch'.format(len(df[sel])))

    sel &= (
        (   (df['pm_ra'] != 0 ) & (df['pm_dec'] != 0 ) &
            ( np.sign(df['pm_ra']) == np.sign(df['gaia_pmra']) ) &
            ( np.sign(df['pm_dec']) == np.sign(df['gaia_pmdec']) )
        )
        |
        (
            (df['pm_ra'] == 0 ) & (df['pm_dec'] == 0 )
        )
    )
    df = df[sel]
    print('{} stars in sep < 10 as xmatch, G<18, after pm cut (xor zero pm)'.
          format(len(df)))

    # make multiplicity column. then sort by name, then by distance. then drop
    # name duplicates, keeping the first (you have nearest neighbor saved!)
    _, inv, cnts = np.unique(df['name'], return_inverse=True,
                             return_counts=True)

    df['n_in_nbhd'] = cnts[inv]

    df['name'] = df['name'].str.decode('utf-8')
    df['assoc'] = df['assoc'].str.decode('utf-8')

    df = df.sort_values(['name','dist'])

    df = df.drop_duplicates(subset='name', keep='first')

    df['source_id'] = df['source_id'].astype('int64')

    print('{} stars after above cuts + chosing nearest nbhr by spatial sep'.
          format(len(df)))

    df.to_csv(xmatchoutpath, index=False)
    print('made {}'.format(xmatchoutpath))
    print(79*'=')


def Tian2020_to_csv():

    tablepath = os.path.join(clusterdatadir, 'v05', 'apjabbf4bt1_mrt.txt')
    df = Table.read(tablepath, format='ascii.cds').to_pandas()

    outdf = pd.DataFrame({
        'source_id':list(df['Gaia'].astype(np.int64))
    })
    outdf['cluster'] = 'OrionSnake'
    outdf['age'] = np.round(np.log10(3.38e7),2)
    outdf = outdf[["source_id", "cluster", "age"]]

    outpath = os.path.join(
        clusterdatadir, 'v05', 'Tian2020_cut_cluster_source_age.csv'
    )
    outdf.to_csv(outpath, index=False)
    print(f'Made {outpath}')


def Kerr2021_to_csv():

    tablepath = os.path.join(
        clusterdatadir, 'v06', 'Kerr_2021_Table1_accepted.txt'
    )
    df = Table.read(tablepath, format='ascii.cds').to_pandas()

    outdf = pd.DataFrame({
        'source_id':list(df['Gaia'].astype(np.int64)),
        'cluster':list('TLC_'+df['TLC'].astype(str)),
        'age':np.round(np.log10(1e6*np.array(df['Age'].astype(float))), 2)
    })
    sel = (
        (outdf.cluster == 'TLC_-1')
        |
        (outdf.cluster == 'TLC_0')
    )
    outdf.loc[sel, 'cluster'] = 'N/A'

    outpath = os.path.join(
        clusterdatadir, 'v06', 'Kerr2021_cut_cluster_source_age.csv'
    )
    outdf.to_csv(outpath, index=False)
    print(f'Made {outpath}')



def Gagne2020_to_csv():

    tablepath = os.path.join(clusterdatadir, 'v05', 'apjabb77et12_mrt.txt')
    df = Table.read(tablepath, format='ascii.cds').to_pandas()

    sel = (
        (df['Memb-Type'] == 'IM')
        |
        (df['Memb-Type'] == 'CM')
        |
        (df['Memb-Type'] == 'LM')
    )

    sdf = df[sel]

    outdf = pd.DataFrame({
        'source_id':list(df['Gaia'].astype(np.int64))
    })
    outdf['cluster'] = 'muTau'
    outdf['age'] = np.round(np.log10(6.2e7),2)
    outdf = outdf[["source_id", "cluster", "age"]]

    outpath = os.path.join(
        clusterdatadir, 'v05', 'Gagne2020_cut_cluster_source_age.csv'
    )
    outdf.to_csv(outpath, index=False)
    print(f'Made {outpath}')


def Rizzuto2017_to_csv():

    # actual contents
    tablepath = os.path.join(clusterdatadir, 'v05', 'ajaa9070t2_mrt.txt')
    df1 = Table.read(tablepath, format='ascii.cds').to_pandas()
    df1.EPIC = df1.EPIC.astype(np.int64)

    # SIMBAD xmatch
    df2 = pd.read_csv(os.path.join(
        clusterdatadir, 'v05', 'SIMBAD_2017AJ....154..224R.csv')
    )

    df2['EPIC'] = df2['IDS'].str.extract("EPIC\ (\d*)")
    sdf2 = df2[~pd.isnull(df2.EPIC)]
    sdf2['EPIC'] = sdf2.EPIC.astype(np.int64)

    print(f'N initially in table: {len(df1)}')
    print(f'N in table after EPIC/GAIA XMATCH: {len(sdf2)}')

    mdf = sdf2.merge(df1[["ID","EPIC"]], how='left', on='EPIC')
    assert len(mdf) == len(sdf2)

    mdf['cluster'] = mdf.ID

    outdf = mdf[['source_id', 'cluster']]

    outpath = os.path.join(
        clusterdatadir, 'v05', 'Rizzuto2017_cut_cluster_source.csv'
    )
    outdf.to_csv(outpath, index=False)
    print(f'Made {outpath}')





def Pavlidou2021_to_csv():

    tablepaths = glob(os.path.join(clusterdatadir, 'v05', 'table_*new.tex'))

    dfs = []
    for tablepath in tablepaths:

        dfs.append(pd.read_csv(
            tablepath, sep=" & ",
            names='ix,source_id,ra,dec,pmra,pmdec,plx,Gmag,allwise'.split(',')
        ))


    df = pd.concat(dfs)

    outdf = pd.DataFrame({
        'source_id':list(df['source_id'].astype(np.int64))
    })
    outdf['cluster'] = 'Perseus'
    outdf['age'] = np.round(np.log10(5e6),2)
    outdf = outdf[["source_id", "cluster", "age"]]

    outpath = os.path.join(
        clusterdatadir, 'v05', 'Pavlidou2021_cut_cluster_source_age.csv'
    )
    outdf.to_csv(outpath, index=False)
    print(f'Made {outpath}')
