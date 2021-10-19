# -*- coding: utf-8 -*-
"""
a hack from https://github.com/lgbouma/extend_tess to make the bombest TESS
footprints (modulo reality)
"""

from __future__ import division, print_function

import numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tessmaps.get_time_on_silicon import (
        given_cameras_get_stars_on_silicon as gcgss
)
from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord
from numpy import array as nparr

import cdips as cd

import os, json

datadir = os.path.join(os.path.dirname(cd.__path__[0]),
                       'data/skymap_data')

def _shift_lon_get_x(lon, origin):
    x = np.array(np.remainder(lon+360-origin,360)) # shift lon values
    ind = x>180
    x[ind] -=360    # scale conversion to [-180, 180]
    x=-x    # reverse the scale: East to the left
    return x


def plot_mwd(lon, dec, color_val, origin=0, size=3,
             title='Mollweide projection', projection='mollweide', savdir='../results/',
             savname='mwd_0.pdf', overplot_galactic_plane=True, is_tess=False,
             coordsys=None, cbarbounds=None, for_proposal=False,
             overplot_k2_fields=False, overplot_cdips=False, plot_tess=True,
             primary_plus_extended=False):

    '''
    args, kwargs:

        lon, lat are arrays of same length. they can be (RA,dec), or (ecliptic
            long, ecliptic lat). lon takes values in [0,360), lat in [-90,90],

        coordsys (str): one of 'ecliptic', 'icrs', 'galactic'

        title is the title of the figure.

        projection is the kind of projection: 'mollweide', 'aitoff', ...

    comments: see
    http://balbuceosastropy.blogspot.com/2013/09/the-mollweide-projection.html.
    '''
    if coordsys == None:
        raise AssertionError

    # for matplotlib mollweide projection, x and y values (usually RA/dec, or
    # lat/lon) must be in -pi<x<pi, -pi/2<y<pi/2.
    # In astronomical coords, RA increases east (left on celestial charts).
    # Here, the default horizontal scale has positive to the right.

    x = _shift_lon_get_x(lon, origin)

    plt.close('all')
    fig = plt.figure(figsize=(6, 4.5)) # figszie doesn't do anything...
    ax = fig.add_subplot(111, projection=projection, facecolor='White')

    if is_tess:

        if for_proposal:

            # 14 color
            if len(cbarbounds) < 17:

                # orange to purple
                colors = ["#ffffff", "#e7d914", "#ceb128", "#b58a3d",
                          "#866c50", "#515263", "#1b3876", "#002680",
                          "#001d80", "#001480", "#000c80", "#000880",
                          "#000480", "#000080"]

                # blues
                colors = ["#ffffff", # N=0 white
                          "#84ccff", # N=1 pale blue
                          "#35aaff", # N=2 a little more saturated
                          "#279aea", # N=3-5 a little more saturated
                          "#279aea", # N=6-11 more saturated blue
                          "#279aea", "#1f77b4", "#1f77b4",
                          "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4",
                          "#126199", "#126199"] # N=12,13 saturated blue

                # grays
                colors = ["#ffffff", # N=0 white
                          "#dedede", # N=1 pale gray
                          "#b3b3b3", # N=2 a little more saturated
                          "#9c9c9c", # N=3-5 a little more saturated
                          "#9c9c9c", # N=6-11 more saturated blue
                          "#9c9c9c", "#757575", "#757575",
                          "#757575", "#757575", "#757575", "#757575",
                          "#363636", "#363636"] # N=12,13 saturated gray

                if primary_plus_extended:
                    colors += ["#1d1d1d"] # for extend colorbar

            cmap = LinearSegmentedColormap.from_list(
                'my_cmap', colors, N=len(colors))
        else:
            rgbs = sns.color_palette('Paired', n_colors=12, desat=0.9)
            cmap = mpl.colors.ListedColormap(rgbs)

        if isinstance(cbarbounds,np.ndarray):
            bounds=cbarbounds
        else:
            bounds = np.arange(-27.32/2, np.max(df['obs_duration'])+1/2*27.32, 27.32)

        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # plot zero-size for colorbar
        cax = ax.scatter(np.radians(x[::1000]),np.radians(dec[::1000]),
                         c=color_val[::1000],
                         s=0, lw=0, zorder=2, cmap=cmap, norm=norm,
                         rasterized=True)

        if plot_tess:
            max_cv = np.max(color_val)
            for ix, cv in enumerate(np.sort(np.unique(color_val))):
                if cv == 0:
                    continue
                sel = color_val == cv
                zorder = int(- max_cv - 1 + ix)
                _ = ax.scatter(np.radians(x[sel]),np.radians(dec[sel]),
                               c=color_val[sel], s=size,
                               lw=0, zorder=zorder, cmap=cmap, norm=norm,
                               marker='o',
                               rasterized=True)

            sel = color_val > 0
            _ = ax.scatter(np.radians(x[~sel]),np.radians(dec[~sel]),
                           c=color_val[~sel],
                           s=size/4, lw=0, zorder=-50, cmap=cmap, norm=norm,
                           marker='s',
                           rasterized=True)

        ticks = (np.arange(-1,13)+1)
        ylabels = list(map(str,np.round((np.arange(0,14)),1)))

        if primary_plus_extended:
            extend = 'max'
            ticks = (np.arange(-1,14)+1)
            ylabels = list(map(str,np.round((np.arange(0,13)),1)))
            ylabels += ['$\geq$13']
        else:
            extend = 'neither'

        cbar = fig.colorbar(cax, cmap=cmap, norm=norm, boundaries=bounds,
                            fraction=0.025, pad=0.03,
                            ticks=ticks,
                            orientation='vertical', extend=extend)

        # cbar.ax.set_yticklabels(ylabels, fontsize='x-small')
        cbar.set_label('Lunar months observed', rotation=270, labelpad=10)
        cbar.ax.tick_params(direction='in')

    else:
        ax.scatter(np.radians(x),np.radians(dec), c=color_val, s=size,
                   zorder=2)


    if overplot_galactic_plane:

        ##########
        # make many points, and also label the galactic center. ideally you
        # will never need to follow these coordinate transformations.
        glons = np.arange(0,360,0.2)
        glats = np.zeros_like(glons)
        coords = SkyCoord(glons*u.degree, glats*u.degree, frame='galactic')
        gplane_ra, gplane_dec = coords.icrs.ra.value, coords.icrs.dec.value
        gplane_elon = coords.barycentrictrueecliptic.lon.value
        gplane_elat = coords.barycentrictrueecliptic.lat.value
        gplane_glon = coords.galactic.l.value
        gplane_glat = coords.galactic.b.value

        if coordsys == 'ecliptic':
            gplane_x = _shift_lon_get_x(gplane_elon, origin)
            gplane_dec = gplane_elat
        elif coordsys == 'galactic':
            gplane_x = _shift_lon_get_x(gplane_glon, origin)
            gplane_dec = gplane_glat
        elif coordsys == 'icrs':
            gplane_x = _shift_lon_get_x(gplane_ra, origin)

        ax.scatter(np.radians(gplane_x),np.radians(gplane_dec),
                   c='lightgray', s=0.2, zorder=3, rasterized=True)
        gcenter = SkyCoord('17h45m40.04s', '-29d00m28.1s', frame='icrs')
        gcenter_ra, gcenter_dec = gcenter.icrs.ra.value, gcenter.icrs.dec.value
        gcenter_elon = gcenter.barycentrictrueecliptic.lon.value
        gcenter_elat = gcenter.barycentrictrueecliptic.lat.value
        gcenter_glon = gcenter.galactic.l.value
        gcenter_glat = gcenter.galactic.b.value

        if coordsys == 'ecliptic':
            gcenter_x = _shift_lon_get_x(np.array(gcenter_elon), origin)
            gcenter_dec = gcenter_elat
        elif coordsys == 'galactic':
            gcenter_x = _shift_lon_get_x(np.array(gcenter_glon), origin)
            gcenter_dec = gcenter_glat
        elif coordsys == 'icrs':
            gcenter_x = _shift_lon_get_x(np.array(gcenter_ra), origin)

        ax.scatter(np.radians(gcenter_x),np.radians(gcenter_dec),
                   c='black', s=2, zorder=4, marker='X')
        ax.text(np.radians(gcenter_x), np.radians(gcenter_dec), 'GC',
                fontsize='x-small', ha='left', va='top')
        ##########

    if overplot_cdips:

        from cdips.utils import collect_cdips_lightcurves as ccl
        df = ccl.get_cdips_pub_catalog(ver=0.5)

        if overplot_cdips == 'full':
            pass
        elif overplot_cdips == 'lt1kpc':
            sel = (df.parallax > 1) & (df.parallax/df.parallax_error > 3)
            df = df[sel]
        elif overplot_cdips == 'lt300pc':
            sel = (df.parallax > 3.3333333) & (df.parallax/df.parallax_error > 3)
            df = df[sel]

        c = SkyCoord(np.array(df['ra'])*u.deg, np.array(df['dec'])*u.deg,
                     frame='icrs')

        if coordsys == 'ecliptic':
            xval = _shift_lon_get_x(c.barycentrictrueecliptic.lon.value, origin)
            yval = c.barycentrictrueecliptic.lat.value
        elif coordsys == 'galactic':
            xval = _shift_lon_get_x(c.galactic.l.value, origin)
            yval = c.galactic.b.value
        elif coordsys == 'icrs':
            xval = _shift_lon_get_x(c.ra.value, origin)
            yval = c.dec.value

        _ = ax.scatter(np.radians(xval[::5]), np.radians(yval[::5]), c='C0',
                       s=0.3, lw=0, zorder=-1, marker='o', rasterized=True)


    if overplot_k2_fields:

        # do kepler
        kep = pd.read_csv(os.path.join(datadir,'kepler_field_footprint.csv'))
        # we want the corner points, not the mid-points
        is_mipoint = ((kep['row']==535) & (kep['column']==550))
        kep = kep[~is_mipoint]

        kep_coord = SkyCoord(np.array(kep['ra'])*u.deg,
                             np.array(kep['dec'])*u.deg, frame='icrs')
        kep_elon = kep_coord.barycentrictrueecliptic.lon.value
        kep_elat = kep_coord.barycentrictrueecliptic.lat.value
        kep['elon'] = kep_elon
        kep['elat'] = kep_elat

        kep_d = {}
        for module in np.unique(kep['module']):
            kep_d[module] = {}
            for output in np.unique(kep['output']):
                kep_d[module][output] = {}
                sel = (kep['module']==module) & (kep['output']==output)

                _ra = list(kep.loc[sel, 'ra'])
                _dec = list(kep.loc[sel, 'dec'])
                _elon = list(kep.loc[sel, 'elon'])
                _elat = list(kep.loc[sel, 'elat'])

                _ra = [_ra[0], _ra[1], _ra[3], _ra[2] ]
                _dec =  [_dec[0], _dec[1], _dec[3], _dec[2] ]
                _elon = [_elon[0], _elon[1], _elon[3], _elon[2] ]
                _elat = [_elat[0], _elat[1], _elat[3], _elat[2] ]

                _ra.append(_ra[0])
                _dec.append(_dec[0])
                _elon.append(_elon[0])
                _elat.append(_elat[0])

                kep_d[module][output]['corners_ra'] = _ra
                kep_d[module][output]['corners_dec'] = _dec
                kep_d[module][output]['corners_elon'] = _elon
                kep_d[module][output]['corners_elat'] = _elat

        # finally, make the plot!
        for mod in np.sort(list(kep_d.keys())):
            for op in np.sort(list(kep_d[mod].keys())):
                print(mod, op)

                this = kep_d[mod][op]

                ra = nparr(this['corners_ra'])
                dec = nparr(this['corners_dec'])
                elon = nparr(this['corners_elon'])
                elat = nparr(this['corners_elat'])

                if coordsys == 'ecliptic':
                    ch_x = _shift_lon_get_x(np.array(elon), origin)
                    ch_y = np.array(elat)
                elif coordsys == 'galactic':
                    raise NotImplementedError
                elif coordsys == 'icrs':
                    ch_x = _shift_lon_get_x(np.array(ra), origin)
                    ch_y = np.array(dec)

                # draw the outline of the fields -- same as fill.
                #ax.plot(np.radians(ch_x), np.radians(ch_y), c='gray',
                #        alpha=0.3, lw=0.15, rasterized=True)

                # fill in the fields
                ax.fill(np.radians(ch_x), np.radians(ch_y), c='lightgray',
                        alpha=0.95, lw=0, rasterized=True)

                # label them (NOTE: skipping)
                # txt_x, txt_y = np.mean(ch_x), np.mean(ch_y)
                # if mod == 13 and output == 2:
                #     ax.text(np.radians(txt_x), np.radians(txt_y),
                #             'Kepler', fontsize=4, va='center',
                #             ha='center', color='gray', zorder=10)

        # done kepler!
        # do k2
        footprint_dictionary = json.load(open(
            os.path.join(datadir,"k2-footprint.json")))

        for cn in np.sort(list(footprint_dictionary.keys())):
            if cn in ['c1','c10'] and coordsys=='icrs':
                continue
            if cn in ['c1','c10'] and coordsys=='ecliptic':
                continue
            print(cn)

            channel_ids = footprint_dictionary[cn]["channels"].keys()

            for channel_id in channel_ids:
                channel = footprint_dictionary[cn]["channels"][channel_id]
                ra = channel["corners_ra"] + channel["corners_ra"][:1]
                dec = channel["corners_dec"] + channel["corners_dec"][:1]

                if coordsys=='icrs':
                    ch_x = _shift_lon_get_x(np.array(ra), origin)
                    ch_y = np.array(dec)
                elif coordsys=='ecliptic':
                    ch_coord = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
                    ch_elon = ch_coord.barycentrictrueecliptic.lon.value
                    ch_elat = ch_coord.barycentrictrueecliptic.lat.value
                    ch_x = _shift_lon_get_x(np.array(ch_elon), origin)
                    ch_y = np.array(ch_elat)
                else:
                    raise NotImplementedError

                # draw the outline of the fields -- same as fill
                # ax.plot(np.radians(ch_x), np.radians(ch_y), c='gray',
                #         alpha=0.3, lw=0.15, rasterized=True)

                # fill in the fields
                ax.fill(np.radians(ch_x), np.radians(ch_y), c='lightgray',
                        alpha=.95, lw=0, rasterized=True)

                # label them (NOTE: skipping)
                # txt_x, txt_y = np.mean(ch_x), np.mean(ch_y)
                # if channel_id == '41':
                #     ax.text(np.radians(txt_x), np.radians(txt_y),
                #             cn.lstrip('c'), fontsize=4, va='center',
                #             ha='center', color='gray')

        # done k2!



    xticklabels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    xticklabels = np.remainder(xticklabels+360+origin,360)
    xticklabels = np.array([str(xtl)+'$\!$$^\circ$' for xtl in xticklabels])
    ax.set_xticklabels(xticklabels, fontsize='x-small', zorder=5)

    yticklabels = np.arange(-75,75+15,15)
    yticklabels = np.array([str(ytl)+'$\!$$^\circ$' for ytl in yticklabels])
    ax.set_yticklabels(yticklabels, fontsize='x-small')

    if not for_proposal:
        ax.set_title(title, y=1.05, fontsize='small')

    if coordsys == 'ecliptic':
        ax.set_xlabel('Ecliptic longitude', fontsize='small')
        ax.set_ylabel('Ecliptic latitude', fontsize='small')
    elif coordsys == 'galactic':
        ax.set_xlabel('Galactic longitude', fontsize='small')
        ax.set_ylabel('Galactic latitude', fontsize='small')
    elif coordsys == 'icrs':
        ax.set_xlabel('Right ascension', fontsize='small')
        ax.set_ylabel('Declination', fontsize='small')

    #ax.set_axisbelow(True)
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5, zorder=-3,
            alpha=0.15)

    if not for_proposal:
        ax.text(0.99,0.01,'github.com/lgbouma/extend_tess',
                fontsize='4',transform=ax.transAxes,
                ha='right',va='bottom')

    fig.tight_layout()
    if overplot_cdips:
        savname = savname.replace('OVERPLOT', f'{overplot_cdips}')
    fig.savefig(os.path.join(savdir,savname),dpi=350, bbox_inches='tight')
    print('saved {}'.format(os.path.join(savdir,savname)))



def get_n_observations(dirnfile, outpath, n_stars, merged=False,
                       withgaps=True,
                       aligncelestial=False):

    np.random.seed(42)

    # pick points uniformly on celestial sphere. they don't strictly need to be
    # random.  takes >~1 minute to draw random numbers after ~2*10^5. faster to
    # just do it on an appropriate grid.

    # e.g., http://mathworld.wolfram.com/SpherePointPicking.html
    # uniform0 = np.linspace(0,1,n_stars)
    # uniform1 = np.linspace(0,1,n_stars)
    rand0 = np.random.uniform(low=0,high=1,size=n_stars)
    rand1 = np.random.uniform(low=0,high=1,size=n_stars)

    theta = (2*np.pi*rand0 * u.rad).to(u.deg).value
    phi = (np.arccos(2*rand1 - 1) * u.rad).to(u.deg).value - 90

    ras = theta*u.deg
    decs = phi*u.deg

    coords = SkyCoord(ra=ras, dec=decs, frame='icrs')

    if merged:
        df_pri = pd.read_csv(os.path.join(
            datadir,'primary_mission_truenorth.csv', sep=';'))
        df_ext = pd.read_csv(dirnfile, sep=';')
        df = pd.concat([df_pri, df_ext])

    else:
        df = pd.read_csv(dirnfile, sep=';')

    lats = nparr([
        nparr(df['cam1_elat']),
        nparr(df['cam2_elat']),
        nparr(df['cam3_elat']),
        nparr(df['cam4_elat'])]).T
    lons = nparr([
        nparr(df['cam1_elon']),
        nparr(df['cam2_elon']),
        nparr(df['cam3_elon']),
        nparr(df['cam4_elon'])]).T

    cam_directions = []
    for lat, lon in zip(lats, lons):

        c1lat,c2lat,c3lat,c4lat = lat[0],lat[1],lat[2],lat[3]
        c1lon,c2lon,c3lon,c4lon = lon[0],lon[1],lon[2],lon[3]

        this_cam_dirn = [(c1lat, c1lon),
                         (c2lat, c2lon),
                         (c3lat, c3lon),
                         (c4lat, c4lon)]

        cam_directions.append(this_cam_dirn)

    df['camdirection'] = cam_directions

    n_observations = np.zeros_like(coords)

    for ix, row in df.iterrows():

        print(row['start'])
        cam_direction = row['camdirection']

        onchip = gcgss(coords, cam_direction, verbose=False, withgaps=withgaps,
                       aligncelestial=aligncelestial)

        n_observations += onchip

    outdf = pd.DataFrame({'ra':coords.ra.value,
                          'dec':coords.dec.value,
                          'elat':coords.barycentrictrueecliptic.lat.value,
                          'elon':coords.barycentrictrueecliptic.lon.value,
                          'n_observations': n_observations })
    outdf[['ra','dec','elon','elat','n_observations']].to_csv(
        outpath, index=False, sep=';')
    print('saved {}'.format(outpath))


def plot_tess_skymap(overplot_galactic_plane=True, for_proposal=False,
                     overplot_k2_fields=False, plot_tess=True,
                     overplot_cdips=False, primary_plus_extended=False):
    """
    make plots the primary mission.
    (no merging)
    """

    savdir = os.path.join(os.path.dirname(cd.__path__[0]),
                          'results/paper_V_figures')
    orbit_duration_days = 1/2 #27.32 / 2

    # things to change

    namestr = 'idea_15_final_truenorth' # could be e.g., 'primary_mission_truenorth'

    filenames = [ f'{namestr}.csv' ]

    eclsavnames = [ f'{namestr}_eclmap.png' ]

    icrssavnames = [ f'{namestr}_icrsmap.png' ]

    titles = [ '' ]

    dirnfiles = [ os.path.join(datadir,fname) for fname in filenames]

    for ix, dirnfile, eclsavname, icrssavname, title in zip(
        range(len(titles)), dirnfiles, eclsavnames, icrssavnames, titles):

        size=0.8

        if primary_plus_extended:
            eclsavname = eclsavname.replace('.png','_merged.png')
        if for_proposal:
            eclsavname = eclsavname.replace('.png','_forproposal.png')
            size=0.5
        if overplot_k2_fields:
            eclsavname = eclsavname.replace('.png','_k2overplot.png')
        if not plot_tess:
            eclsavname = eclsavname.replace('.png','_notess.png')
        if overplot_cdips:
            eclsavname = eclsavname.replace('.png',f'_cdipstargetsOVERPLOT.png')
        icrssavname = eclsavname.replace('eclmap','icrsmap')
        galsavname = eclsavname.replace('eclmap','galacticmap')

        obsdstr = ''
        if primary_plus_extended:
            obsdstr += '_merged'
        if for_proposal:
            obsdstr += '_forproposal'
        obsdpath = dirnfile.replace('.csv', f'_coords_observed{obsdstr}.csv')

        if not os.path.exists(obsdpath):
            # takes about 1 minute per strategy
            if for_proposal:
                npts = 12e5
            else:
                npts = 1e5

            get_n_observations(dirnfile, obsdpath, int(npts))

        df = pd.read_csv(obsdpath, sep=';')
        df['obs_duration'] = orbit_duration_days*df['n_observations']
        c = SkyCoord(np.array(df['ra'])*u.deg, np.array(df['dec'])*u.deg,
                     frame='icrs')
        df['glon'] = c.galactic.l.value
        df['glat'] = c.galactic.b.value

        cbarbounds = np.arange(-1/2, 14, 1)
        if primary_plus_extended:
            cbarbounds = np.arange(-1/2, 15, 1)
        sel_durn = (nparr(df['obs_duration']) >= 0)

        coordsyss = ['galactic','ecliptic','icrs']
        lons = ['glon','elon','ra']
        lats = ['glat','elat','dec']
        savnames = [galsavname, eclsavname, icrssavname]

        for c, lon, lat, s in zip(coordsyss, lons, lats, savnames):

            if c == 'galactic' and overplot_k2_fields:
                continue

            plot_mwd(nparr(df[lon])[sel_durn],
                     nparr(df[lat])[sel_durn],
                     nparr(df['obs_duration'])[sel_durn],
                     origin=0, size=size, title=title,
                     projection='mollweide', savdir=savdir,
                     savname=s,
                     overplot_galactic_plane=overplot_galactic_plane,
                     is_tess=True, coordsys=c, cbarbounds=cbarbounds,
                     for_proposal=for_proposal,
                     overplot_k2_fields=overplot_k2_fields,
                     overplot_cdips=overplot_cdips, plot_tess=plot_tess,
                     primary_plus_extended=primary_plus_extended)


if __name__=="__main__":

    # BEGIN OPTIONS

    for_proposal=1             # true to activate options only for the proposal
    overplot_k2_fields=0       # true to activate k2 field overplot
    plot_tess=1                # true to activate tess field overplot
    overplot_cdips=1           # true to overplot CDIPS target stars
    overplot_sfr_labels=0      # TODO true to overplot names of nearby star forming regions
    overplot_galactic_plane=0  # ...
    primary_plus_extended = 1  # whether to use "merged" coordinates

    # END OPTIONS

    plot_tess_skymap(overplot_galactic_plane=overplot_galactic_plane,
                     for_proposal=for_proposal,
                     overplot_k2_fields=1,
                     plot_tess=plot_tess, overplot_cdips=0,
                     primary_plus_extended=primary_plus_extended)

    for overplot_cdips in [0,'lt300pc','full','lt1kpc']:
        plot_tess_skymap(overplot_galactic_plane=overplot_galactic_plane,
                         for_proposal=for_proposal,
                         overplot_k2_fields=overplot_k2_fields,
                         plot_tess=plot_tess, overplot_cdips=overplot_cdips,
                         primary_plus_extended=primary_plus_extended)
