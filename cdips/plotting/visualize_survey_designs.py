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


def plot_mwd(lon,dec,color_val,origin=0,size=3,title='Mollweide projection',
             projection='mollweide',savdir='../results/',savname='mwd_0.pdf',
             overplot_galactic_plane=True, is_tess=False, is_radec=None,
             cbarbounds=None, for_proposal=False, overplot_k2_fields=False,
             for_GRR=False, plot_tess=True):

    '''
    args, kwargs:

        lon, lat are arrays of same length. they can be (RA,dec), or (ecliptic
            long, ecliptic lat). lon takes values in [0,360), lat in [-90,90],

        is_radec: mandatory. True if (RA,dec), else elong/elat.

        title is the title of the figure.

        projection is the kind of projection: 'mollweide', 'aitoff', ...

    comments: see
    http://balbuceosastropy.blogspot.com/2013/09/the-mollweide-projection.html.
    '''
    if is_radec == None:
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

        if for_proposal or for_GRR:
            # # 13-color (no "0" stars)
            # colors = ["#e7d914", "#ceb128", "#b58a3d", "#866c50", "#515263",
            #           "#1b3876", "#002680", "#001d80", "#001480", "#000c80",
            #           "#000880", "#000480", "#000080"]

            # 14 color
            if len(cbarbounds) < 15:

                colors = ["#ffffff", "#e7d914", "#ceb128", "#b58a3d",
                          "#866c50", "#515263", "#1b3876", "#002680",
                          "#001d80", "#001480", "#000c80", "#000880",
                          "#000480", "#000080"]

            # 28 color (kind of)
            elif len(cbarbounds) < 30:

                # first three orbits obsd get different colors. some difference
                # between 1yr and 2yr too.
                # take 1
                colors0 = ["#ffffff", "#e7d914", "#ceb128",
                          "#b58a3d", "#b58a3d", "#b58a3d", "#b58a3d",
                          "#866c50", "#866c50", "#866c50", "#866c50",
                          "#515263", "#515263", "#515263", "#515263",
                          "#1b3876", "#1b3876", "#1b3876", "#1b3876",
                          "#002680", "#002680", "#002680", "#002680",
                          "#000c80", "#000880", "#000880", "#000880",
                          "#000c80"]

                colors1 = ["#ffffff", # N=0 white
                          "#e7d914", # N=1 pale yellow
                          "#e7a013", # N=2 a little more saturated
                          "#f7781d", # N=3-5 a little more saturated
                          "#f7781d", # N=6-11 saturated orange
                          "#f7781d", "#e86000", "#e86000",
                          "#e86000", "#e86000", "#e86000", "#e86000",
                          "#e80000", "#e80000", # N=12,13 saturated red
                          "#12aee7", # N=14-20 different blue
                          "#12aee7","#12aee7","#12aee7",
                          "#12aee7","#12aee7","#12aee7",
                          "#126ae7", # N=21-26 saturated blue
                          "#126ae7", "#126ae7", "#126ae7",
                          "#126ae7", "#126ae7", "#126ae7"]

                colors = ["#ffffff", # N=0 white
                          "#84ccff", # N=1 pale blue
                          "#35aaff", # N=2 a little more saturated
                          "#279aea", # N=3-5 a little more saturated
                          "#279aea", # N=6-11 more saturated blue
                          "#279aea", "#1f77b4", "#1f77b4",
                          "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4",
                          "#126199", "#126199", # N=12,13 saturated blue
                          "#ffa251", # N=14-20 light orange
                          "#ffa251","#ffa251","#ffa251",
                          "#ffa251","#ffa251","#ffa251",
                          "#ff7f0e", # N=21-26 saturated orange
                          "#ff7f0e", "#ff7f0e", "#ff7f0e",
                          "#ff7f0e", "#ff7f0e", "#ff7f0e"]

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

        # plot the stars
        cax = ax.scatter(np.radians(x[::100]),np.radians(dec[::100]),
                         c=color_val[::100],
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

        # #FIXME WORKS!
        # _ = ax.scatter(np.radians(x[sel]),np.radians(dec[sel]),
        #                c=color_val[sel], s=size,
        #                lw=0, zorder=-1, cmap=cmap, norm=norm,
        #                marker='o',
        #                rasterized=True)

        # _ = ax.scatter(np.radians(x[~sel]),np.radians(dec[~sel]),
        #                c=color_val[~sel],
        #                s=size/4, lw=0, zorder=0, cmap=cmap, norm=norm,
        #                marker='s',
        #                rasterized=True)
        # #FIXME WORKS!

        # set up colorbar
        if len(colors) < 15:
            ticks = 27.32*(np.arange(-1,13)+1)
            ylabels = list(map(str,np.round(27.32*(np.arange(0,13)),1)))
            ylabels[-1] = '$\geq \! 328$'
        elif len(colors) < 30:
            #ticks = 27.32*(np.arange(-1,26)+1) #FIXME
            #ylabels = list(map(str,np.round(27.32*(np.arange(0,27)),1)))
            #ylabels[-1] = '$\geq \! 710$'

            ticks = (np.arange(-1,26)+1)
            ylabels = list(map(str,np.round((np.arange(0,27)),1)))


        cbar = fig.colorbar(cax, cmap=cmap, norm=norm, boundaries=bounds,
                            fraction=0.025, pad=0.03,
                            ticks=ticks,
                            orientation='vertical')

        cbar.ax.set_yticklabels(ylabels, fontsize='x-small')
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
        if is_radec:
            gplane_x = _shift_lon_get_x(gplane_ra, origin)
        else:
            gplane_x = _shift_lon_get_x(gplane_elon, origin)
            gplane_dec = gplane_elat
        ax.scatter(np.radians(gplane_x),np.radians(gplane_dec),
                   c='lightgray', s=0.2, zorder=3, rasterized=True)
        gcenter = SkyCoord('17h45m40.04s', '-29d00m28.1s', frame='icrs')
        gcenter_ra, gcenter_dec = gcenter.icrs.ra.value, gcenter.icrs.dec.value
        gcenter_elon = gcenter.barycentrictrueecliptic.lon.value
        gcenter_elat = gcenter.barycentrictrueecliptic.lat.value
        if is_radec:
            gcenter_x = _shift_lon_get_x(np.array(gcenter_ra), origin)
        else:
            gcenter_x = _shift_lon_get_x(np.array(gcenter_elon), origin)
            gcenter_dec = gcenter_elat
        ax.scatter(np.radians(gcenter_x),np.radians(gcenter_dec),
                   c='black', s=2, zorder=4, marker='X')
        ax.text(np.radians(gcenter_x), np.radians(gcenter_dec), 'GC',
                fontsize='x-small', ha='left', va='top')
        ##########

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

                _ra = list(kep.ix[sel]['ra'])
                _dec = list(kep.ix[sel]['dec'])
                _elon = list(kep.ix[sel]['elon'])
                _elat = list(kep.ix[sel]['elat'])

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

                if is_radec:
                    ch_x = _shift_lon_get_x(np.array(ra), origin)
                    ch_y = np.array(dec)
                else:
                    ch_x = _shift_lon_get_x(np.array(elon), origin)
                    ch_y = np.array(elat)

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
            if cn in ['c1','c10'] and is_radec:
                continue
            if cn in ['c1','c10'] and not is_radec:
                continue
            print(cn)

            channel_ids = footprint_dictionary[cn]["channels"].keys()

            for channel_id in channel_ids:
                channel = footprint_dictionary[cn]["channels"][channel_id]
                ra = channel["corners_ra"] + channel["corners_ra"][:1]
                dec = channel["corners_dec"] + channel["corners_dec"][:1]

                if is_radec:
                    ch_x = _shift_lon_get_x(np.array(ra), origin)
                    ch_y = np.array(dec)
                else:
                    ch_coord = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
                    ch_elon = ch_coord.barycentrictrueecliptic.lon.value
                    ch_elat = ch_coord.barycentrictrueecliptic.lat.value
                    ch_x = _shift_lon_get_x(np.array(ch_elon), origin)
                    ch_y = np.array(ch_elat)

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

    if is_radec:
        ax.set_xlabel('Right ascension', fontsize='x-small')
        ax.set_ylabel('Declination', fontsize='x-small')
    else:
        ax.set_xlabel('Ecliptic longitude', fontsize='x-small')
        ax.set_ylabel('Ecliptic latitude', fontsize='x-small')

    #ax.set_axisbelow(True)
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5, zorder=-3,
            alpha=0.15)

    if not for_proposal and not for_GRR:
        ax.text(0.99,0.01,'github.com/lgbouma/extend_tess',
                fontsize='4',transform=ax.transAxes,
                ha='right',va='bottom')

    fig.tight_layout()
    fig.savefig(os.path.join(savdir,savname),dpi=350, bbox_inches='tight')
    print('saved {}'.format(os.path.join(savdir,savname)))



def get_n_observations(dirnfile, outpath, n_stars, merged=False,
                       is_deming=False, withgaps=True,
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

    if is_deming:
        x, y = np.meshgrid(
            np.arange(-90,91,1),
            np.arange(0,361,1),
            indexing='ij'
        )
        # dumb
        theta = np.array(y.flatten())
        phi = np.array(x.flatten())
        withgaps=False

    ras = theta*u.deg
    decs = phi*u.deg

    coords = SkyCoord(ra=ras, dec=decs, frame='icrs')

    if merged:
        df_pri = pd.read_csv(os.path.join(
            datadir,'primary_mission.csv', sep=';'))
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


def only_extended_only_primary(is_deming=False, for_proposal=False,
                               overplot_k2_fields=False, for_GRR=False,
                               plot_tess=True):
    """
    make plots the primary mission.
    (no merging)
    """

    savdir = os.path.join(os.path.dirname(cd.__path__[0]),
                          'results/paper_figures')
    orbit_duration_days = 1/2 #27.32 / 2

    # things to change
    filenames = [ 'primary_mission.csv' ]

    eclsavnames = [ 'primary_mission_eclmap.png' ]

    icrssavnames = [ 'primary_mission_icrsmap.png' ]

    titles = [ 'primary mission' ]

    dirnfiles = [ os.path.join(datadir,fname) for fname in filenames]

    for ix, dirnfile, eclsavname, icrssavname, title in zip(
        range(len(titles)), dirnfiles, eclsavnames, icrssavnames, titles):

        size=0.8

        if for_proposal:
            eclsavname = eclsavname.replace('.png','_forproposal.png')
            icrssavname = icrssavname.replace('.png','_forproposal.png')
            size=0.5

        if overplot_k2_fields:
            eclsavname = eclsavname.replace('.png','_forproposal_k2overplot.png')
            icrssavname = icrssavname.replace('.png','_forproposal_k2overplot.png')

        if not plot_tess:
            eclsavname = eclsavname.replace('.png','_notess.png')
            icrssavname = icrssavname.replace('.png','_notess.png')

        obsdstr = '' if not for_proposal else '_forproposal'
        obsdpath = dirnfile.replace(
            '.csv', '_coords_observed{}.csv'.format(obsdstr))

        if not os.path.exists(obsdpath):
            # takes about 1 minute per strategy
            if for_proposal:
                npts = 12e5
            else:
                npts = 1e5

            if for_GRR:
                get_n_observations(dirnfile, obsdpath, int(npts),
                                   is_deming=is_deming, aligncelestial=True)
            else:
                get_n_observations(dirnfile, obsdpath, int(npts),
                                   is_deming=is_deming)

        df = pd.read_csv(obsdpath, sep=';')
        df['obs_duration'] = orbit_duration_days*df['n_observations']

        # post 20190131
        #cbarbounds = np.arange(-27.32/2, 2*13.5*27.32, 27.32)
        cbarbounds = np.arange(-1/2, 27, 1) #FIXME

        sel_durn = (nparr(df['obs_duration']) >= 0)

        plot_mwd(nparr(df['elon'])[sel_durn],
                 nparr(df['elat'])[sel_durn],
                 nparr(df['obs_duration'])[sel_durn],
                 origin=0, size=size, title=title,
                 projection='mollweide', savdir=savdir,
                 savname=eclsavname,
                 overplot_galactic_plane=True, is_tess=True, is_radec=False,
                 cbarbounds=cbarbounds,
                 for_proposal=for_proposal,
                 overplot_k2_fields=overplot_k2_fields,
                 for_GRR=for_GRR, plot_tess=plot_tess)

        plot_mwd(nparr(df['ra'])[sel_durn],
                 nparr(df['dec'])[sel_durn],
                 nparr(df['obs_duration'])[sel_durn],
                 origin=0, size=size, title=title,
                 projection='mollweide', savdir=savdir,
                 savname=icrssavname,
                 overplot_galactic_plane=True, is_tess=True, is_radec=True,
                 cbarbounds=cbarbounds,
                 for_proposal=for_proposal,
                 overplot_k2_fields=overplot_k2_fields,
                 for_GRR=for_GRR, plot_tess=plot_tess)


if __name__=="__main__":

    # BEGIN OPTIONS

    for_proposal=1          # true to activate options only for the proposal
    overplot_k2_fields=1    # true to activate k2 field overplot
    plot_tess=0             # true to activate tess field overplot

    # END OPTIONS

    only_extended_only_primary(for_proposal=for_proposal,
                               overplot_k2_fields=overplot_k2_fields,
                               plot_tess=plot_tess)
