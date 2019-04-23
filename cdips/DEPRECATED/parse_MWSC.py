'''
DESCRIPTION
----------
An assortment of code written for sanity checks on our 2017 TESS GI proposal
about difference imaging of clusters.

Most of this involving parsing Kharchenko et al (2013)'s table, hence the name
`parse_MWSC.py`.

The tools here do things like:

* Find how many open clusters we could observe

* Find how many member stars within those we could observe

* Compute TESS mags for everything (mostly via `ticgen`)

* Estimate blending effects, mainly through the dilution (computed just by
    summing magnitudes appropriately)

* Using K+13's King profile fits, estimate the surface density of member stars.
    It turns out that this radically underestimates the actual surface density
    of stars (because of all the background blends). Moreover, for purposes of
    motivating our difference imaging, "the number of stars in your aperture"
    is more relevant than "a surface density", and even more relevant than both
    of those is dilution.
    So I settled on the dilution calculation.

The plotting scripts here also make the skymap figure of the proposal. (Where
are the clusters on the sky?)


USAGE
----------

From /src/, select desired functions from __main__ below. Then:
>>> python parse_MWSC.py > output.log

'''

import matplotlib.pyplot as plt, seaborn as sns
import pandas as pd, numpy as np

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u

from math import pi
import pickle, os

from scipy.interpolate import interp1d

global COLORS
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# cite:
# 
# Jaffe, T. J. & Barclay, T. 2017, ticgen: A tool for calculating a TESS
# magnitude, and an expected noise level for stars to be observed by TESS.,
# v1.0.0, Zenodo, doi:10.5281/zenodo.888217
#
# and Stassun & friends (2017).
#import ticgen as ticgen


# # These two, from the website
# # http://dc.zah.uni-heidelberg.de/mwsc/q/clu/form
# # are actually outdated or something. They provided too few resuls..
# close_certain = pd.read_csv('../data/MWSC_search_lt_2000_pc_type_certain.csv')
# close_junk = pd.read_csv('../data/MWSC_search_lt_2000_pc_type_certain.csv')


def get_cluster_data():

    # Downloaded the MWSC from
    # http://cdsarc.u-strasbg.fr/viz-bin/Cat?cat=J%2FA%2BA%2F558%2FA53&target=http&
    tab = Table.read('../data/Kharchenko_2013_MWSC.vot', format='votable')

    df = tab.to_pandas()

    for colname in ['Type', 'Name', 'n_Type', 'SType']:
        df[colname] = [e.decode('utf-8') for e in list(df[colname])]

    # From erratum:
    # For the Sun-like star, a 4 Re planet produces a transit depth of 0.13%. The
    # limiting magnitude for transits to be detectable is about I_C = 11.4 . This
    # also corresponds to K_s ~= 10.6 and a maximum distance of 290 pc, assuming no
    # extinction.

    cinds = np.array(df['d']<500)
    close = df[cinds]
    finds = np.array(df['d']<1000)
    far = df[finds]

    N_c_r0 = int(np.sum(close['N1sr0']))
    N_c_r1 = int(np.sum(close['N1sr1']))
    N_c_r2 = int(np.sum(close['N1sr2']))
    N_f_r0 = int(np.sum(far['N1sr0']))
    N_f_r1 = int(np.sum(far['N1sr1']))
    N_f_r2 = int(np.sum(far['N1sr2']))

    type_d = {'a':'association', 'g':'globular cluster', 'm':'moving group',
              'n':'nebulosity/presence of nebulosity', 'r':'remnant cluster',
              's':'asterism', '': 'no label'}

    ntype_d = {'o':'object','c':'candidate','':'no label'}

    print('*'*50)
    print('\nMilky Way Star Clusters (close := <500pc)'
          '\nN_clusters: {:d}'.format(len(close))+\
          '\nN_stars (in core): {:d}'.format(N_c_r0)+\
          '\nN_stars (in central part): {:d}'.format(N_c_r1)+\
          '\nN_stars (in cluster): {:d}'.format(N_c_r2))


    print('\n'+'*'*50)
    print('\nMilky Way Star Clusters (far := <1000pc)'
          '\nN_clusters: {:d}'.format(len(far))+\
          '\nN_stars (in core): {:d}'.format(N_f_r0)+\
          '\nN_stars (in central part): {:d}'.format(N_f_r1)+\
          '\nN_stars (in cluster): {:d}'.format(N_f_r2))

    print('\n'+'*'*50)

    ####################
    # Post-processing. #
    ####################
    # Compute mean density
    mean_N_star_per_sqdeg = df['N1sr2'] / (pi * df['r2']**2)
    df['mean_N_star_per_sqdeg'] = mean_N_star_per_sqdeg

    # Compute King profiles
    king_profiles, theta_profiles = [], []
    for rt, rc, k, d in zip(np.array(df['rt']),
                            np.array(df['rc']),
                            np.array(df['k']),
                            np.array(df['d'])):

        sigma, theta = get_king_proj_density_profile(rt, rc, k, d)
        king_profiles.append(sigma)
        theta_profiles.append(theta)

    df['king_profile'] = king_profiles
    df['theta'] = theta_profiles

    ra = np.array(df['RAJ2000'])
    dec = np.array(df['DEJ2000'])

    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    galactic_long = np.array(c.galactic.l)
    galactic_lat = np.array(c.galactic.b)
    ecliptic_long = np.array(c.barycentrictrueecliptic.lon)
    ecliptic_lat = np.array(c.barycentrictrueecliptic.lat)

    df['galactic_long'] = galactic_long
    df['galactic_lat'] = galactic_lat
    df['ecliptic_long'] = ecliptic_long
    df['ecliptic_lat'] = ecliptic_lat

    cinds = np.array(df['d']<500)
    close = df[cinds]
    finds = np.array(df['d']<1000)
    far = df[finds]

    return close, far, df


def distance_histogram(df):

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    hist, bin_edges = np.histogram(
            df['d'],
            bins=np.append(np.logspace(1,6,1e3), 1e7),
            normed=False)
    ax.step(bin_edges[:-1], np.cumsum(hist), 'k-', where='post')

    ax.set_xlabel('distance [pc]')
    ax.set_ylabel('cumulative N clusters in MWSC')
    ax.set_xlim([5e1,1e4])
    ax.set_xscale('log')
    ax.set_yscale('log')
    f.tight_layout()
    f.savefig('d_cumdistribn_MWSC.pdf', dpi=300, bbox_inches='tight')


def angular_scale_cumdist(close, far):

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    axt = ax.twiny()

    scale_d = {'r0': 'angular radius of the core (0 if no core)',
               'r1': '"central" radius',
               'r2': 'cluster radius'}
    ix = 0

    for t, dat in [('$d<0.5$ kpc',close), ('$d<1$ kpc',far)]:

        for k in ['r2']:

            hist, bin_edges = np.histogram(
                    dat[k],
                    bins=np.append(np.logspace(-2,1,1e3), 1e7),
                    normed=False)

            ax.step(bin_edges[:-1], np.cumsum(hist),
                    where='post', label=t+' '+scale_d[k])

            ix += 1

    def tick_function(angle_deg):
        tess_px = 21*u.arcsec
        vals = angle_deg/tess_px.to(u.deg).value
        return ['%.1f' % z for z in vals]

    ax.legend(loc='upper left', fontsize='xx-small')
    ax.set_xlabel('ang scale [deg]')
    ax.set_ylabel('cumulative N clusters in MWSC')
    ax.set_xscale('log')
    #ax.set_yscale('log')

    axt.set_xscale('log')
    axt.set_xlim(ax.get_xlim())
    new_tick_locations = np.array([1e-2, 1e-1, 1e0, 1e1])
    axt.set_xticks(new_tick_locations)
    axt.set_xticklabels(tick_function(new_tick_locations))
    axt.set_xlabel('angular scale [TESS pixels]')


    f.tight_layout()
    f.savefig('angscale_cumdistribn_MWSC.pdf', dpi=300, bbox_inches='tight')


def angular_scale_hist(close, far):

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    axt = ax.twiny()

    scale_d = {'r0': 'angular radius of the core (0 if no core)',
               'r1': '"central" radius',
               'r2': 'cluster radius'}

    ix = 0

    for t, dat in [('$d<0.5$ kpc',close), ('$d<1$ kpc',far)]:

        for k in ['r2']:

            hist, bin_edges = np.histogram(
                    dat[k],
                    bins=np.append(np.logspace(-2,1,7), 1e7),
                    normed=False)

            ax.step(bin_edges[:-1], hist, where='post', label=t+' '+scale_d[k],
                    alpha=0.7)

            ix += 1

    def tick_function(angle_deg):
        tess_px = 21*u.arcsec
        vals = angle_deg/tess_px.to(u.deg).value
        return ['%.1f' % z for z in vals]

    ax.legend(loc='best', fontsize='xx-small')
    ax.set_xlabel('ang scale [deg]')
    ax.set_ylabel('N clusters in MWSC')
    ax.set_xscale('log')
    #ax.set_yscale('log')

    axt.set_xscale('log')
    axt.set_xlim(ax.get_xlim())
    new_tick_locations = np.array([1e-2, 1e-1, 1e0, 1e1])
    axt.set_xticks(new_tick_locations)
    axt.set_xticklabels(tick_function(new_tick_locations))
    axt.set_xlabel('angular scale [TESS pixels]')


    f.tight_layout()
    f.savefig('angscale_distribn_MWSC.pdf', dpi=300, bbox_inches='tight')


def mean_density_hist(close, far):

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    axt = ax.twiny()

    ix = 0

    for t, dat in [('$d<0.5$ kpc',close), ('$d<1$ kpc',far)]:

          hist, bin_edges = np.histogram(
                  dat['mean_N_star_per_sqdeg'],
                  bins=np.append(np.logspace(0,4,9), 1e7),
                  normed=False)

          ax.step(bin_edges[:-1], hist, where='post', label=t,
                  alpha=0.7)

          ix += 1

    def tick_function(N_star_per_sqdeg):
        tess_px = 21*u.arcsec
        tess_px_area = tess_px**2
        deg_per_tess_px = tess_px_area.to(u.deg**2).value
        vals = N_star_per_sqdeg * deg_per_tess_px
        outstrs = ['%.1E'%z for z in vals]
        outstrs = ['$'+o[0] + r'\! \cdot \! 10^{\mathrm{-}' + o[-1] + r'}$' \
                   for o in outstrs]
        return outstrs

    ax.legend(loc='best', fontsize='xx-small')
    ax.set_xlabel('mean areal density [stars/$\mathrm{deg}^{2}$]')
    ax.set_ylabel('N clusters in MWSC')
    ax.set_xscale('log')
    #ax.set_yscale('log')

    axt.set_xscale('log')
    axt.set_xlim(ax.get_xlim())
    new_tick_locations = np.logspace(0,4,5)
    axt.set_xticks(new_tick_locations)
    axt.set_xticklabels(tick_function(new_tick_locations))
    axt.set_xlabel('mean areal density [stars/$\mathrm{(TESS\ px)}^{2}$]')


    f.tight_layout()
    f.savefig('mean_density_distribn_MWSC.pdf', dpi=300, bbox_inches='tight')


def plot_king_profiles(close, far):

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.close('all')
    f, axs = plt.subplots(figsize=(4,7), nrows=2, ncols=1, sharex=True)

    for theta, profile in zip(close['theta'], close['king_profile']):
        axs[0].plot(theta, profile, alpha=0.2, c=colors[0])
    for theta, profile in zip(far['theta'], far['king_profile']):
        axs[1].plot(theta, profile, alpha=0.1, c=colors[1])


    # Add text in top right.
    axs[0].text(0.95, 0.95, '$d < 500\ \mathrm{pc}$', verticalalignment='top',
            horizontalalignment='right', transform=axs[0].transAxes,
            fontsize='large')
    axs[1].text(0.95, 0.95, '$d < 1\ \mathrm{kpc}$', verticalalignment='top',
            horizontalalignment='right', transform=axs[1].transAxes,
            fontsize='large')

    xmin, xmax = 1, 1e3

    for ax in axs:

        ax.set_xscale('log')
        ax.set_xlim([xmin, xmax])

        if ax == axs[1]:
            ax.xaxis.set_ticks_position('both')
            ax.set_xlabel('angular distance [TESS px]')

        ax.tick_params(which='both', direction='in', zorder=0)

        ax.set_ylabel(r'$\Sigma(r)$ [stars/$\mathrm{(TESS\ px)}^{2}$]')

    f.tight_layout(h_pad=0)
    f.savefig('king_density_profiles_close_MWSC.pdf', dpi=300,
              bbox_inches='tight')



def get_king_proj_density_profile(r_t, r_c, k, d):
    '''
    r_t: King's tidal radius [pc]
    r_c: King's core radius [pc]
    k: normalization [pc^{-2}]
    d: distance [pc]

    returns density profile in number per sq tess pixel
    '''

    # Eq 4 of Ernst et al, 2010 https://arxiv.org/pdf/1009.0710.pdf
    # citing King (1962).

    r = np.logspace(-2, 2.4, num=int(2e4))

    X = 1 + (r/r_c)**2
    C = 1 + (r_t/r_c)**2

    vals = k * (X**(-1/2) - C**(-1/2))**2

    #NOTE: this fails when r_t does not exist. This might be important...
    vals[r>r_t] = 0

    # vals currently in number per square parsec. want in number per TESS px.
    # first convert to number per square arcsec

    # N per sq arcsec. First term converts to 1/AU^2. Then the angular surface
    # density scales as the square of the distance (same number of things,
    # smaller angle)
    sigma = vals * 206265**(-2) * d**2

    tess_px = 21*u.arcsec

    arcsec_per_px = 21
    sigma_per_sq_px = sigma * arcsec_per_px**2 # N per px^2


    # r is in pc. we want the profile vs angular distance.
    AU_per_pc = 206265
    r *= AU_per_pc # r now in AU
    theta = r / d # angular distance in arcsec

    tess_px = 21 # arcsec per px
    theta *= (1/tess_px) # angular distance in px

    return sigma_per_sq_px, theta


def make_wget_script(df):
    '''
    to download stellar data for each cluster, need to run a script of wgets.
    this function makes the script.
    '''

    # get MWSC ids in "0012", "0007" format
    mwsc = np.array(df['MWSC'])
    mwsc_ids = np.array([str(int(f)).zfill(4) for f in mwsc])

    names = np.array(df['Name'])

    f = open('../data/MWSC_stellar_data/get_stellar_data.sh', 'w')

    outstrs = []
    for mwsc_id, name in zip(mwsc_ids, names):
        startstr = 'wget '+\
                 'ftp://cdsarc.u-strasbg.fr/pub/cats/J/A%2BA/558/A53/stars/2m_'
        middlestr = str(mwsc_id) + '_' + str(name)
        endstr = '.dat.bz2 ;\n'
        outstr = startstr + middlestr + endstr
        outstrs.append(outstr)

    f.writelines(outstrs)
    f.close()

    print('made wget script!')


def get_stellar_data_too(df, savstr, p_0=61):
    '''
    args:
      savstr (str): gets the string used to ID the output pickle

      p_0: probability for inclusion. See Eqs in Kharchenko+ 2012. p_0=61 (not
      sure why not 68.27) is 1 sigma members by kinematic and photometric
      membership probability, also accounting for spatial step function and
      proximity within stated cluster radius.

    call after `get_cluster_data`.

    This function reads the Kharchenko+ 2013 "stars/*" tables for each cluster,
    and selects the stars that are "most probably cluster members, that is,
    stars with kinematic and photometric membership probabilities >61%".

    (See Kharchenko+ 2012 for definitions of these probabilities)

    It then computes T mags for all of the members.

    For each cluster, it computes surface density vs angular distance from
    cluster center.

    %%%Method 1 (outdated):
    %%%Interpolating these results over the King profiles, it associates a surface
    %%%    density with each star.
    %%%(WARNING: how many clusters do not have King profiles?)

    Method 2 (used):
        Associate a surface density with each star by counting stars in annuli.
        This is also not very useful.

    It then returns "close", "far", and the entire dataframe
    '''

    names = np.array(df['Name'])
    r2s = np.array(df['r2']) # cluster radius (deg)
    # get MWSC ids in "0012", "0007" format
    mwsc = np.array(df['MWSC'])
    mwsc_ids = np.array([str(int(f)).zfill(4) for f in mwsc])

    readme = '../data/stellar_data_README'

    outd = {}

    # loop over clusters
    ix = 0
    for mwsc_id, name, r2 in list(zip(mwsc_ids, names, r2s)):

        print('\n'+50*'*')
        print('{:d}. {:s}: {:s}'.format(ix, str(mwsc_id), str(name)))
        outd[name] = {}

        middlestr = str(mwsc_id) + '_' + str(name)
        fpath = '../data/MWSC_stellar_data/2m_'+middlestr+'.dat'
        if name != 'Melotte_20':
            tab = ascii.read(fpath, readme=readme)
        else:
            continue

        # Select 1-sigma cluster members by photometry & kinematics.

        # From Kharchenko+ 2012, also require that:
        # * the 2MASS flag Qflg is "A" (i.e., signal-to-noise ratio
        #   S/N > 10) in each photometric band for stars fainter than
        #   Ks = 7.0;
        # * the mean errors of proper motions are smaller than 10 mas/yr
        #   for stars with δ ≥ −30deg , and smaller than 15 mas/yr for
        #   δ < −30deg.

        inds = (tab['Ps'] == 1)
        inds &= (tab['Pkin'] > p_0)
        inds &= (tab['PJKs'] > p_0)
        inds &= (tab['PJH'] > p_0)
        inds &= (tab['Rcl'] < r2)
        inds &= ( ((tab['Ksmag']>7) & (tab['Qflg']=='AAA')) | (tab['Ksmag']<7))
        pm_inds = ((tab['e_pm'] < 10) & (tab['DEdeg']>-30)) | \
                  ((tab['e_pm'] < 15) & (tab['DEdeg']<=-30))
        inds &= pm_inds

        members = tab[inds]
        mdf = members.to_pandas()

        # Compute T mag and 1-sigma, 1 hour integration noise using Mr Tommy
        # B's ticgen utility. NB relevant citations are listed at top.
        # NB I also modified his code to fix the needlessly complicated
        # np.savetxt formatting.
        mags = mdf[['Bmag', 'Vmag', 'Jmag', 'Hmag', 'Ksmag']]
        mags.to_csv('temp.csv', index=False)

        ticgen.ticgen_csv({'input_fn':'temp.csv'})

        temp = pd.read_csv('temp.csv-ticgen.csv')
        member_T_mags = np.array(temp['Tmag'])
        noise = np.array(temp['noise_1sig'])

        mdf['Tmag'] = member_T_mags
        mdf['noise_1hr'] = noise

        #########################################################################
        ## METHOD #1 to assign surface densities:
        ## The King profile for the cluster is already known. Assign each member
        ## star a surface density from the King profile evaluated at the member
        ## star's angular position.
        #king_profile = np.array(df.loc[df['Name']==name, 'king_profile'])[0]
        #king_theta = np.array(df.loc[df['Name']==name, 'theta'])[0]

        ## theta is saved in units of TESS px. Get each star's distance from the
        ## center in TESS pixels.
        #arcsec_per_tesspx = 21
        #Rcl = np.array(mdf['Rcl'])*u.deg
        #dists_from_center = np.array(Rcl.to(u.arcsec).value/arcsec_per_tesspx)

        ## interpolate over the King profile
        #func = interp1d(theta, king_profile, fill_value='extrapolate')

        #try:
        #    density_per_sq_px = func(dists_from_center)
        #except:
        #    print('SAVED OUTPUT TO ../data/Kharachenko_full.p')
        #    pickle.dump(outd, open('../data/Kharachenko_full.p', 'wb'))
        #    print('interpolation failed. check!')
        #    import IPython; IPython.embed()

        #mdf['density_per_sq_px'] = density_per_sq_px
        #########################################################################

        #########################################################################
        # METHOD #2 for surface densities (because Method #1 only counts
        # member stars!).
        # Just count stars in annuli.
        king_profile = np.array(df.loc[df['Name']==name, 'king_profile'])[0]
        king_theta = np.array(df.loc[df['Name']==name, 'theta'])[0]

        inds = (tab['Rcl'] < r2)
        stars_in_annulus = tab[inds]
        sia = stars_in_annulus.to_pandas()

        arcsec_per_tesspx = 21
        Rcl = np.array(sia['Rcl'])*u.deg
        dists_from_center = np.array(Rcl.to(u.arcsec).value/arcsec_per_tesspx)

        maxdist = ((r2*u.deg).to(u.arcsec).value/arcsec_per_tesspx)
        n_pts = np.min((50, int(len(sia)/2)))
        angsep_grid = np.linspace(0, maxdist, num=n_pts)

        # Attempt to compute Tmags for everything. Only count stars with
        # T<limiting magnitude as "contaminants" (anything else is probably too
        # faint to really matter!)
        mags = sia[['Bmag', 'Vmag', 'Jmag', 'Hmag', 'Ksmag']]
        mags.to_csv('temp.csv', index=False)
        ticgen.ticgen_csv({'input_fn':'temp.csv'})
        temp = pd.read_csv('temp.csv-ticgen.csv')
        T_mags = np.array(temp['Tmag'])

        all_dists = dists_from_center[(T_mags > 0) & (T_mags < 17) & \
                                      (np.isfinite(T_mags))]

        N_in_bin, edges = np.histogram(
                          all_dists,
                          bins=angsep_grid,
                          normed=False)

        # compute empirical surface density, defined on the midpoints
        outer, inner = angsep_grid[1:], angsep_grid[:-1]
        sigma = N_in_bin / (pi * (outer**2 - inner**2))
        midpoints = angsep_grid[:-1] + np.diff(angsep_grid)/2

        # interpolate over the empirical surface density as a function of
        # angular separation to assign surface densities to member stars.
        func = interp1d(midpoints, sigma, fill_value='extrapolate')

        member_Rcl = np.array(mdf['Rcl'])*u.deg
        member_dists_from_center = np.array(member_Rcl.to(u.arcsec).value/\
                                            arcsec_per_tesspx)

        try:
            member_density_per_sq_px = func(member_dists_from_center)
        except:
            print('SAVED OUTPUT TO ../data/Kharachenko_full_{:s}.p'.format(savstr))
            pickle.dump(outd, open(
                '../data/Kharachenko_full_{:s}.p'.format(savstr), 'wb'))
            print('interpolation failed. check!')
            import IPython; IPython.embed()

        mdf['density_per_sq_px'] = member_density_per_sq_px
        #########################################################################

        N_catalogd = int(df.loc[df['Name']==name, 'N1sr2'])
        N_my_onesigma = int(len(mdf))
        got_Tmag = (np.array(mdf['Tmag']) > 0)
        N_with_Tmag = len(mdf[got_Tmag])

        print('N catalogued as in cluster: {:d}'.format(N_catalogd))
        print('N I got as in cluster: {:d}'.format(N_my_onesigma))
        print('N of them with Tmag: {:d}'.format(N_with_Tmag))

        diff = abs(N_catalogd - N_with_Tmag)
        if diff > 5:
            print('\nWARNING: my cuts different from Kharachenko+ 2013!!')

        lens = np.array([len(member_T_mags),
                         len(noise),
                         len(member_dists_from_center),
                         len(member_density_per_sq_px)])
        np.testing.assert_equal(lens, lens[0]*np.ones_like(lens))

        # for members
        outd[name]['Tmag'] = np.array(mdf['Tmag'])
        outd[name]['noise_1hr'] = np.array(mdf['noise_1hr'])
        outd[name]['Rcl'] = member_dists_from_center
        outd[name]['density_per_sq_px'] = member_density_per_sq_px

        # Ocassionally, do some output plots to compare profiles
        if ix%50 == 0:

            plt.close('all')
            f, ax=plt.subplots()

            ax.scatter(member_dists_from_center, member_density_per_sq_px)
            ax.plot(king_theta, king_profile)

            ax.set_ylim([0,np.max((np.max(member_density_per_sq_px),
                                   np.max(king_profile) ) )])
            ax.set_xlim([0, 1.02*np.max(member_dists_from_center)])

            ax.set_xlabel('angular sep [TESS px]')
            ax.set_ylabel('surface density (line: King model, dots: empirical'
                          ' [per tess px area]', fontsize='xx-small')

            f.savefig('king_v_empirical/{:s}_{:d}.pdf'.format(name, ix),
                     bbox_inches='tight')

        del mdf
        ix += 1

    print(50*'*')
    print('SAVED OUTPUT TO ../data/Kharchenko_full_{:s}.p'.format(savstr))
    pickle.dump(outd, open(
        '../data/Kharchenko_full_{:s}.p'.format(savstr), 'wb'))
    print(50*'*')

    close = df[df['d'] < 500]
    far = df[df['d'] < 1000]
    return close, far, df


def get_dilutions_and_distances(df, savstr, faintest_Tmag=16, p_0=61):
    '''
    args:
      savstr (str): gets the string used to ID the output pickle

      p_0: probability for inclusion. See Eqs in Kharchenko+ 2012. p_0=61 (not
      sure why not 68.27) is 1 sigma members by kinematic and photometric
      membership probability, also accounting for spatial step function and
      proximity within stated cluster radius.

    call after `get_cluster_data`.

    This function reads the Kharchenko+ 2013 "stars/*" tables for each cluster,
    and selects the stars that are "most probably cluster members, that is,
    stars with kinematic and photometric membership probabilities >61%".

    (See Kharchenko+ 2012 for definitions of these probabilities)

    It then computes T mags for all of the members.

    For each cluster member, it then finds all cataloged stars (not necessarily
    cluster members) within 2, 3, 4, 5, 6 TESS pixels.

    It sums the fluxes, and computes a dilution.

    It saves (for each cluster member):
        * number of stars in various apertures
        * dilution for various apertures
        * distance of cluster member
        * Tmag of cluster member
        * noise_1hr for cluster member
        * ra,dec for cluster member
    '''

    names = np.array(df['Name'])
    r2s = np.array(df['r2'])
    # get MWSC ids in "0012", "0007" format
    mwsc = np.array(df['MWSC'])
    mwsc_ids = np.array([str(int(f)).zfill(4) for f in mwsc])

    readme = '../data/stellar_data_README'

    outd = {}

    # loop over clusters
    ix = 0
    start, step = 3, 7
    for mwsc_id, name, r2 in list(zip(mwsc_ids, names, r2s))[start::step]:

        print('\n'+50*'*')
        print('{:d}. {:s}: {:s}'.format(ix, str(mwsc_id), str(name)))
        outd[name] = {}

        outpath = '../data/MWSC_dilution_calc/{:s}.csv'.format(str(name))
        if os.path.exists(outpath):
            print('found {:s}, continue'.format(outpath))
            continue

        middlestr = str(mwsc_id) + '_' + str(name)
        fpath = '../data/MWSC_stellar_data/2m_'+middlestr+'.dat'
        if name not in ['Melotte_20', 'Sco_OB4']:
            tab = ascii.read(fpath, readme=readme)
        else:
            continue

        # Select 1-sigma cluster members by photometry & kinematics.

        # From Kharchenko+ 2012, also require that:
        # * the 2MASS flag Qflg is "A" (i.e., signal-to-noise ratio
        #   S/N > 10) in each photometric band for stars fainter than
        #   Ks = 7.0;
        # * the mean errors of proper motions are smaller than 10 mas/yr
        #   for stars with δ ≥ −30deg , and smaller than 15 mas/yr for
        #   δ < −30deg.

        inds = (tab['Ps'] == 1)
        inds &= (tab['Pkin'] > p_0)
        inds &= (tab['PJKs'] > p_0)
        inds &= (tab['PJH'] > p_0)
        inds &= (tab['Rcl'] < r2)
        inds &= ( ((tab['Ksmag']>7) & (tab['Qflg']=='AAA')) | (tab['Ksmag']<7))
        pm_inds = ((tab['e_pm'] < 10) & (tab['DEdeg']>-30)) | \
                  ((tab['e_pm'] < 15) & (tab['DEdeg']<=-30))
        inds &= pm_inds

        members = tab[inds]
        mdf = members.to_pandas()

        # Compute T mag and 1-sigma, 1 hour integration noise using Mr Tommy
        # B's ticgen utility. NB relevant citations are listed at top.
        # NB I also modified his code to fix the needlessly complicated
        # np.savetxt formatting.
        mags = mdf[['Bmag', 'Vmag', 'Jmag', 'Hmag', 'Ksmag']]
        mags.to_csv('temp{:s}.csv'.format(name), index=False)

        ticgen.ticgen_csv({'input_fn':'temp{:s}.csv'.format(name)})

        temp = pd.read_csv('temp{:s}.csv-ticgen.csv'.format(name))
        member_T_mags = np.array(temp['Tmag'])
        member_noise = np.array(temp['noise_1sig'])

        mdf['Tmag'] = member_T_mags
        mdf['noise_1hr'] = member_noise

        desired_Tmag_inds = ((member_T_mags > 0) & (member_T_mags < faintest_Tmag) & \
                                      (np.isfinite(member_T_mags)) )

        sel_members = mdf[desired_Tmag_inds]

        # Compute T mag for everything in this cluster field. NOTE this
        # consistently seems to fail for ~10% of the stars. This is not
        # precision science (we are getting coarse estimates), so ignore this
        # likely bug.
        mags = tab[['Bmag', 'Vmag', 'Jmag', 'Hmag', 'Ksmag']]
        mags.to_pandas().to_csv('temp{:s}.csv'.format(name), index=False)
        ticgen.ticgen_csv({'input_fn':'temp{:s}.csv'.format(name)})
        temp = pd.read_csv('temp{:s}.csv-ticgen.csv'.format(name))

        all_Tmag = np.array(temp['Tmag'])
        tab['Tmag'] = all_Tmag

        Tmag_inds = ((all_Tmag>0) & (all_Tmag<28) & (np.isfinite(all_Tmag)))

        sel_in_field = tab[Tmag_inds]

        # Want, for all cluster members with T<faintest_Tmag
        # * distance of cluster member
        # * Tmag of cluster member
        # * noise_1hr for cluster member
        # * ra,dec for cluster member
        # * number of stars in various apertures
        # * dilution for various apertures

        sel_members['dist'] = np.ones_like(np.array(sel_members['RAhour']))*\
                               float(df.loc[df['Name']==name, 'd'])

        Nstar_dict, dil_dict = {}, {}
        arcsec_per_px = 21

        for aper_radius in [2,3,4,5,6]:

            Nstar_str = 'Nstar_{:d}px'.format(aper_radius)
            dil_str = 'dil_{:d}px'.format(aper_radius)

            Nstar_dict[Nstar_str] = []
            dil_dict[dil_str] = []

        # Iterate over members, then over apertures.
        print('finding all neighbors and computing dilutions')
        for sm_ra, sm_dec, sm_Tmag in zip(sel_members['RAhour'],
                                          sel_members['DEdeg'],
                                          sel_members['Tmag']):

            member_c = SkyCoord(ra=sm_ra*u.hourangle, dec=sm_dec*u.degree)

            nbhr_RAs = np.array(sel_in_field['RAhour'])*u.hourangle
            nbhr_DECs = np.array(sel_in_field['DEdeg'])*u.degree

            c = SkyCoord(ra=nbhr_RAs, dec=nbhr_DECs)

            seps = c.separation(member_c)

            # Find neighboring stars in aperture.
            for aper_radius in [2,3,4,5,6]:

                Nstar_str = 'Nstar_{:d}px'.format(aper_radius)
                dil_str = 'dil_{:d}px'.format(aper_radius)

                aper_radius_in_as = aper_radius * arcsec_per_px * u.arcsecond

                in_aperture = (seps < aper_radius_in_as)

                stars_in_aperture = sel_in_field[in_aperture]
                Nstar_in_aperture = len(stars_in_aperture)

                # NB this list includes the target star.
                Tmags_in_aperture = np.array(stars_in_aperture['Tmag'])

                # Compute dilution.
                numerator = 10**(-0.4 * sm_Tmag)
                denominator = np.sum( 10**(-0.4 * Tmags_in_aperture) )
                dilution = numerator/denominator

                Nstar_dict[Nstar_str].append(Nstar_in_aperture)
                dil_dict[dil_str].append(dilution)

        for aper_radius in [2,3,4,5,6]:

            Nstar_str = 'Nstar_{:d}px'.format(aper_radius)
            dil_str = 'dil_{:d}px'.format(aper_radius)

            sel_members[Nstar_str] = Nstar_dict[Nstar_str]
            sel_members[dil_str] = dil_dict[dil_str]

        print('done computing dilutions')

        out = sel_members[
                  ['dist','Tmag','noise_1hr','RAhour','DEdeg',
                   'Nstar_2px','Nstar_3px','Nstar_4px','Nstar_5px','Nstar_6px',
                   'dil_2px','dil_3px','dil_4px','dil_5px','dil_6px'
                  ]
        ]

        #########################################################################

        N_catalogd = int(df.loc[df['Name']==name, 'N1sr2'])
        N_my_onesigma = len(mdf)
        N_with_Tmag = len(out)

        print('N catalogued as in cluster: {:d}'.format(N_catalogd))
        print('N I got as in cluster: {:d}'.format(N_my_onesigma))
        print('N of them with Tmag: {:d}'.format(N_with_Tmag))

        diff = abs(N_catalogd - N_with_Tmag)
        if diff > 5:
            print('\nWARNING: my cuts different from Kharachenko+ 2013!!')

        #########################################################################

        fpath = '../data/MWSC_dilution_calc/{:s}.csv'.format(str(name))
        print('saving to {:s}'.format(fpath))
        out.to_csv(fpath, index=False)

    print('done with dilution calculation')


def plot_King_density_vs_Tmag_scatter(close, far):

    c_names = np.sort(close['Name'])
    f_names = np.sort(far['Name'])

    obj = pickle.load(open('../data/Kharachenko_full.p','rb'))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Close clusters
    Tmags, densities = np.array([]), np.array([])
    for c_name in c_names:
        c = obj[c_name]
        #XXX FIXME THIS IS WRONG!!!!!!!!
        Tmags = np.concatenate((Tmags, c['Tmag']))
        densities = np.concatenate((densities, c['density_per_sq_px']))

    inds = (Tmags > 0) & (np.isfinite(densities)) & (densities < 1e10)
    inds &= (densities > 1e-20)

    df = pd.DataFrame({'Tmag':Tmags[inds],
                       'log10_density_per_sq_px':np.log10(densities[inds])})

    plt.close('all')
    g = sns.jointplot(x='Tmag', y='log10_density_per_sq_px',
                      data=df,
                      kind='hex',
                      color=colors[0],
                      size=4,
                      space=0,
                      stat_func=None,
                      xlim=[9,17],
                      ylim=[-6,0])
    g.set_axis_labels('TESS-band magnitude',
              '$\log_{10}$($\Sigma_{\mathrm{King}}\ [\mathrm{member\ stars/TESS\ px}^2]$)')

    g.savefig('king_density_vs_Tmag_scatter_close.pdf', dpi=300,
              bbox_inches='tight')

    # Far clusters
    Tmags, densities = np.array([]), np.array([])
    for f_name in f_names:
        c = obj[f_name]
        #XXX FIXME THIS IS WRONG
        Tmags = np.concatenate((Tmags, c['Tmag']))
        densities = np.concatenate((densities, c['density_per_sq_px']))

    inds = (Tmags > 0) & (np.isfinite(densities)) & (densities < 1e10)
    inds &= (densities > 1e-20)

    df = pd.DataFrame({'Tmag':Tmags[inds],
                       'log10_density_per_sq_px':np.log10(densities[inds])})

    plt.close('all')
    g = sns.jointplot(x='Tmag', y='log10_density_per_sq_px',
                      data=df,
                      kind='hex',
                      color=colors[1],
                      size=4,
                      space=0,
                      stat_func=None,
                      xlim=[9,17],
                      ylim=[-6,0])
    g.set_axis_labels('TESS-band magnitude',
              '$\log_{10}$($\Sigma_{\mathrm{King}}\ [\mathrm{member\ stars/TESS\ px}^2]$)')

    g.savefig('king_density_vs_Tmag_scatter_far.pdf', dpi=300,
              bbox_inches='tight')


def plot_empirical_density_vs_Tmag_scatter(close, far):

    c_names = np.sort(close['Name'])
    f_names = np.sort(far['Name'])

    obj = pickle.load(open('../data/Kharchenko_full_Tmag_lt_18.p','rb'))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Close clusters
    Tmags, densities = np.array([]), np.array([])
    for c_name in c_names:
        c = obj[c_name]
        Tmags = np.concatenate((Tmags, c['Tmag']))
        densities = np.concatenate((densities, c['density_per_sq_px']))

    inds = (Tmags > 0) & (np.isfinite(densities)) & (densities < 1e10)
    inds &= (densities > 1e-20)

    df = pd.DataFrame({'Tmag':Tmags[inds],
                       'log10_density_per_sq_px':np.log10(densities[inds])})

    plt.close('all')
    g = sns.jointplot(x='Tmag', y='log10_density_per_sq_px',
                      data=df,
                      kind='kde',
                      color=colors[0],
                      size=4,
                      space=0,
                      stat_func=None,
                      xlim=[9,17],
                      ylim=[-1.5,0.5])
    g.set_axis_labels('TESS-band magnitude',
              '$\log_{10}$($\Sigma_{\mathrm{empirical}}\ [\mathrm{obsd\ stars/TESS\ px}^2]$)')

    g.savefig('empirical_density_vs_Tmag_scatter_close.pdf', dpi=300,
              bbox_inches='tight')

    # Far clusters
    Tmags, densities = np.array([]), np.array([])
    for f_name in f_names:
        c = obj[f_name]
        #XXX FIXME THIS IS WRONG!!
        Tmags = np.concatenate((Tmags, c['Tmag']))
        densities = np.concatenate((densities, c['density_per_sq_px']))

    inds = (Tmags > 0) & (np.isfinite(densities)) & (densities < 1e10)
    inds &= (densities > 1e-20)

    df = pd.DataFrame({'Tmag':Tmags[inds],
                       'log10_density_per_sq_px':np.log10(densities[inds])})

    plt.close('all')
    g = sns.jointplot(x='Tmag', y='log10_density_per_sq_px',
                      data=df,
                      kind='kde',
                      color=colors[1],
                      size=4,
                      space=0,
                      stat_func=None,
                      xlim=[9,17],
                      ylim=[-1.5,0.5])
    g.set_axis_labels('TESS-band magnitude',
              '$\log_{10}$($\Sigma_{\mathrm{empirical}}\ [\mathrm{obsd\ stars/TESS\ px}^2]$)')

    g.savefig('empirical_density_vs_Tmag_scatter_far.pdf', dpi=300,
              bbox_inches='tight')



def plot_cluster_positions(close, far):
    '''
    Show the positions on Kavrayskiy VII, a global projection similar to
    Robinson, used widely in the former Soviet Union.
    '''

    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for coord in ['galactic','ecliptic']:

        plt.close('all')
        f, ax = plt.subplots(figsize=(4,4))
        m = Basemap(projection='kav7',lon_0=0, resolution='c', ax=ax)

        lats = np.array(close[coord+'_lat'])
        lons = np.array(close[coord+'_long'])
        x, y = m(lons, lats)
        m.scatter(x,y,3,marker='o',color=colors[0], label='$d<0.5$kpc',
                zorder=4)

        lats = np.array(far[coord+'_lat'])
        lons = np.array(far[coord+'_long'])
        x, y = m(lons, lats)
        m.scatter(x,y,3,marker='o',color=colors[1], label='$0.5<d<1$kpc',
                zorder=3)

        parallels = np.arange(-90.,120.,30.)
        meridians = np.arange(0.,420.,60.)
	# labels = [left,right,top,bottom]
        m.drawparallels(parallels, labels=[1,0,0,0], zorder=2,
                fontsize='small')
        ms = m.drawmeridians(meridians, labels=[0,0,0,1], zorder=2,
                fontsize='small')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
            box.width, box.height * 0.9])

	# Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.91, -0.07),
                fancybox=True, ncol=1, fontsize='x-small')

        for _m in ms:
            try:
                ms[_m][1][0].set_rotation(45)
            except:
                pass

        ax.set_xlabel(coord+' long', labelpad=25, fontsize='small')
        ax.set_ylabel(coord+' lat', labelpad=25, fontsize='small')

        #################### 
	# add TESS footprint
        dat = np.genfromtxt('../data/fig4_bundle/nhemi_shemi.csv', delimiter=',')
        dat = pd.DataFrame(np.transpose(dat), columns=['icSys', 'tSys', 'teff',
            'logg', 'r', 'm', 'eLat', 'eLon', 'micSys', 'mvSys', 'mic', 'mv',
            'stat', 'nPntg'])
        eLon, eLat = np.array(dat.eLon), np.array(dat.eLat)
        nPntg = np.array(dat.nPntg)
        if coord=='galactic':
            c = SkyCoord(lat=eLat*u.degree, lon=eLon*u.degree,
                    frame='barycentrictrueecliptic')
            lon = np.array(c.galactic.l)
            lat = np.array(c.galactic.b)

        elif coord=='ecliptic':
            lon, lat = eLon, eLat

        nPntg[nPntg >= 4] = 4

        ncolor = 4
        cmap1 = mpl.colors.ListedColormap(
                sns.color_palette("Greys", n_colors=ncolor, desat=1))
        bounds= list(np.arange(0.5,ncolor+1,1))
        norm1 = mpl.colors.BoundaryNorm(bounds, cmap1.N)

        x, y = m(lon, lat)
        out = m.scatter(x,y,s=0.2,marker='s',c=nPntg, zorder=1, cmap=cmap1,
                norm=norm1, rasterized=True, alpha=0.5)
        out = m.scatter(x,y,s=0, marker='s',c=nPntg, zorder=-1, cmap=cmap1,
                norm=norm1, rasterized=True, alpha=1)
        m.drawmapboundary()
        cbar = f.colorbar(out, cmap=cmap1, norm=norm1, boundaries=bounds,
            fraction=0.025, pad=0.05, ticks=np.arange(ncolor)+1,
            orientation='vertical')

        ylabels = np.arange(1,ncolor+1,1)
        cbarlabels = list(map(str, ylabels))[:-1]
        cbarlabels.append('$\geq 4$')
        cbar.ax.set_yticklabels(cbarlabels)
        cbar.set_label('N pointings', rotation=270, labelpad=5)
        #################### 

        f.savefig('cluster_positions_'+coord+'.pdf', bbox_inches='tight')



def plot_cluster_positions_scicase(df):
    '''
    Show the positions of d<2kpc clusters, and highlight those with rotation
    period measurements & transiting planets.
    '''

    rotn_clusters = ['NGC_1976', # AKA the orion nebula cluster
                     'NGC_6530',
                     'NGC_2264',
                     'Cep_OB3',
                     'NGC_2362',
                     'NGC_869', # h Per, one of the double cluster
                     'NGC_2547',
                     'IC_2391',
                     'Melotte_20', # alpha Persei cluster, alpha Per
                     'Melotte_22', # AKA Pleiades
                     'NGC_2323', # M 50
                     'NGC_2168', #M 35
                     'NGC_2516',
                     'NGC_1039', #M 34
                     'NGC_2099', # M 37
                     #'NGC_2632', #Praesepe, comment out to avoid overlap
                     #'NGC_6811', #comment out to avoid overlap
                     'NGC_2682' ] #M 67

    transiting_planet_clusters = [
                     'NGC_6811',
                     'NGC_2632' #Praesepe
                    ]

    df = df[df['d'] < 2000]

    df_rotn = df.loc[df['Name'].isin(rotn_clusters)]
    df_rotn = df_rotn[
            ['ecliptic_lat','ecliptic_long','galactic_lat','galactic_long',
            'Name']
            ]

    df_tra = df.loc[df['Name'].isin(transiting_planet_clusters)]

    # Above rotation lists were from Table 1 of Gallet & Bouvier 2015,
    # including M67 which was observed by K2.  Transiting planets from the few
    # papers that have them.  They are cross-matching MWSC's naming scheme. I
    # could not find the Hyades or ScoCen OB.  They both have transiting
    # planets, and the former has rotation studies done.

    c_Hyades = SkyCoord(ra='4h27m', dec=15*u.degree + 52*u.arcminute)
    df_hyades = pd.DataFrame({
            'Name':'Hyades',
            'ecliptic_long':float(c_Hyades.barycentrictrueecliptic.lon.value),
            'ecliptic_lat':float(c_Hyades.barycentrictrueecliptic.lat.value),
            'galactic_long':float(c_Hyades.galactic.l.value),
            'galactic_lat':float(c_Hyades.galactic.b.value)}, index=[0])

    c_ScoOB2 = SkyCoord(ra='16h10m14.73s', dec='-19d19m09.38s') # Mann+2016's position
    df_ScoOB2 = pd.DataFrame({
            'Name':'Sco_OB2',
            'ecliptic_long':float(c_ScoOB2.barycentrictrueecliptic.lon.value),
            'ecliptic_lat':float(c_ScoOB2.barycentrictrueecliptic.lat.value),
            'galactic_long':float(c_ScoOB2.galactic.l.value),
            'galactic_lat':float(c_ScoOB2.galactic.b.value)}, index=[0])

    df_tra = df_tra.append(df_hyades, ignore_index=True)
    df_tra = df_tra.append(df_ScoOB2, ignore_index=True)
    #df_rotn = df_rotn.append(df_hyades, ignore_index=True) #avoid overlap

    # End of data wrangling.


    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap

    for coord in ['galactic','ecliptic']:

        plt.close('all')
        #f, ax = plt.subplots(figsize=(4,4))
        f = plt.figure(figsize=(0.7*5,0.7*4))
        ax = plt.gca()
        m = Basemap(projection='kav7',lon_0=0, resolution='c', ax=ax)

        lats = np.array(df[coord+'_lat'])
        lons = np.array(df[coord+'_long'])
        x, y = m(lons, lats)
        m.scatter(x,y,2,marker='o',facecolor=COLORS[0], zorder=4,
                alpha=0.9,edgecolors=COLORS[0], lw=0)

        lats = np.array(df_rotn[coord+'_lat'])
        lons = np.array(df_rotn[coord+'_long'])
        x, y = m(lons, lats)
        m.scatter(x,y,42,marker='*',color=COLORS[1],edgecolors='k',
                label='have rotation studies', zorder=5,lw=0.4)

        lats = np.array(df_tra[coord+'_lat'])
        lons = np.array(df_tra[coord+'_long'])
        x, y = m(lons, lats)
        m.scatter(x,y,13,marker='s',color=COLORS[1],edgecolors='k',
                label='also have transiting planets', zorder=6, lw=0.45)

        parallels = np.arange(-90.,120.,30.)
        meridians = np.arange(0.,420.,60.)
	# labels = [left,right,top,bottom]
        ps = m.drawparallels(parallels, labels=[1,0,0,0], zorder=2,
                fontsize='x-small')
        ms = m.drawmeridians(meridians, labels=[0,0,0,1], zorder=2,
                fontsize='x-small')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
            box.width, box.height * 0.9])

	# Put a legend below current axis
        #ax.legend(loc='upper center', bbox_to_anchor=(0.01, 0.02),
        #        fancybox=True, ncol=1, fontsize='x-small')

        for _m in ms:
            try:
                #ms[_m][1][0].set_rotation(45)
                if '60' in ms[_m][1][0].get_text():
                    ms[_m][1][0].set_text('')
            except:
                pass
        for _p in ps:
            try:
                if '30' in ps[_p][1][0].get_text():
                    ps[_p][1][0].set_text('')
            except:
                pass

        ax.set_xlabel(coord+' long', labelpad=13, fontsize='x-small')
        ax.set_ylabel(coord+' lat', labelpad=13, fontsize='x-small')

        ######################
	# add TESS footprint #
        ######################
        dat = np.genfromtxt('../data/fig4_bundle/nhemi_shemi.csv', delimiter=',')
        dat = pd.DataFrame(np.transpose(dat), columns=['icSys', 'tSys', 'teff',
            'logg', 'r', 'm', 'eLat', 'eLon', 'micSys', 'mvSys', 'mic', 'mv',
            'stat', 'nPntg'])
        eLon, eLat = np.array(dat.eLon), np.array(dat.eLat)
        nPntg = np.array(dat.nPntg)
        if coord=='galactic':
            c = SkyCoord(lat=eLat*u.degree, lon=eLon*u.degree,
                    frame='barycentrictrueecliptic')
            lon = np.array(c.galactic.l)
            lat = np.array(c.galactic.b)

        elif coord=='ecliptic':
            lon, lat = eLon, eLat

        nPntg[nPntg >= 4] = 4

        ncolor = 4
        cmap1 = mpl.colors.ListedColormap(
                sns.color_palette("Greys", n_colors=ncolor, desat=1))
        bounds= list(np.arange(0.5,ncolor+1,1))
        norm1 = mpl.colors.BoundaryNorm(bounds, cmap1.N)

        x, y = m(lon, lat)
        out = m.scatter(x,y,s=0.2,marker='s',c=nPntg, zorder=1, cmap=cmap1,
                norm=norm1, rasterized=True, alpha=0.5)
        out = m.scatter(x,y,s=0, marker='s',c=nPntg, zorder=-1, cmap=cmap1,
                norm=norm1, rasterized=True, alpha=1)
        m.drawmapboundary()
        #cbar = f.colorbar(out, cmap=cmap1, norm=norm1, boundaries=bounds,
        #    fraction=0.025, pad=0.05, ticks=np.arange(ncolor)+1,
        #    orientation='vertical')

        #ylabels = np.arange(1,ncolor+1,1)
        #cbarlabels = list(map(str, ylabels))[:-1]
        #cbarlabels.append('$\geq\! 4$')
        #cbar.ax.set_yticklabels(cbarlabels, fontsize='x-small')
        #cbar.set_label('N pointings', rotation=270, labelpad=5, fontsize='x-small')
        #################### 
        f.tight_layout()

        f.savefig('cluster_positions_'+coord+'_scicase.pdf', bbox_inches='tight')



def plot_HATS_field_positions():
    '''
    Show the positions on Kavrayskiy VII, a global projection similar to
    Robinson, used widely in the former Soviet Union.

    N.B. we're just markering the HATS field center (13x13 deg each)
    '''

    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    df = pd.read_csv('../data/HATPI_field_ids.txt', delimiter='|')
    ra = df['ra']
    dec = df['decl']
    fieldnums = df['field_num']
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    lons = np.array(c.barycentrictrueecliptic.lon)
    lats = np.array(c.barycentrictrueecliptic.lat)

    for coord in ['ecliptic']:

        plt.close('all')
        f, ax = plt.subplots(figsize=(4,4))
        m = Basemap(projection='kav7',lon_0=0, resolution='c', ax=ax)

        x, y = m(lons, lats)
        m.scatter(x,y,13,marker='s',color=colors[0], label='HATPI fields',
                zorder=4)
        for s, _x, _y in list(zip(fieldnums, x,y)):
            ax.text(x=_x, y=_y, s=s, fontsize='xx-small',
            verticalalignment='center', horizontalalignment='center', zorder=6)

        parallels = np.arange(-90.,120.,30.)
        meridians = np.arange(0.,420.,60.)
	# labels = [left,right,top,bottom]
        m.drawparallels(parallels, labels=[1,0,0,0], zorder=2,
                fontsize='small')
        ms = m.drawmeridians(meridians, labels=[0,0,0,1], zorder=2,
                fontsize='small')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
            box.width, box.height * 0.9])

	# Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.91, -0.07),
                fancybox=True, ncol=1, fontsize='x-small')

        for _m in ms:
            try:
                ms[_m][1][0].set_rotation(45)
            except:
                pass

        ax.set_xlabel(coord+' long', labelpad=25, fontsize='small')
        ax.set_ylabel(coord+' lat', labelpad=25, fontsize='small')

        #################### 
	# add TESS footprint
        dat = np.genfromtxt('../data/fig4_bundle/nhemi_shemi.csv', delimiter=',')
        dat = pd.DataFrame(np.transpose(dat), columns=['icSys', 'tSys', 'teff',
            'logg', 'r', 'm', 'eLat', 'eLon', 'micSys', 'mvSys', 'mic', 'mv',
            'stat', 'nPntg'])
        eLon, eLat = np.array(dat.eLon), np.array(dat.eLat)
        nPntg = np.array(dat.nPntg)
        if coord=='galactic':
            c = SkyCoord(lat=eLat*u.degree, lon=eLon*u.degree,
                    frame='barycentrictrueecliptic')
            lon = np.array(c.galactic.l)
            lat = np.array(c.galactic.b)

        elif coord=='ecliptic':
            lon, lat = eLon, eLat

        nPntg[nPntg >= 4] = 4

        ncolor = 4
        cmap1 = mpl.colors.ListedColormap(
                sns.color_palette("Greys", n_colors=ncolor, desat=1))
        bounds= list(np.arange(0.5,ncolor+1,1))
        norm1 = mpl.colors.BoundaryNorm(bounds, cmap1.N)

        x, y = m(lon, lat)
        out = m.scatter(x,y,s=0.2,marker='s',c=nPntg, zorder=1, cmap=cmap1,
                norm=norm1, rasterized=True, alpha=0.5)
        out = m.scatter(x,y,s=0, marker='s',c=nPntg, zorder=-1, cmap=cmap1,
                norm=norm1, rasterized=True, alpha=1)
        m.drawmapboundary()
        cbar = f.colorbar(out, cmap=cmap1, norm=norm1, boundaries=bounds,
            fraction=0.025, pad=0.05, ticks=np.arange(ncolor)+1,
            orientation='vertical')

        ylabels = np.arange(1,ncolor+1,1)
        cbarlabels = list(map(str, ylabels))[:-1]
        cbarlabels.append('$\geq 4$')
        cbar.ax.set_yticklabels(cbarlabels)
        cbar.set_label('N pointings', rotation=270, labelpad=5)
        #################### 

        f.savefig('HATPI_field_positions_'+coord+'.pdf', bbox_inches='tight')


def plot_dilution_vs_dist_and_Tmag():
    '''
    2d distribution plots:

        dil_2px vs dist
        dil_3px vs dist
        dil_4px vs dist
        dil_2px vs Tmag
        dil_3px vs Tmag
        dil_4px vs Tmag
    '''
    # Collect all dilutions, distances, Tmags
    data_dir = '../data/MWSC_dilution_calc/'
    csv_paths = [data_dir+f for f in os.listdir(data_dir)]

    df = pd.concat((pd.read_csv(f) for f in csv_paths), ignore_index=True)

    df['log10_dist'] = np.log10(df['dist'])

    # vs dist plots
    for ydim in ['dil_2px', 'dil_3px', 'dil_4px']:
        plt.close('all')
        g = sns.jointplot(x='log10_dist', y=ydim,
                          data=df[::5],
                          kind='kde',
                          color=COLORS[0],
                          size=4,
                          space=0,
                          stat_func=None,
                          xlim=[1.8,4.2],
                          ylim=[0, 1])
        g.set_axis_labels('$\log_{10}$ distance [pc]',
                'dilution, {:s} aperture'.format(ydim[-3:]))

        outname = '{:s}_vs_log10dist_Tmaglt16_members.pdf'.format(ydim)
        print('saving {:s}'.format(outname))
        g.savefig(outname, dpi=300, bbox_inches='tight')

    # vs Tmag plots
    for ydim in ['dil_2px', 'dil_3px', 'dil_4px']:
        plt.close('all')
        g = sns.jointplot(x='Tmag', y=ydim,
                          data=df[::5],
                          kind='kde',
                          color=COLORS[0],
                          size=4,
                          space=0,
                          stat_func=None,
                          xlim=[9,16.5],
                          ylim=[0, 1])
        g.set_axis_labels('T mag',
                'dilution, {:s} aperture'.format(ydim[-3:]))

        outname = '{:s}_vs_Tmag_Tmaglt16_members.pdf'.format(ydim)
        print('saving {:s}'.format(outname))
        g.savefig(outname, dpi=300, bbox_inches='tight')


def plot_dilution_scicase():
    '''
    Make the plot of log10(dilution [2px aperture]) vs log10(distance [pc]) for
    T<16 mag, d<2kpc cluster members.
    '''
    # Collect all dilutions, distances, Tmags
    data_dir = '../data/MWSC_dilution_calc/'
    csv_paths = [data_dir+f for f in os.listdir(data_dir)]

    df = pd.concat((pd.read_csv(f) for f in csv_paths), ignore_index=True)

    inds = df['dist'] < 2000
    df = df[inds]

    dil_2px = np.array(df['dil_2px']) # y
    dil_2px[dil_2px > 0.999 ] = 0.999

    plt.close('all')
    fig, ax = plt.subplots(figsize=(4,4))

    ax.set_xscale('log')
    ax.set_xlabel('(target flux)/(total flux in 2px TESS aperture)')
    ax.set_ylabel('probability density')

    ax.set_xlim((10**(-2.05), 1.1))
    ax.tick_params(which='both', direction='in', zorder=0)

    xmin, xmax = 10**(-3), 10**1

    log_dil_2px_bins = np.linspace(np.log10(xmin), np.log10(xmax), 17)

    x = 10**log_dil_2px_bins
    y = np.histogram(np.log10(dil_2px), log_dil_2px_bins)[0]
    x = np.array(list(zip(x[:-1], x[1:]))).flatten()
    y = np.array(list(zip(y, y))).flatten()
    ax.plot(x, y, lw=1, color='black')
    inds = (x <= 0.1)
    ax.fill_between(x[inds], y[inds], np.zeros_like(y[inds]), facecolor='none',
            hatch='/', edgecolor='gray', lw=0)

    frac_not_ok = np.sum(y[inds]) / np.sum(y)
    nonrecov_str = r'$\approx$'+'{:d}%\ntoo crowded'.format(int(100*frac_not_ok))
    recov_str = r'$\approx$'+'{:d}%\nrecoverable'.format(
            int(round(100*(1-frac_not_ok))))

    t = ax.text(10**(-0.5), 11500, recov_str,
            verticalalignment='center',horizontalalignment='center',fontsize='large')
    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='gray'))
    t= ax.text(10**(-1.5), 11500, nonrecov_str,
            verticalalignment='center',horizontalalignment='center',fontsize='large')
    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='gray'))

    #ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(which='both', direction='in', zorder=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim((0, max(ax.get_ylim())))

    ax.set_ylim((0, max(ax.get_ylim())))

    outname = 'dil_Tmaglt16_dlt2kpc_members.pdf'
    print('saving {:s}'.format(outname))
    fig.savefig(outname, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_dilution_fancy():
    '''
    Make the marginalized pllot of log10(dilution [2px aperture])for T<16 mag,
    d<2kpc cluster members.

    This one is to be included with the proposal.
    '''
    # Collect all dilutions, distances, Tmags
    data_dir = '../data/MWSC_dilution_calc/'
    csv_paths = [data_dir+f for f in os.listdir(data_dir)]

    df = pd.concat((pd.read_csv(f) for f in csv_paths), ignore_index=True)

    inds = df['dist'] < 2000
    df = df[inds]

    dist = np.array(df['dist']) # x
    dil_2px = np.array(df['dil_2px']) # y
    dil_2px[dil_2px > 0.999 ] = 0.999

    plt.close('all')
    fig = plt.figure(figsize=(4,4))

    #####################
    # Scatter and lines #
    #####################

    ax = plt.axes([0.1, 0.1, 0.6, 0.6])
    ax.plot(
        dist, dil_2px, 'o', color=COLORS[0], ms=3,
        alpha=0.02, rasterized=True, markeredgewidth=0, fillstyle='full'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    xmin, xmax = 10**(1.8), 10**(3.4)
    ymin, ymax = 10**(-3), 10**1
    ax.set_xlabel('distance [pc]')
    ax.set_ylabel('dilution, 2px TESS aperture')
    ax.xaxis.set_label_coords(0.5, -0.07)

    ax.set_xlim((10**1.8, 2050))
    ax.set_ylim((10**(-2.5), 1.1))
    ax.tick_params(which='both', direction='in', zorder=0)

    ##############
    # Histograms #
    ##############
    log_dil_2px_bins = np.linspace(np.log10(ymin), np.log10(ymax), 17)
    log_dist_bins = np.linspace(np.log10(xmin), np.log10(xmax), 17)

    n_bins, log_dil_2px_bins, log_dist_bins = np.histogram2d(
        np.log10(dil_2px), np.log10(dist), (log_dil_2px_bins, log_dist_bins),
    )

    # Top:
    ax = plt.axes([0.1, 0.71, 0.6, 0.15])

    x = 10**log_dist_bins
    y = np.histogram(np.log10(dist), log_dist_bins)[0]
    x = np.array(list(zip(x[:-1], x[1:]))).flatten()
    y = np.array(list(zip(y, y))).flatten()
    ax.plot(x, y, lw=1, color=COLORS[0])
    ax.fill_between(x, y, np.zeros_like(y), color=COLORS[0], alpha=0.2)

    ax.set_xscale('log')
    ax.set_xlim((10**1.8, 2050))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(which='both', direction='in', zorder=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim((0, max(ax.get_ylim())))

    # Right:
    ax = plt.axes([0.71, 0.1, 0.15, 0.6])

    x = 10**log_dil_2px_bins
    y = np.histogram(np.log10(dil_2px), log_dil_2px_bins)[0]
    x = np.array(list(zip(x[:-1], x[1:]))).flatten()
    y = np.array(list(zip(y, y))).flatten()
    ax.plot(y, x, lw=1, color=COLORS[0])
    ax.fill_betweenx(x, y, np.zeros_like(y), color=COLORS[0], alpha=0.2)

    ax.set_yscale('log')
    ax.set_ylim((10**(-2.5), 1.1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(which='both', direction='in', zorder=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim((0, max(ax.get_xlim())))

    outname = 'dil_vs_dist_Tmaglt16_dlt2kpc_members.pdf'
    print('saving {:s}'.format(outname))
    fig.savefig(outname, dpi=400, bbox_inches='tight')
    plt.close(fig)



def get_dilution_stats_for_text():
    '''
    An analysis of cluster members tabulated by Kharchenko et al. (2013) shows
    that the median cluster member with T < 16 will suffer flux dilution by
    factors of XX (YY) [ZZ] for TESS apertures with radii of 3 (4) and [5]
    pixels. This dilution is from both background stars and cluster neighbors.

    From experience, stars with dilution <1% are viable for aperture
    photometry, and those with dilution <50% for image subtraction. Image
    subtraction (or else PSF-fitting) is thus necessary to produce a useful
    number of precise cluster member lightcurves. The most dense fields, for
    instance those of distant globular clusters, are too crowded, but for the
    majority of clusters within ∼ 2 kpc our method is both feasible (Fig. 2)
    and necessary.
    '''

    # Collect all dilutions, distances, Tmags
    data_dir = '../data/MWSC_dilution_calc/'
    csv_paths = [data_dir+f for f in os.listdir(data_dir)]

    df = pd.concat((pd.read_csv(f) for f in csv_paths), ignore_index=True)

    df_lt_2kpc = df[df['dist']<2000]
    df_lt_1kpc = df[df['dist']<1000]
    df_lt_pt5kpc = df[df['dist']<500]

    # NOTE this calculation only kept track of dilutions for T<16 members. So
    # we're set.
    outstr = \
    '''
    ALL STATS ARE FOR MEMBER STARS WITH T<16 (ALL SKY).
    ##################################################

    T < 16
    {:s}

    ##################################################

    T < 16, d < 2kpc
    {:s}

    ##################################################

    T < 16, d < 1kpc
    {:s}

    ##################################################

    T < 16, d < 500pc
    {:s}
    '''.format(
    repr(df.describe()),
    repr(df_lt_2kpc.describe()),
    repr(df_lt_1kpc.describe()),
    repr(df_lt_pt5kpc.describe()),
    )

    with open('dilution_stats.out', 'w') as f:
        f.writelines(outstr)

    print(outstr)



def write_coord_list_for_MAST_crossmatch(df, max_dist, p_0=61):
    '''
    to find TIC crossmatches (and e.g., their radii) you need to do it by
    coordinates via MAST.

    so take the member stars. write a list of their coordinates.

    see `get_stellar_data_too` for arg description.

    args:
        max_dist (float): in pc, the maximum allowed distance for coords to
        write.
    '''

    names = np.array(df['Name'])
    r2s = np.array(df['r2'])
    # get MWSC ids in "0012", "0007" format
    mwsc = np.array(df['MWSC'])
    mwsc_ids = np.array([str(int(f)).zfill(4) for f in mwsc])

    readme = '../data/stellar_data_README'

    # loop over clusters
    ix = 0
    for mwsc_id, name, r2 in list(zip(mwsc_ids, names, r2s)):

        print('\n'+50*'*')
        print('{:d}. {:s}: {:s}'.format(ix, str(mwsc_id), str(name)))

        this_d = float(np.array(df.loc[df['Name']==name, 'd'])[0])
        if this_d > max_dist:
            print('SKIPPING. (Above max distance).')
            print('\n'+10*'~')
            continue

        middlestr = str(mwsc_id) + '_' + str(name)
        fpath = '../data/MWSC_stellar_data/2m_'+middlestr+'.dat'
        if name != 'Melotte_20':
            tab = ascii.read(fpath, readme=readme)
        else:
            continue

        # Select 1-sigma cluster members by photometry & kinematics.

        # From Kharchenko+ 2012, also require that:
        # * the 2MASS flag Qflg is "A" (i.e., signal-to-noise ratio
        #   S/N > 10) in each photometric band for stars fainter than
        #   Ks = 7.0;
        # * the mean errors of proper motions are smaller than 10 mas/yr
        #   for stars with δ ≥ −30deg , and smaller than 15 mas/yr for
        #   δ < −30deg.

        inds = (tab['Ps'] == 1)
        inds &= (tab['Pkin'] > p_0)
        inds &= (tab['PJKs'] > p_0)
        inds &= (tab['PJH'] > p_0)
        inds &= (tab['Rcl'] < r2)
        inds &= ( ((tab['Ksmag']>7) & (tab['Qflg']=='AAA')) | (tab['Ksmag']<7))
        pm_inds = ((tab['e_pm'] < 10) & (tab['DEdeg']>-30)) | \
                  ((tab['e_pm'] < 15) & (tab['DEdeg']<=-30))
        inds &= pm_inds

        members = tab[inds]
        members = members.to_pandas()

        print(np.max(members['RAhour']), np.min(members['RAhour']))

        RAs = (np.array(members['RAhour'])*u.hourangle).to(u.deg).value
        decs = np.array(members['DEdeg'])

        # length checks
        N_catalogd = int(df.loc[df['Name']==name, 'N1sr2'])
        N_my_onesigma = int(len(members))

        print('N catalogued as in cluster: {:d}'.format(N_catalogd))
        print('N I got as in cluster: {:d}'.format(N_my_onesigma))

        diff = abs(N_catalogd - N_my_onesigma)
        if diff > 5:
            print('\nWARNING: my cuts different from Kharachenko+ 2013!!')


        outdf = pd.DataFrame({'RA':RAs,
                              'DEC':decs,
                              'dist':np.ones_like(decs)*this_d
                            })

        fname = '../data/coords_lt_{:s}pc.csv'.format(str(max_dist))
        if ix == 0:
            outdf.to_csv(fname, index=False)
            ix += 1
        else:
            with open(fname, 'a') as f:
                outdf.to_csv(f, header=False, index=False)

        del members, tab, outdf


def raw_number_stats(df):
    '''
    1.
    how many cluster members brighter than T=16 are there in Kharchenko's list?

    2.
    how many cluster members brighter than T=16 with d<2,1,0.5kpc are there in
    Kharchenko's list?

    args:
        df: from `get_cluster_data()`. No distance cut, yet.
    '''

    obj = pickle.load(open('../data/Kharchenko_full_Tmag_lt_17.p','rb'))

    Nstar_T_lt_16 = 0
    Nstar_T_lt_16_d_lt_2kpc = 0
    Nstar_T_lt_16_d_lt_1kpc = 0
    Nstar_T_lt_16_d_lt_pt5kpc = 0

    type_d = {'a':'association',
              'g':'globular cluster',
              'm':'moving group',
              'n':'nebulosity/presence of nebulosity',
              'r':'remnant cluster',
              's':'asterism',
              '': 'no label'}

    type_count = {}

    for type_k in type_d.keys():
        type_count[type_k] = 0

    for k in obj.keys():

        #do not count globular clusters
        try:
            this_type = str(df.loc[df['Name']==k, 'Type'].iloc[0])
        except:
            import IPython; IPython.embed()

        if this_type == 'g':
            continue

        #only count clusters w/ centers in southern ecliptic hemisphere
        this_elat = float(df.loc[df['Name']==k, 'ecliptic_lat'])
        if this_elat > 0:
            continue

        type_count[this_type] += 1

        member_tmags = obj[k]['Tmag']
        tmag_inds = (member_tmags < 16) & (member_tmags > 0)

        this_dist = float(df.loc[df['Name']==k, 'd'])
        #note that many of these Tmag calculations failed (probably bc of
        #non-appropriate photometry).

        Nstar_T_lt_16 += len(member_tmags[tmag_inds])

        if this_dist < 500:
            Nstar_T_lt_16_d_lt_pt5kpc += len(member_tmags[tmag_inds])
        if this_dist < 1000:
            Nstar_T_lt_16_d_lt_1kpc += len(member_tmags[tmag_inds])
        if this_dist < 2000:
            Nstar_T_lt_16_d_lt_2kpc += len(member_tmags[tmag_inds])

    outstr = \
    '''
    do not count globular clusters.
    only count clusters whose centers are in the southern ecliptic hemisphere.

    1.
    how many cluster members brighter than T=16 are there in Kharchenko's list?
    {:d}

    2.
    how many cluster members brighter than T=16 with d<2,1,0.5kpc are there in
    Kharchenko's list?

    {:d},{:d},{:d} (in order)\n

    3.
    type count of each?
    association {:d}
    globular {:d}
    moving group {:d}
    nebulosity {:d}
    remnant {:d}
    asterism {:d}
    no label {:d}
    '''.format(
    int(Nstar_T_lt_16),
    int(Nstar_T_lt_16_d_lt_2kpc),
    int(Nstar_T_lt_16_d_lt_1kpc),
    int(Nstar_T_lt_16_d_lt_pt5kpc),
    int(type_count['a']),
    int(type_count['g']),
    int(type_count['m']),
    int(type_count['n']),
    int(type_count['r']),
    int(type_count['s']),
    int(type_count[''])
    )

    with open('../data/raw_number_stats.out', 'w') as f:
        f.writelines(outstr)

    print(outstr)


def planet_detection_estimate(df):
    '''
    estimate number of detections of hot Jupiters and hot Neptunes.
    '''

    def _get_detection_estimate(df, N_originally_in_globular, Rp, writetype,
            dilkey, occ_rate):
        '''
        args:
        -----
        df: a DataFrame of cluster members with T<16, selected to not include
        anything in globulars (i.e. our correct cluster definition) and to only
        be in the southern ecliptic hemisphere.

        N_originally_in_globular: number of member stars originally in globular
        clusters.

        dilkey (str): key to column header of dilution, e.g., 'dil_4px'

        Rp: astropy unitful planet radius assumed.

        writetype (str): 'a' for append, 'w' for write.

        occ_rate (float): fraction of stars with planet
        -----

        This routine assumes all cluster members are solar like stars. Using
        the Kharchenko+ 2013 cluster member photometry, I found T mags + noises
        for all the members (via ticgen).

        For a P=10day and P=3day planet of radius Rp, what fraction of the
        stars are detectable, at what thresholds?
        '''

        noise_1hr_in_ppm = np.array(df['noise_1hr'])
        noise_1hr_in_frac = noise_1hr_in_ppm/1e6

        dilution = df[dilkey]

        Rstar = np.array(df['radius'])*u.Rsun

        signal = ((Rp/Rstar).cgs)**2

        SNR_1hr = (signal / noise_1hr_in_frac)*np.sqrt(dilution)

        T_obs = 28*u.day

        P_long = 10*u.day

        # Compute transit duration, avg over impact param
        Mstar = np.array(df['mass'])*u.Msun
        vol_star = (4*pi/3)*Rstar**3
        rho_star = Mstar / vol_star
        vol_sun = (4*pi/3)*u.Rsun**3
        rho_sun = u.Msun / vol_sun

        T_dur_long = 13*u.hr * (P_long.to(u.yr).value)**(1/3) \
                             * (rho_star/rho_sun)**(-1/3)

        P_short = 3*u.day
        T_dur_short = 13*u.hr * (P_short.to(u.yr).value)**(1/3) \
                              * (rho_star/rho_sun)**(-1/3)

        T_in_transit_long = (T_obs / P_long)*T_dur_long*pi/4
        T_in_transit_short = (T_obs / P_short)*T_dur_short*pi/4

        SNR_pf_long = SNR_1hr * (T_in_transit_long.to(u.hr).value)**(1/2)
        SNR_pf_short = SNR_1hr * (T_in_transit_short.to(u.hr).value)**(1/2)

        # For how many cluster members can you get SNR > 10 in ONE HOUR?
        N_1hr = len(SNR_1hr[SNR_1hr > 10])

        # For how many cluster members can you get SNR > 10 phase folded,
        # assuming the long period?
        N_pf_long = len(SNR_pf_long[SNR_pf_long > 10])

        # For how many cluster members can you get SNR > 10 phase folded,
        # assuming the short period?
        N_pf_short = len(SNR_pf_short[SNR_pf_short > 10])


        import astropy.constants as const
        a_long = (const.G * Mstar / (4*pi*pi) * P_long**2 )**(1/3)
        transit_prob_long = (Rstar/a_long).cgs.value
        a_short = (const.G * Mstar / (4*pi*pi) * P_short**2 )**(1/3)
        transit_prob_short = (Rstar/a_short).cgs.value

        # For how many planets do you get SNR>10 in one hour?
        N_pla_1hr_long = 0
        N_pla_1hr_short = 0
        N_pla_pf_long = 0
        N_pla_pf_short = 0

        for ix, this_transit_prob in enumerate(transit_prob_long):
            if np.random.rand() < occ_rate * this_transit_prob:
                # Congrats, you have a transiting planet that exists
                if SNR_1hr[ix] > 10:
                    # Congrats, it's detected (1hr integration)
                    N_pla_1hr_long += 1
                if SNR_pf_long[ix] > 10:
                    # Congrats, it's detected (phase-folded)
                    N_pla_pf_long += 1

        for ix, this_transit_prob in enumerate(transit_prob_short):
            if np.random.rand() < occ_rate * this_transit_prob:
                # Congrats, you have a transiting planet that exists
                if SNR_1hr[ix] > 10:
                    # Congrats, it's detected (1hr integration)
                    N_pla_1hr_short += 1
                if SNR_pf_short[ix] > 10:
                    # Congrats, it's detected (phase-folded)
                    N_pla_pf_short += 1

        outstr = \
        '''
        ##################################################
        do not count globular clusters (threw out {:d} stars in globulars).
        only count clusters whose centers are in the southern ecliptic hemisphere.

        For Rp = {:.1f}, cluster star radii and masses from Joel's isochrones,
        dilution aperture radius of {:s}

        FRACTION OF STARS WITH PLANETS IS {:s}

        MEDIAN STELLAR RADIUS IS {:s}

        For how many cluster members can you get SNR > 10 in ONE HOUR?
        {:d}

        For how many cluster members can you get SNR > 10 phase folded, assuming
        the long period?
        {:d}

        For how many cluster members can you get SNR > 10 phase folded, assuming
        the short period?
        {:d}

        N_pla_1_hr_long: {:d}
        N_pla_1_hr_short: {:d}
        N_pla_pf_long: {:d}
        N_pla_pf_short: {:d}

        ##################################################
        '''.format(
        N_originally_in_globular,
        Rp,
        dilkey,
        repr(occ_rate),
        repr(np.median(Rstar)),
        N_1hr,
        N_pf_long,
        N_pf_short,
        N_pla_1hr_long,
        N_pla_1hr_short,
        N_pla_pf_long,
        N_pla_pf_short
        )

        with open('../data/planet_detection_estimate.out', writetype) as f:
            f.writelines(outstr)

        print(outstr)

    # These data files contain information for cluster members over the entire
    # sky with T<16.  Cluster members were selected with the Kharchenko+13
    # definition of 1sigma cluster members.  I attempted to find T mags with
    # `ticgen` for all such members. About 80% gave finite values (b/c of
    # whatever photometry was available for the members).  I then computed the
    # dilution as in `get_dilutions_and_distances` for various aperture sizes.
    # Handing this information + the Kharchenko catalogs off to Joel, he ran
    # them through Siess isochrones (thought not used often in exoplanet
    # studies, these have the advantage of covering 1Myr to 10Gyr and 0.1Msun
    # to 7Msun). About 90% gave finite values (those that did not were most
    # likely absolute Ks mags fell outside bounds spanned by isochrone at fixed
    # cluster age and [Fe/H]). Most of the stars are, unsurprisingly, larger
    # than Sun.  ALso note that the ages and FeHs Joel calculated in `dildf`
    # are the interpolated values in the isochrones -- AKA the FeHs will
    # generally disagree with those of Kharchenko. The ages should generally be
    # very close, except for Kharchenko values below 10Myr when the ages will
    # be set at ~10Myr b/c of the isochrone grid bounds.

    data_dir = '../data/MWSC_dilution_stellarparam_calc/'
    csv_paths = [data_dir+f for f in os.listdir(data_dir)]

    # This ungodly one-liner: to get all member stars into one DataFrame, read
    # in what's in the cluster-level data file, append a "name" column to the
    # read-in pandas dataframe (using pandas' equivalent of numpy array
    # broadcasting), then concatenate all the dataframes into a ~280k entry
    # long dataframe of members.
    dildf = pd.concat(((
                pd.read_csv(f).assign(
                    name=str(f.split('/')[-1][:-4]))
                ) for f in csv_paths), ignore_index=True)

    # Get cluster types for each cluster (and assign them to the member stars).
    # type_d = {'a':'association',
    #           'g':'globular cluster',
    #           'm':'moving group',
    #           'n':'nebulosity/presence of nebulosity',
    #           'r':'remnant cluster',
    #           's':'asterism',
    #           '': 'no label'}

    cluster_names = np.array(dildf['name'])
    cluster_types = np.repeat(np.array('', dtype=object), len(dildf))

    for cluster_name in np.unique(cluster_names):

        this_type = str(df.loc[df['Name']==cluster_name, 'Type'].iloc[0])

        matches = np.in1d(cluster_names, np.array(cluster_name))

        cluster_types[matches] = np.repeat(np.array(this_type, dtype=object),
                                           len(cluster_types[matches]))

    dildf = dildf.assign(cluster_type=cluster_types)

    c = SkyCoord(ra = np.array(dildf['RAhour'])*u.hourangle,
                 dec = np.array(dildf['DEdeg'])*u.degree)
    ecliptic_lat = np.array(c.barycentrictrueecliptic.lat)

    # Remove members in globular clusters.
    N_originally_in_globular = len(dildf[dildf['cluster_type']=='g'])
    no_globs = (dildf['cluster_type'] != 'g')

    # Only southern ecliptic hemisphere
    southern_only = (ecliptic_lat < 0)

    # Require Joel's radii are gt 0
    radii_gt_0 = (dildf['radius'] > 0)
    mass_gt_0 = (dildf['mass'] > 0)
    sensible_stars = radii_gt_0 & mass_gt_0

    df = dildf[no_globs & southern_only & sensible_stars]

    # Now with the above dataframe, you are set to get detection estimate.
    np.random.seed(42)
    # Use Howard's numbers for reasonable cases...
    _get_detection_estimate(df, N_originally_in_globular, 11*u.Rearth, 'w',
        'dil_2px', 0.005)
    _get_detection_estimate(df, N_originally_in_globular, 11*u.Rearth, 'a',
        'dil_3px', 0.005)
    _get_detection_estimate(df, N_originally_in_globular, 5*u.Rearth, 'a',
        'dil_2px', 0.01)
    _get_detection_estimate(df, N_originally_in_globular, 5*u.Rearth, 'a',
        'dil_3px', 0.01)
    _get_detection_estimate(df, N_originally_in_globular, 3*u.Rearth, 'a',
        'dil_2px', 0.025)
    _get_detection_estimate(df, N_originally_in_globular, 3*u.Rearth, 'a',
        'dil_3px', 0.025)



if __name__ == '__main__':

    #make_wget_script(df)

    close, far, df = get_cluster_data()
    #close, far, df = get_stellar_data_too(df, 'Tmag_lt_17')
    #get_dilutions_and_distances(df, 'Tmag_lt_16', faintest_Tmag=16, p_0=61)

    #distance_histogram(df)
    #angular_scale_cumdist(close, far)
    #angular_scale_hist(close, far)

    #mean_density_hist(close, far)
    #plot_king_profiles(close, far)
    #plot_density_vs_Tmag_scatter(close, far)

    #plot_empirical_density_vs_Tmag_scatter(close, far)

    #plot_HATS_field_positions()

    #write_coord_list_for_MAST_crossmatch(df, 2000, p_0=61)

    #raw_number_stats(df)

    #planet_detection_estimate(df)

    #get_dilution_stats_for_text()

    #plot_dilution_vs_dist_and_Tmag()


    ############################
    # FIGURES USED IN PROPOSAL #
    ############################

    # Both outdated versions from draft #1
    #plot_cluster_positions(close, far)
    #plot_dilution_fancy()

    # Present versions
    plot_cluster_positions_scicase(df)
    #plot_dilution_scicase()
