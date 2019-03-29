"""
tools for plotting publication-quality (or proposal quality) RMS vs mag
diagrams.
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, subprocess, itertools
from datetime import datetime

from numpy import array as nparr

from astroquery.gaia import Gaia
from astropy.io.votable import from_table, writeto, parse
from astropy.table import Table

from tqdm import tqdm

from noise_model import noise_model

def scp_lightcurves(lcbasenames,
                    lcdir='/nfs/phtess1/ar1/TESS/FFI/LC/FULL/s0002/ISP_1-2-1163',
                    outdir='../data/cluster_data/lightcurves/Blanco_1/'):

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for lcbasename in lcbasenames:

        fromstr = "lbouma@phn12:{}/{}".format(lcdir, lcbasename)
        tostr = "{}/.".format(outdir)

        p = subprocess.Popen([
            "scp",
            fromstr,
            tostr,
        ])
        sts = os.waitpid(p.pid, 0)

    return 1


def get_rms_v_mag_from_phtess1():

    # pattern:
    # /nfs/phtess1/ar1/TESS/FFI/LC/FULL/s0002/ISP_1-2-1301/stats_files/percentiles_RMS_vs_med_mag_TF2.png

    projids = range(1300,1309) # currently all that are done

    for projid in projids:
        remoteglob = (
            '/nfs/phtess1/ar1/TESS/FFI/LC/FULL/s0002/'
            'ISP_?-?-{}/stats_files/percentiles_RMS_vs_med_mag_TF2.png'.
            format(projid)
        )

        localdir = '../results/rms_vs_mag/20190310_status_check'
        if not os.path.exists(localdir):
            os.mkdir(localdir)

        localpath = os.path.join(
            localdir, '{}_percentiles_RMS_vs_med_mag_TF2.png'.format(projid))

        fromstr = "lbouma@phn12:{}".format(remoteglob)
        tostr = localpath

        p = subprocess.Popen([
            "scp",
            fromstr,
            tostr,
        ])
        sts = os.waitpid(p.pid, 0)

        print('scp {} {}'.format(fromstr, tostr))


def read_stats_file(statsfile, fovcathasgaiaids=False):
    """
    Reads the stats file into a numpy recarray.
    (Annoying code duplication from pipe-trex)
    """

    if fovcathasgaiaids:
        idstrlength = 19
    else:
        idstrlength = 17

    # open the statfile and read all the columns
    stats = np.genfromtxt(
        statsfile,
        dtype=(
            'U{:d},f8,'
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RM1
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RM2
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RM3
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # EP1
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # EP2
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # EP3
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # TF1
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # TF2
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # TF3
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RF1
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RF2
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RF3
            'f8,f8,f8'.format(idstrlength)    # corrmags
        ),
        names=[
            'lcobj','cat_mag',
            'med_rm1','mad_rm1','mean_rm1','stdev_rm1','ndet_rm1',
            'med_sc_rm1','mad_sc_rm1','mean_sc_rm1','stdev_sc_rm1',
            'ndet_sc_rm1',
            'med_rm2','mad_rm2','mean_rm2','stdev_rm2','ndet_rm2',
            'med_sc_rm2','mad_sc_rm2','mean_sc_rm2','stdev_sc_rm2',
            'ndet_sc_rm2',
            'med_rm3','mad_rm3','mean_rm3','stdev_rm3','ndet_rm3',
            'med_sc_rm3','mad_sc_rm3','mean_sc_rm3','stdev_sc_rm3',
            'ndet_sc_rm3',
            'med_ep1','mad_ep1','mean_ep1','stdev_ep1','ndet_ep1',
            'med_sc_ep1','mad_sc_ep1','mean_sc_ep1','stdev_sc_ep1',
            'ndet_sc_ep1',
            'med_ep2','mad_ep2','mean_ep2','stdev_ep2','ndet_ep2',
            'med_sc_ep2','mad_sc_ep2','mean_sc_ep2','stdev_sc_ep2',
            'ndet_sc_ep2',
            'med_ep3','mad_ep3','mean_ep3','stdev_ep3','ndet_ep3',
            'med_sc_ep3','mad_sc_ep3','mean_sc_ep3','stdev_sc_ep3',
            'ndet_sc_ep3',
            'med_tf1','mad_tf1','mean_tf1','stdev_tf1','ndet_tf1',
            'med_sc_tf1','mad_sc_tf1','mean_sc_tf1','stdev_sc_tf1',
            'ndet_sc_tf1',
            'med_tf2','mad_tf2','mean_tf2','stdev_tf2','ndet_tf2',
            'med_sc_tf2','mad_sc_tf2','mean_sc_tf2','stdev_sc_tf2',
            'ndet_sc_tf2',
            'med_tf3','mad_tf3','mean_tf3','stdev_tf3','ndet_tf3',
            'med_sc_tf3','mad_sc_tf3','mean_sc_tf3','stdev_sc_tf3',
            'ndet_sc_tf3',
            'med_rf1','mad_rf1','mean_rf1','stdev_rf1','ndet_rf1',
            'med_sc_rf1','mad_sc_rf1','mean_sc_rf1','stdev_sc_rf1',
            'ndet_sc_rf1',
            'med_rf2','mad_rf2','mean_rf2','stdev_rf2','ndet_rf2',
            'med_sc_rf2','mad_sc_rf2','mean_sc_rf2','stdev_sc_rf2',
            'ndet_sc_rf2',
            'med_rf3','mad_rf3','mean_rf3','stdev_rf3','ndet_rf3',
            'med_sc_rf3','mad_sc_rf3','mean_sc_rf3','stdev_sc_rf3',
            'ndet_sc_rf3',
            'corr_mag_ap1','corr_mag_ap2','corr_mag_ap3',
        ]
    )

    return stats


def plot_rms_vs_mag(stats, yaxisval='RMS', percentiles_xlim=[5.8,16.2],
                    percentiles_ylim=[1e-5,1e-1], percentiles=[25,50,75],
                    outdir='../results/rms_vs_mag/', outprefix='',
                    catmagstr=None):
    """
    catmagstr (str): None, 'Tmag', or 'cat_mag' (where the latter is Gaia Rp)
    """

    if isinstance(catmagstr,str):
        if catmagstr=='Tmag':
            df = pd.read_csv('../data/rms_vs_mag/projid_1301_gaia_2mass_tic_all_matches.csv')

            matched_gaia_ids = np.array(df['source_id']).astype(int)
            stats_ids = stats['lcobj'].astype(int)

            int1d, stats_ind, df_ind = np.intersect1d(stats_ids, matched_gaia_ids,
                                                      return_indices=True)

            print('starting with {} LCs w/ Gaia IDs'.format(len(stats)))
            stats = stats[stats_ind]
            print('post xmatch {} LCs w/ Gaia IDs & 2mass ids & Tmag'.
                  format(len(stats)))

            # NOTE: this makes no sense. wtf. why are so few in the intersection?
            df = df.iloc[df_ind]

            assert np.array_equal(stats['lcobj'].astype(int),
                                  df['source_id'].astype(int))

            # if above is true, then can do this
            mags = df['tmag']


    apstr='tf2'

    medstr = 'med_'+apstr if not isinstance(catmagstr,str) else catmagstr
    yvalstr = 'stdev_'+apstr

    if catmagstr != 'Tmag':
        mags = stats[medstr]
    rms = stats[yvalstr]

    minmag = np.floor(np.nanmin(mags)).astype(int)
    maxmag = np.ceil(np.nanmax(mags)).astype(int)
    magdiff = 0.25
    mag_bins = [(me, me+magdiff) for me in np.arange(minmag, maxmag, magdiff)]

    percentile_dict = {}
    for mag_bin in mag_bins:

        thismagmean = np.round(np.mean(mag_bin),2)
        percentile_dict[thismagmean] = {}

        thismin, thismax = min(mag_bin), max(mag_bin)
        sel = (mags > thismin) & (mags <= thismax)

        for percentile in percentiles:
            val = np.nanpercentile(stats[sel][yvalstr], percentile)
            percentile_dict[thismagmean][percentile] = np.round(val,7)

    pctile_df = pd.DataFrame(percentile_dict)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(0.8*4,0.8*3))

    # this plot is lines for each percentile in [2,25,50,75,98],
    for ix, row in pctile_df.iterrows():
        pctile = row.name
        label = '{}%'.format(str(pctile))
        if label == '50%':
            label = "Median"
        midbins, vals = nparr(row.index), nparr(row)

        # NOTE: not showing
        #ax.plot(midbins, vals, label=label, marker='o', ms=0, zorder=3, lw=0.5)

    ax.scatter(mags, rms, c='k', alpha=0.12, zorder=-5, s=0.5,
               rasterized=True, linewidths=0)

    if yaxisval=='RMS':
        Tmag = np.linspace(6, 16, num=200)

        # lnA = 3.29685004771
        # B = 0.8500214657
        # C = -0.2850416324
        # D = 0.039590832137
        # E = -0.00223080159
        # F = 4.73508403525e-5
        # ln_sigma_1hr = (lnA + B*Tmag + C*Tmag**2 + D*Tmag**3 + E*Tmag**4 +
        #                 F*Tmag**5)
        # sigma_1hr = np.exp(ln_sigma_1hr)
        # sigma_30min = sigma_1hr * np.sqrt(2)

        # RA, dec. (90, -66) is southern ecliptic pole. these are "good
        # coords", but we don't care!
        coords = np.array([90*np.ones_like(Tmag), -66*np.ones_like(Tmag)]).T
        out = noise_model(Tmag, coords=coords, exptime=1800)

        noise_star = out[2,:]
        noise_ro = out[4,:]
        noise_star_plus_ro = np.sqrt(noise_star**2 + noise_ro**2)

        ax.plot(Tmag, noise_star_plus_ro, ls='-', zorder=-2, lw=1, color='C1',
                label='Photon + read')
        ax.plot(Tmag, noise_star, ls='--', zorder=-3, lw=1, color='C3',
                label='Photon')
        ax.plot(Tmag, noise_ro, ls='--', zorder=-4, lw=1, color='C4',
                label='Read')

    ax.legend(loc='upper left', fontsize='xx-small')
    ax.set_yscale('log')
    xlabel = 'Instrument magnitude'
    if catmagstr=='cat_mag':
        xlabel = 'TESS magnitude' # NOTE: this is a lie. but a white one -- we checked, they're the same!! just cross-matching is insanely annoying 'Gaia Rp magnitude'
    if catmagstr=='Tmag':
        xlabel = 'TESS magnitude'
    ax.set_xlabel(xlabel, labelpad=0.8)
    ax.set_ylabel('RMS (30 minutes)', labelpad=0.8)
    ax.set_xlim(percentiles_xlim)
    ax.set_ylim(percentiles_ylim)

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')

    if not isinstance(catmagstr,str):
        savname = os.path.join(outdir,'percentiles_{:s}_vs_med_mag_{:s}.png'.
            format(yaxisval, apstr.upper()))
    else:
        if catmagstr=='cat_mag':
            savname = os.path.join(outdir,'percentiles_{:s}_vs_GaiaRp_mag_{:s}.png'.
                format(yaxisval, apstr.upper()))
        if catmagstr=='Tmag':
            savname = os.path.join(outdir,'percentiles_{:s}_vs_TESS_mag_{:s}.png'.
                format(yaxisval, apstr.upper()))
    fig.tight_layout(pad=0.2)
    fig.savefig(savname, dpi=400)
    print('%sZ: made plot: %s' % (datetime.utcnow().isoformat(), savname))


def xmatch_gaia_to_Tmag_relations(stats, projid='1301'):
    """
    using gaia IDs. get matches in tmass_best_neighbour table. use the
    resulting Gmag and 2mass photometry values, + the relations given by
    Stassun, to get Tmags.

    (simpler than spatial+magnitude crossmatch)
    """

    catdir = '/home/luke/local/tess-trex/catalogs/proj1301'
    reformedcatalogfile = os.path.join(
        catdir,
        'GAIADR2-RA1.2049336110863-DEC-26.0582248258775-SIZE24.reformed_catalog')

    columns='id,ra,dec,xi,eta,G,Rp,Bp,plx,pmra,pmdec'
    columns = columns.split(',')
    catarr = np.genfromtxt(reformedcatalogfile,
                           comments='#',
                           usecols=list(range(len(columns))),
                           names=columns,
                           dtype='U19,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8',
                           delimiter=' ')

    catarr_w_lcs = catarr[np.in1d(catarr['id'],stats['lcobj'])]

    # match catarr to whatever "stats" lightcurve exist TODO

    # i have gaia IDs. i will get 2mass information via... the gaia to 2mass
    # crossmatch table
    tmass_ids = []
    for source_id in tqdm(stats['lcobj']):
        querystr = (
            "select top 1 * from gaiadr2.tmass_best_neighbour "
            "where source_id = {}".format(source_id)
        )
        job = Gaia.launch_job(querystr)
        res = job.get_results()

        if len(res)==0:
            tmass_ids.append(-1)
        elif len(res)==1:
            tmass_ids.append(res['original_ext_source_id'].data.data[0].decode('utf-8'))
        else:
            raise AssertionError

    outdf = pd.DataFrame(
        {'gaia_dr2_id':stats['lcobj'], 'tmass_id':tmass_ids}
    )
    outpath = (
        '../data/rms_vs_mag/proj{}_xmatch_gaia_to_tmass.csv'.format(projid)
    )
    outdf.to_csv(outpath,index=False)
    print('saved {}'.format(outpath))


def make_votable_given_ids(gaiaids,
                           outpath='../data/rms_vs_mag/proj1301_gaiaids.xml'):

    t = Table()
    t['source_id'] = gaiaids.astype(int)

    votable = from_table(t)

    writeto(votable, outpath)
    print('made {}'.format(outpath))

    return outpath


def vector_match_tmassbestneighbor_to_gaiaids(stats, homedir='/home/luke/',
                                              outdir='../data/rms_vs_mag/',
                                              projid=1301):
    """
    if you do smart ADQL, it is like factor of >100x faster than doing
    item-by-item crossmatching.

    astroquery.Gaia is sick for this, because it lets you remotely upload
    tables on the fly. (at least, somewhat small ones, of <~10^5 members)
    """

    outfile = os.path.join(outdir,'proj{}_xmatch.xml.gz'.format(projid))

    xmlpath = '../data/rms_vs_mag/proj1301_gaiaids.xml'
    if not os.path.exists(outfile):
        gaiaids = stats['lcobj']
        xmlpath = make_votable_given_ids(gaiaids)

        Gaia.login(credentials_file=os.path.join(homedir, '.gaia_credentials'))

        jobstr = (
            'SELECT top 100000 upl.source_id, tm.source_id, '
            'tm.original_ext_source_id FROM '
            'gaiadr2.tmass_best_neighbour AS tm '
            'JOIN tap_upload.foobar as upl '
            'using (source_id)'
        )
        if not os.path.exists(outfile):
            # might do async if this times out. but it doesn't.
            j = Gaia.launch_job(query=jobstr,
                                upload_resource=xmlpath,
                                upload_table_name="foobar", verbose=True,
                                dump_to_file=True, output_file=outfile)

        Gaia.logout()

    vot = parse(outfile)
    tab = vot.get_first_table().to_table()

    return tab

def single_repeated_query_mast_casjob(tab):
    """
    the dumb way to get the Tmags once you have the 2mass IDs.
    i really wish TIC 8 existed. and that the interface for doing this was
    better. i guess an alternative is going to the MAST website and clicking
    around? but FALSE. it's not. you can only crossmatch on position LOL.
    """
    outpath = '../data/rms_vs_mag/projid_1301_gaia_2mass_tic_all_matches.csv'
    if os.path.exists(outpath):
        print('skipping single repeated casjob query')
        return

    import mastcasjobs

    with open('~/.mast_casjob_credentials', 'r') as f:
        lines = f.readlines()

    user = lines[0].rstrip('\n')
    pwd = lines[1].rstrip('\n') # http://mastweb.stsci.edu/mcasjobs/
    wsid = lines[2].rstrip('\n')  # given in the profile

    jobs = mastcasjobs.MastCasJobs(
        userid=wsid, password=pwd, context="TESS_v7")

    tmags = []
    for ix, tmass_id in enumerate(tab['original_ext_source_id']):
        print('{}: {}/{}'.format(datetime.utcnow().isoformat(), ix, len(tab) ))

        query = (
            "select top 1 TWOMASS, Tmag from catalogRecord where TWOMASS = \'{}\'".
            format(tmass_id.decode('utf-8'))
        )

        results = jobs.quick(query, task_name="foobar")
        if len(results.split('\n'))==3:
            tmags.append(float(results.split('\n')[1].split(',')[-1]))
        else:
            tmags.append(-1)

    tab['tmag'] = tmags
    outdf = tab.to_pandas()
    outdf.to_csv(outpath,index=False)
    print('saved {}'.format(outpath))

def vector_match_mast_casjob(tab):
    """
    (currently not working) attempt at getting a proper table join to work on
    the proj1301_xmatch.csv table that I uploaded to
    http://mastweb.stsci.edu/mcasjobs/.

    uses the https://github.com/rlwastro/mastcasjobs wrapper to DFM's casjobs
    wrapper

    (wrappers on wrappers)
    """

    assert NotImplementedError

    import mastcasjobs

    with open('~/.mast_casjob_credentials', 'r') as f:
        lines = f.readlines()

    user = lines[0].rstrip('\n')
    pwd = lines[1].rstrip('\n') # http://mastweb.stsci.edu/mcasjobs/
    wsid = lines[2].rstrip('\n')  # given in the profile

    jobs = mastcasjobs.MastCasJobs(
        userid=wsid, password=pwd, context="TESS_v7"
    )
    query = "select top 100 * from catalogRecord"
    results = jobs.quick(query, task_name="foobar")
    print(results)

    #FIXME
    # # buest guess so far... but it doesn't work!!!
    # SELECT top 100 tic.Tmag, tic.TWOMASS, upl.tmass_id
    # FROM catalogRecord as tic
    # JOIN MyDB.proj1301_xmatch as upl
    # ON (tic.TWOMASS = CONVERT(nvarchar(max),upl.tmass_id);

    query = (
        'SELECT top 100 tic.Tmag, tic.TWOMASS, upl.tmass_id '
        'FROM catalogRecord as tic '
        'JOIN MyDB.proj1301_xmatch as upl '
        'ON (tic.TWOMASS = CONVERT(nvarchar(max),upl.tmass_id);'
    )

    jobid = jobs.submit(query, task_name="foobar")


    results = jobs.status(jobid)

    #FIXME

    import IPython; IPython.embed()
    pass


if __name__ == "__main__":

    # scp RMS v mag plots from phtess1 to see which projid to use
    get_rms_v_mag = 0
    # TIC8 is postponed. we need relations to go from Gaia DR2 to TESS mag.
    calibrate_gaia_to_Tmag=0
    # make the RMS v mag plot for cycle II proposal / paper
    make_rms_v_mag = 1

    if get_rms_v_mag:
        get_rms_v_mag_from_phtess1()

    statsfile = "../data/rms_vs_mag/proj1301_camera1_ccd2.tfastats"
    stats = read_stats_file(statsfile, fovcathasgaiaids=True)

    if calibrate_gaia_to_Tmag:

        # Crossmatch Gaia IDs against gaiadr2.tmass_best_neighbour table in
        # like a minute with astroquery.Gaia's remote upload
        tab = vector_match_tmassbestneighbor_to_gaiaids(stats)

        # Crossmatch 2MASS IDs against TIC.  Takes like 4 hours, b/c late at
        # night I couldn't figure out the SQL vectorization, and/or if CasJobs
        # lets you upload tables from remote (w/out needing to login to their
        # website...)
        single_repeated_query_mast_casjob(tab)
        # xmatch_gaia_to_Tmag_relations(stats)

    if make_rms_v_mag:

        plot_rms_vs_mag(stats, outprefix='proj1301(proposal)',
                        catmagstr='Tmag')

        plot_rms_vs_mag(stats, outprefix='proj1301(proposal)',
                        catmagstr='cat_mag')

        plot_rms_vs_mag(stats, outprefix='proj1301(proposal)', catmagstr=None)
