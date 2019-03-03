import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os, pickle, subprocess, itertools
from numpy import array as nparr
from glob import glob

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def scp_rms_percentile_files(statsdir, outdir):

    if not os.path.exists(outdir):
        os.mkdir(outdir)
        print('made {}'.format(outdir))

    p = subprocess.Popen([
        "scp", "lbouma@phn12:{}/percentiles_RMS_vs_med_mag*.csv".format(statsdir),
        "{}/.".format(outdir),
    ])
    sts = os.waitpid(p.pid, 0)

    return 1


def scp_knownplanets(statsdir, outdir):

    if not os.path.exists(outdir):
        os.mkdir(outdir)
        print('made {}'.format(outdir))

    p = subprocess.Popen([
        "scp", "lbouma@phn12:{}/*-0*/*_TFA*_snr_of_fit.csv".format(statsdir),
        "{}/.".format(outdir),
    ])
    sts = os.waitpid(p.pid, 0)

    return 1



def get_rms_file_dict(projids, camccdstr, ap='TF1'):

    d = {}
    for projid in projids:

        run_id = "{}-{}".format(camccdstr, projid)
        indir = (
            '../results/optimizing_pipeline_parameters/{}_stats'.format(run_id)
        )
        inpath = os.path.join(
            indir, "percentiles_RMS_vs_med_mag_{}.csv".format(ap))

        d[projid] = pd.read_csv(inpath)

    return d


def get_knownplanet_file_dict(projids, camccdstr, ap='TFA1'):

    d = {}
    for projid in projids:

        d[projid] = {}

        run_id = "{}-{}".format(camccdstr, projid)
        indir = (
            '../results/optimizing_pipeline_parameters/{}_knownplanets'.format(run_id)
        )

        inpaths = glob(
            os.path.join(
                indir, "*-0*_{}_snr_of_fit.csv".format(ap)
            )
        )

        for inpath in inpaths:

            toi_id = str(
                os.path.basename(inpath).split('_')[0].replace('-','.')
            )

            d[projid][toi_id] = pd.read_csv(inpath)

    return d


def plot_rms_comparison(d, projids, camccdstr, expmtstr, overplot_theory=True,
                        apstr='TF1', yaxisval='RMS',
                        outdir='../results/optimizing_pipeline_parameters/',
                        xlim=[6,16], ylim=[3e-5*1e6, 1e-2*1e6],
                        descriptionkey=None):
    """
    descriptionkey: e.g., "kernelspec". or "aperturelist", or whatever.
    """

    fig,ax = plt.subplots(figsize=(4,3))

    pctiles = [25,50,75]

    defaultcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    defaultlines = ['-.','-',':','--']*2
    if len(projids) > len(defaultcolors):
        raise NotImplementedError('need a new color scheme')
    if len(projids) > len(defaultlines):
        raise NotImplementedError('need a new line scheme')

    linestyles = defaultlines[:len(projids)]
    colors = defaultcolors[:len(projids)]
    #linestyles = itertools.cycle(defaultlines[:len(projids)])
    #colors = itertools.cycle(defaultcolors[:len(projids)])

    for projid, color in zip(projids, colors):

        run_id = "{}-{}".format(camccdstr, projid)

        for ix, ls in zip([0,1,2], linestyles):

            midbins = nparr(d[projid].iloc[ix].index).astype(float)
            rms_vals = nparr(d[projid].iloc[ix])

            if isinstance(descriptionkey,str):
                pdict = pickle.load(open(
                    '../data/reduction_param_pickles/projid_{}.pickle'.
                    format(projid), 'rb')
                )
                description = pdict[descriptionkey]

            label = '{}. {}%'.format(
                description, str(pctiles[ix])
            )

            ax.plot(midbins, rms_vals*1e6/(2**(1/2.)), label=label,
                    color=color, lw=1, ls=ls)

    ax.text(0.98, 0.02, camccdstr, transform=ax.transAxes, ha='right',
            va='bottom', fontsize='xx-small')

    if overplot_theory:
        # show the sullivan+2015 interpolated model
        Tmag = np.linspace(6, 13, num=200)
        lnA = 3.29685004771
        B = 0.8500214657
        C = -0.2850416324
        D = 0.039590832137
        E = -0.00223080159
        F = 4.73508403525e-5
        ln_sigma_1hr = (
            lnA + B*Tmag + C*Tmag**2 + D*Tmag**3 +
            E*Tmag**4 + F*Tmag**5
        )
        sigma_1hr = np.exp(ln_sigma_1hr)
        sigma_30min = sigma_1hr * np.sqrt(2)

        ax.plot(Tmag, sigma_1hr, 'k-', zorder=3, lw=1,
                label='S+15 $\sigma_{\mathrm{1hr}}$ (interp)')

    ax.legend(loc='upper left', fontsize=4)

    ax.set_yscale('log')
    ax.set_xlabel('{:s} median instrument magnitude'.
                  format(apstr.upper()))
    ax.set_ylabel('{:s} {:s}'.
                  format(apstr.upper(), yaxisval)
                  +' [$\mathrm{ppm}\,\mathrm{hr}^{1/2}$]')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    projidstr = ''
    for p in projids:
        projidstr += '_{}'.format(p)

    theorystr = '_withtheory' if overplot_theory else ''

    savname = ( os.path.join(
        outdir,'{}compare_percentiles{}{}_{:s}_vs_med_mag_{:s}.png'.
        format(expmtstr, projidstr, theorystr, yaxisval, apstr.upper())
    ))
    fig.tight_layout()
    fig.savefig(savname, dpi=300)
    print('made {}'.format(savname))


def plot_knownplanet_comparison(d, projids, camccdstr, expmtstr, apstr='TFA1',
                                descriptionkey='kernelspec',
                                outdir='../results/optimizing_pipeline_parameters/',
                                ylim=None, divbymaxsnr=False):

    plt.close('all')
    fig,ax = plt.subplots(figsize=(4,3))

    defaultcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(projids) < len(defaultcolors):
        colors = defaultcolors[:len(projids)]
    if len(projids) > len(defaultcolors):
        if len(projids)==13:
            # cf http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
            # seaborn paired 12 color + black.
            colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
                      "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a",
                      "#ffed6f", "#b15928", "#000000"]
        else:
            # give up trying to do smart color schemes. just loop it, and
            # overlap. (the use of individual colors becomes small at N>13
            # anyway).
            colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
                      "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a",
                      "#ffed6f", "#b15928", "#000000"]
            colors *= 4
            colors = colors[:len(projids)]

    #colors = itertools.cycle(defaultcolors[:len(projids)])

    # may need to get max snr over projids, for each toi.
    n_projids = len(projids)

    toi_counts, temp_tois = [], []
    for projid in projids:
        toi_ids = np.sort(list(d[projid].keys()))
        toi_counts.append(len(toi_ids))
        temp_tois.append(toi_ids)

    # some projids have different numbers of TOIs. they will get nans.
    utois = np.sort(np.unique(np.concatenate(temp_tois)))

    # some TOIs are "bad" and need to be filtered. (e.g., if they have
    # systematics that mess with BLS, and so you dont want them in this
    # comparison). NOTE: these are manually appended.
    manual_bad_tois = ['243', '354', '359']
    utois = [u for u in utois if u.split('.')[0] not in manual_bad_tois]

    n_tois = len(utois)

    arr = np.zeros((n_projids, n_tois))
    for i, projid in enumerate(projids):
        for j, toi in enumerate(utois):
            if toi in list(d[projid].keys()):
                arr[i,j] = d[projid][toi]['trapz_snr']
            else:
                arr[i,j] = np.nan

    arr[arr == np.inf] = 0
    max_snrs = np.nanmax(arr, axis=0)
    max_snr_d = {}
    for toi, max_snr in zip(utois, max_snrs):
        max_snr_d[toi] = max_snr

    # if you didn't get anything, set snr to 0. this way helps with the mean.
    snrs_normalized = arr/max_snrs[None,:]
    snrs_normalized[np.isnan(snrs_normalized)] = 0
    normalized_snr_avgd_over_tois = np.mean(snrs_normalized, axis=1)
    nzeros = np.sum((snrs_normalized==0).astype(int),axis=1)

    for projid, color, nsnr, nz in zip(
        projids, colors, normalized_snr_avgd_over_tois, nzeros
    ):

        run_id = "{}-{}".format(camccdstr, projid)

        labeldone = False
        for ix, toi_id in enumerate(utois):

            if toi_id in list(d[projid].keys()):
                snr_val = float(d[projid][toi_id]['trapz_snr'])
            else:
                snr_val = np.nan
            print(toi_id, snr_val)

            if isinstance(descriptionkey,str):
                pdict = pickle.load(open(
                    '../data/reduction_param_pickles/projid_{}.pickle'.
                    format(projid), 'rb')
                )
                description = pdict[descriptionkey]

            label = '{}'.format(description)
            if divbymaxsnr:
                label = '{} ({:.2f}) [{}/{}]'.format(
                    description, nsnr, nz, len(utois)
                )

            if not pd.isnull(snr_val) and not labeldone:
                yval = snr_val
                if divbymaxsnr:
                    yval /= max_snr_d[toi_id]
                ax.scatter(ix, yval, label=label, color=color, lw=0, s=4)
                labeldone = True
            else:
                yval = snr_val
                if divbymaxsnr:
                    yval /= max_snr_d[toi_id]
                ax.scatter(ix, yval, color=color, lw=0, s=4)

    ax.set_title(camccdstr, fontsize='x-small')

    # shrink axes by 10% in x; add legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=4)
    if divbymaxsnr:
        titlestr = (
            r'kernel ($\langle \mathrm{SNR}/\mathrm{SNR}_{\mathrm{max}} \rangle$)'+
            '[$N_{\mathrm{failed}} / N_{\mathrm{total}}$]'
        )
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=4,
                  title=titlestr, title_fontsize=4)

    ax.set_xticks(range(len(utois)))

    xtl = [u[:-3] for u in utois]
    ax.set_xticklabels(xtl, rotation=60, fontsize='small')
    ax.set_xlabel('TOI ID')
    if divbymaxsnr:
        xtl = [u[:-3] + '({:.1f})'.format(m)
               for u,m in zip(utois, max_snrs)]
        ax.set_xticklabels(xtl, rotation=90, fontsize='small')
        ax.set_xlabel('TOI (SNR$_{\mathrm{max}}$)')

    ax.set_ylabel('{:s} SNR'.format(apstr.upper()))
    if divbymaxsnr:
        ax.set_ylabel('{:s} SNR'.format(apstr.upper())+
                      '/ SNR$_{\mathrm{max}}$')

    if ylim is None:
        ax.set_ylim((0, 1.05*np.nanmax(max_snrs[np.isfinite(max_snrs)])))
    else:
        ax.set_ylim(ylim)

    ax.tick_params(axis='both', which='major', labelsize='small')

    projidstr = ''
    for p in projids:
        projidstr += '_{}'.format(p)

    divbymaxsnrstr = 'divmaxsnr' if divbymaxsnr else ''

    savname = ( os.path.join(
        outdir,'{}compare_knownplanets{}{}_{:s}.png'.
        format(expmtstr, divbymaxsnrstr, projidstr, apstr.upper())
    ))
    fig.tight_layout()
    fig.savefig(savname, dpi=300, bbox_inches='tight')
    print('made {}'.format(savname))
    plt.close('all')


def scp_projid_pickles(indir, outdir):

    if not os.path.exists(outdir):
        os.mkdir(outdir)
        print('made {}'.format(outdir))

    p = subprocess.Popen([
        "scp", "lbouma@phn12:{}/*.pickle".format(indir),
        "{}/.".format(outdir),
    ])
    sts = os.waitpid(p.pid, 0)


def projid_pickle_to_txt(projids, pkldir, outdir, desiredkeys=None):

    d = {}

    for projid in projids:

        pklpath = os.path.join(pkldir, 'projid_{}.pickle'.format(projid))
        d[projid] = pickle.load(open(pklpath,'rb'))

    outlines = []
    for projid in projids:

        outlines.append('='*42)
        outlines.append('PROJID {}'.format(projid))

        if desiredkeys is None:
            for k,v in d[projid].items():
                thisline = '{}: {}'.format(k, v)
                outlines.append(thisline)

        elif isinstance(desiredkeys, list):
            for k in desiredkeys:
                thisline = '{}: {}'.format(k, d[projid][k])
                outlines.append(thisline)

    outname = 'projid_description'
    for p in projids:
        outname += '_{}'.format(p)
    outname += '.txt'

    outpath = os.path.join(outdir, outname)
    with open(outpath, mode='w') as f:
        f.writelines([o+'\n' for o in outlines])
    print('made {}'.format(outpath))




if __name__=="__main__":

    ##########################################
    # USAGE:
    # select your option from the boolean ones below. input your projids and
    # cam/ccd pairs.

    scp_projid_pickle_parameters = 1 # scp the projid pickle descriptions to local
    scp_rms_files = 1 # scp the RMS percentile csv files to local
    scp_knownplanet_files = 1

    find_what_projids_meant = 1 # for given projids, make text file summary
    analyze_rms_files = 0 # for given projids, make plot comparison RMS percentiles
    analyze_knownplanet_files = 1 # for projids, check SNR of known planets

    # expmtstr = 'kernelvarybkgnd_2-2_'
    # projids = [1090,1093,1096,1099]
    # cam, ccd = 2, 2

    # expmtstr = 'kernelvarydeltaorder_2-2_'
    # projids = [1102,1105,1108,1111]
    # cam, ccd = 2, 2

    # expmtstr = 'kernelvaryhalfsize_2-2_'
    # projids = [1111,1114,1117]
    # cam, ccd = 2, 2

    # expmtstr = 'kernelvaryidentityorder_2-2_'
    # projids = [1120,1123,1126]
    # cam, ccd = 2, 2

    # nb. this one is only for "knownplanet" plots (not rms ones)
    # expmtstr = 'kernelsALL190130_2-2_'
    # projids = list(np.arange(1090,1129,3))
    # cam, ccd = 2, 2

    # expmtstr = 'kernelsnobkgd_2-2_'
    # projids = [1093,1154,1155,1156,1157,1158,1159,1160,1161]
    # cam, ccd = 2, 2

    # expmtstr = 'kernelsALL_2-2_'
    # projids = (
    #     list(np.arange(1090,1129,3)) +
    #     [1154,1155,1156,1157,1158,1159,1160,1161]
    # )
    # cam, ccd = 2, 2

    # 20190303 cam2ccd2!
    # # varying half-size...
    # expmtstr = 'kernelvaryhalfsize_2-2_'
    # projids = np.arange(1194,1199,1)
    # cam, ccd = 2, 2

    # # varying spatialorder...
    # expmtstr = 'kernelvaryspatialorder_2-2_'
    # projids = [1194, 1199, 1200, 1201, 1202]
    # cam, ccd = 2, 2

    # # varying identityorder...
    # expmtstr = 'kernelvaryidentityorder_2-2_'
    # projids = [1194, 1203, 1204, 1205, 1206]
    # cam, ccd = 2, 2

    # varying ALL
    expmtstr = 'kernelvaryALL_2-2_'
    projids = np.arange(1194,1207,1)
    cam, ccd = 2, 2

    # 20190303 cam1ccd2
    # # varying half-size...
    # expmtstr = 'kernelvaryhalfsize_1-2_'
    # projids = np.arange(1207,1212,1)
    # cam, ccd = 1, 2

    # # varying spatialorder...
    # expmtstr = 'kernelvaryspatialorder_1-2_'
    # projids = [1207, 1212, 1213, 1214, 1215]
    # cam, ccd = 1, 2

    # # varying identityorder...
    # expmtstr = 'kernelvaryidentityorder_1-2_'
    # projids = [1207, 1216, 1217, 1218, 1219]
    # cam, ccd = 1, 2

    # # varying ALL
    # expmtstr = 'kernelvaryALL_1-2_'
    # projids = np.arange(1207,1220,1)
    # cam, ccd = 1, 2
    #FIXME from here

    # expmtstr = 'kernelvarypreliminary_1-2_'
    # #projids = [1162,1163,1164,1165,1166]
    # projids = [1162,1163,1164,1165]
    # cam, ccd = 1, 2

    # expmtstr = 'kernelvary20190206_1-2_'
    # projids = np.arange(1162,1183+1,1)
    # cam, ccd = 1, 2

    #expmtstr = 'kernelvarybkgnd_4-4_'
    #projids = [1091,1094,1097,1100]
    #cam, ccd = 4, 4

    ##########################################
    # things below here shouldn't need to be changed...

    camccdstr = 'ISP_{}-{}'.format(cam, ccd)

    if scp_projid_pickle_parameters:

        indir = "/nfs/phtess1/ar1/TESS/FFI/PROJ/REDUCTION_PARAM_PICKLES"
        outdir = "../data/reduction_param_pickles"
        scp_projid_pickles(indir, outdir)

    if scp_rms_files:

        for projid in projids:

            run_id = "{}-{}".format(camccdstr, projid)
            outdir = (
                '../results/optimizing_pipeline_parameters/{}_stats'.
                format(run_id)
            )
            statsdir = (
                '/nfs/phtess1/ar1/TESS/FFI/LC/FULL/s0002/{}/stats_files'.
                format(run_id)
            )

            scp_rms_percentile_files(statsdir, outdir)

    if scp_knownplanet_files:

        for projid in projids:

            run_id = "{}-{}".format(camccdstr, projid)
            outdir = (
                '../results/optimizing_pipeline_parameters/{}_knownplanets'.
                format(run_id)
            )
            statsdir = (
                '/nfs/phtess1/ar1/TESS/FFI/LC/FULL/s0002/{}/stats_files/'.
                format(run_id)
            )

            scp_knownplanets(statsdir, outdir)

    # scp tools above
    ##########################################
    # analysis tools below

    if analyze_rms_files:

        # [2,25,50,75,98] for percentiles
        for apstr in ['TF1','TF2','TF3']:
            d = get_rms_file_dict(projids, camccdstr, ap=apstr)

            plot_rms_comparison(d, projids, camccdstr, expmtstr, apstr=apstr,
                                overplot_theory=False,
                                descriptionkey='kernelspec')

            plot_rms_comparison(d, projids, camccdstr, expmtstr, apstr=apstr,
                                overplot_theory=True,
                                descriptionkey='kernelspec')


    if find_what_projids_meant:

        desiredkeys = ['aperturelist', 'camnum', 'ccdnum', 'catalog_faintrmag',
                       'field', 'fitsdir', 'kernelspec']

        pkldir = "../data/reduction_param_pickles"
        outdir = "../results/optimizing_pipeline_parameters/"
        projid_pickle_to_txt(projids, pkldir, outdir, desiredkeys=desiredkeys)


    if analyze_knownplanet_files:

        apd = {}
        apstrs = ['TFA1','TFA2','TFA3']
        for apstr in apstrs:

            d = get_knownplanet_file_dict(projids, camccdstr, ap=apstr)

            plot_knownplanet_comparison(d, projids, camccdstr, expmtstr,
                                        apstr=apstr,
                                        descriptionkey='kernelspec')

            apd[apstr] = d

        # get and plot the best aperture SNR
        tois = []
        for apstr in apstrs:
            for projid in projids:
                tois.append(list(apd[apstr][projid].keys()))
        utois = np.unique(np.concatenate(
            list(np.array(t) for t in np.atleast_1d(tois))).ravel())

        d = {}
        for projid in projids:
            d[projid] = {}
            for toi in utois:
                try:
                    ap_snrs = np.array(
                        [apd[apstr][projid][toi]['trapz_snr'] for apstr in apstrs]
                    )
                except KeyError:
                    ap_snrs = np.ones_like(apstrs).astype(float)
                    ap_snrs *= np.nan

                try:
                    argmaxsnr = np.nanargmax(ap_snrs)
                except ValueError:
                    continue
                ap_max = apstrs[argmaxsnr]

                d[projid][toi] = apd[ap_max][projid][toi]

        plot_knownplanet_comparison(d, projids, camccdstr, expmtstr,
                                    apstr='BESTAP',
                                    descriptionkey='kernelspec')

        plot_knownplanet_comparison(d, projids, camccdstr, expmtstr,
                                    apstr='BESTAP',
                                    descriptionkey='kernelspec',
                                    divbymaxsnr=True, ylim=(0,1.1))

