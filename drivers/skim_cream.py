"""
given initial period finding results (just the best LSP and BLS peaks), check
their distributions.

then do some cuts, and re-iterate their periodograms but save the full results,
and perhaps make checkplots.
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import os
from glob import glob

from cdips.lcproc import period_find_for_cluster as pfc

def plot_initial_period_finding_results(
    df,
    resultsdir
):

    n_tot = len(df)

    plt.close('all')
    plt.figure(figsize=(4,3))
    plt.hist(df['tls_sde'], bins=50, density=True)
    plt.xlabel('tls_sde')
    plt.ylabel('fraction of total')
    plt.title('total: {} stars'.format(n_tot))
    plt.yscale('log')
    outpath = os.path.join(resultsdir, 'histogram_tls_sde.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print('made {}'.format(outpath))

    plt.close('all')
    plt.figure(figsize=(4,3))
    plt.hist(np.log10(df['ls_fap']), bins=50, density=True)
    plt.xlabel('log10_LS_FAP')
    plt.ylabel('fraction of total')
    plt.title('total: {} stars'.format(n_tot))
    plt.yscale('log')
    outpath = os.path.join(resultsdir, 'histogram_log10_ls_fap.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print('made {}'.format(outpath))

    f,ax = plt.subplots(figsize=(4,3))
    ax.scatter(df['tls_period'], df['tls_sde'], c='k', alpha=1, s=0.5,
               rasterized=True, linewidths=0)
    ax.set_xlabel('tls_period')
    ax.set_ylabel('tls_sde')
    outpath = os.path.join(resultsdir, 'scatter_tls_sde_vs_tls_period.png')
    f.savefig(outpath, dpi=300, bbox_inches='tight')
    print('made {}'.format(outpath))

    f,ax = plt.subplots(figsize=(4,3))
    ax.scatter(df['ls_period'], np.log10(df['ls_fap']), c='k', alpha=1, s=0.5,
               rasterized=True, linewidths=0)
    ax.set_xlabel('ls_period')
    ax.set_ylabel('log10(ls_fap)')
    ax.set_xscale('log')
    outpath = os.path.join(resultsdir, 'scatter_log10_ls_fap_vs_ls_period.png')
    f.savefig(outpath, dpi=300, bbox_inches='tight')
    print('made {}'.format(outpath))


def skim_cream(sectornum=6, tls_sde_cut=12, fap_cut=1e-30):

    resultsdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
        'cdips_lc_periodfinding/'
        'sector-{}/'.format(sectornum)
    )
    initpfresultspath = (
        os.path.join(resultsdir, 'initial_period_finding_results.csv')
    )

    df = pd.read_csv(initpfresultspath)

    plot_initial_period_finding_results(df, resultsdir)

    lcdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}'.
        format(sectornum)
    )
    lcpaths = glob(os.path.join(lcdir, 'cam?_ccd?', '*_llc.fits'))

    outdir = os.path.join(resultsdir, 'sde_gt_{}'.format(tls_sde_cut))
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    sel = (df['tls_sde'] > tls_sde_cut) & ~pd.isnull(df['tls_sde'])

    for ix, row in df[sel].sort_values(by='tls_sde',ascending=False).iterrows():
        source_id = str(int(row['source_id']))
        matching = [l for l in lcpaths if source_id in l]
        if len(matching) != 1:
            print('ERR! expected 1 matching lcpath')
            continue
            #import IPython; IPython.embed()
        lcpath = matching[0]
        pfc.do_period_finding_fitslc(lcpath, ap=2, outdir=outdir)


    sel = (df['ls_fap'] < fap_cut) & ~pd.isnull(df['ls_fap'])
    outdir = os.path.join(resultsdir, 'fap_lt_{}'.format(fap_cut))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for ix, row in df[sel].sort_values(by='ls_fap',ascending=True).iterrows():
        source_id = str(int(row['source_id']))
        matching = [l for l in lcpaths if source_id in l]
        if len(matching) != 1:
            print('ERR! expected 1 matching lcpath')
            continue
<<<<<<< HEAD
            #raise AssertionError('expected 1 matching lcpath')
=======
            import IPython; IPython.embed()
>>>>>>> 3ffbbc6db10d0324e9abc0903906cb925ef1d608
        lcpath = matching[0]
        pfc.do_period_finding_fitslc(lcpath, ap=2, outdir=outdir)


if __name__=="__main__":

    skim_cream(sectornum=6)
