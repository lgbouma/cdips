"""
skim_cream(sectornum=6, tls_sde_cut=12, fap_cut=1e-30, SR_TFA=args.tfasr):

    once do_initial_period_finding is done, will
    plot_initial_period_finding_results -- overall distribution stats of SDE,
    FAP, vs periods.

    then it will make cuts, and re-iterate periodogram creation but save the
    full results, and make a 2-lsp checkplot.

usage: skim_cream.py [-h] [--tfasr] [--no-tfasr]

e.g., python skim_cream.py -tfasr &> ../logs/skim_cream_sector6_tfasr.log &
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import os, argparse
from glob import glob
from copy import deepcopy

from cdips.lcproc import period_find_for_cluster as pfc

def plot_initial_period_finding_results(
    df,
    resultsdir):

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
    foo = df[~pd.isnull(df['ls_fap']) & np.isfinite(np.array(df['ls_fap']))]
    log10fap = np.log10(foo['ls_fap'])
    plt.hist(log10fap[np.isfinite(log10fap)], bins=50, density=True)
    plt.xlabel('log10_LS_FAP')
    plt.ylabel('fraction of total')
    plt.title('total: {} stars'.format(n_tot))
    plt.yscale('log')
    outpath = os.path.join(resultsdir, 'histogram_log10_ls_fap.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print('made {}'.format(outpath))

    f,ax = plt.subplots(figsize=(4,3))
    ax.scatter(df['tls_period'], df['tls_sde'], c='k', alpha=1, s=0.2,
               rasterized=True, linewidths=0)

    ax.scatter(df['tls_period'], df['limit'], c='C1', alpha=1, rasterized=True,
               linewidths=0, zorder=2, s=0.2)

    ax.set_title('N_above: {}. N_below: {}'.
                 format(len(df[df['tls_sde']>df['limit']]),
                        len(df[df['tls_sde']<df['limit']])), size='small')
    ax.set_xlabel('tls_period')
    ax.set_ylabel('tls_sde')
    ax.set_xscale('log')
    outpath = os.path.join(resultsdir, 'scatter_tls_sde_vs_tls_period.png')
    f.savefig(outpath, dpi=300, bbox_inches='tight')
    print('made {}'.format(outpath))

    f,ax = plt.subplots(figsize=(4,3))
    ax.scatter(df['ls_period'], np.log10(df['ls_fap']), c='k', alpha=1, s=0.2,
               rasterized=True, linewidths=0)
    ax.set_xlabel('ls_period')
    ax.set_ylabel('log10(ls_fap)')
    ax.set_xscale('log')
    outpath = os.path.join(resultsdir, 'scatter_log10_ls_fap_vs_ls_period.png')
    f.savefig(outpath, dpi=300, bbox_inches='tight')
    print('made {}'.format(outpath))

    # save the relevant numbers to a text file too
    N_above = len(df[df['tls_sde']>df['limit']])
    N_below = len(df[df['tls_sde']<df['limit']])
    outpath = os.path.join(resultsdir, 'n_above_and_below_limit.txt')
    with open(outpath, mode='w') as f:
        f.writelines(
            'N_above={}|N_below={}'.format(N_above, N_below)
        )


def skim_cream(sectornum=6, tls_sde_cut=12, fap_cut=1e-30, SR_TFA=None):

    resultsdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
        'cdips_lc_periodfinding/'
        'sector-{}'.format(sectornum)
    )
    initpfresultspath = (
        os.path.join(resultsdir, 'initial_period_finding_results.csv')
    )
    if SR_TFA:
        resultsdir = os.path.join(
            os.path.dirname(resultsdir),
            'sector-{}_TFA_SR'.format(sectornum)
        )
        if not os.path.exists(resultsdir):
            os.mkdir(resultsdir)

    df = pd.read_csv(initpfresultspath)

    plot_initial_period_finding_results(df, resultsdir)

    if not SR_TFA:
        lcdir = (
            '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}'.
            format(sectornum)
        )
        lcpaths = glob(os.path.join(lcdir, 'cam?_ccd?', '*_llc.fits'))
    else:
        lcdir = (
            '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}_TFA_SR'.
            format(sectornum)
        )
        lcpaths = glob(os.path.join(lcdir, '*_llc.fits'))

    outdir = os.path.join(resultsdir, 'sde_gt_{}'.format(tls_sde_cut))
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    sel = (df['tls_sde'] > tls_sde_cut) & ~pd.isnull(df['tls_sde'])

    fluxap = 'TFA2' if not SR_TFA else 'TFASR2'

    for ix, row in df[sel].sort_values(by='tls_sde',ascending=False).iterrows():
        source_id = str(int(row['source_id']))
        matching = [l for l in lcpaths if source_id in l]
        if len(matching) != 1:
            print('ERR! expected 1 matching lcpath')
            continue
            #import IPython; IPython.embed()
        lcpath = matching[0]
        pfc.do_period_finding_fitslc(lcpath, fluxap=fluxap, outdir=outdir)


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
        lcpath = matching[0]
        pfc.do_period_finding_fitslc(lcpath, fluxap=fluxap, outdir=outdir)


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description=('Make 2-periodogram checkplots, before/after TFA_SR')
    )
    parser.add_argument(
        '--tfasr', dest='tfasr', action='store_true',
        help=('have you run TFA signal reconstruction & do u want that glob?')
    )
    parser.add_argument(
        '--no-tfasr', dest='tfasr', action='store_false',
        help=('before running TFA signal reconstrn, if u want to see plots')
    )
    parser.set_defaults(tfasr=False)
    args = parser.parse_args()

    skim_cream(sectornum=6, tls_sde_cut=12, fap_cut=1e-30, SR_TFA=args.tfasr)
