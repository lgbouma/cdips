"""
Sometimes, you just want to see what's been made for a specific cluster.
"""
import os
from glob import glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt, multiprocessing as mp
from shutil import copyfile
from os.path import join
from astropy.io import fits

from cdips.utils.lcutils import get_lc_data
from cdips.utils.catalogs import get_cdips_pub_catalog
from cdips.paths import RESULTSDIR, DATADIR

def _get_viable_cluster_lcs(df):

    unwrapref = []
    for u in np.unique(df.reference):
        for this in u.split(','):
            unwrapref.append(this)

    # Actual unique cluster membership sources:
    # 'Bell_2017_32Ori', 'CantatGaudin_2018', 'CantatGaudin_2019_velaOB2',
    # 'Dias2014', 'Gagne_2018_BANYAN_XI', 'Gagne_2018_BANYAN_XII',
    # 'Gagne_2018_BANYAN_XIII', 'GaiaCollaboration2018_tab1a',
    # 'GaiaCollaboration2018_tab1b', 'Kharchenko2013', 'Kounkel_2018_Ori',
    # 'Kounkel_2019', 'Kraus_2014_TucHor', 'Oh_2017_clustering',
    # 'Rizzuto_2011_ScoOB2', 'Roser_2011_Hyades', 'Velez_2018_scoOB2',
    # 'Zari_2018_PMS', 'Zari_2018_UMS'
    sdf = df[
        (~(df.reference == 'Dias2014'))
        &
        (~(df.reference == 'Kharchenko2013'))
        &
        (~(df.reference == 'Zari_2018_UMS'))
        &
        (~(df.reference == 'Zari_2018_PMS'))
        &
        (~(pd.isnull(df.cluster)))
    ]

    return sdf


def _apply_cluster_selection_function(sdf, clusterkey):

    allowed_keys = [
        "IC_2602",
        "NGC_2516",
        "CrA",
        "kc19group_113",
        "Hyades",
        "VelaOB2",
        "ScoOB2",
        "Orion"
    ]
    if clusterkey not in allowed_keys:
        raise NotImplementedError

    if clusterkey in ["IC_2602", "NGC_2516"]:

        sel = (
            (sdf.reference.str.contains('Kounkel_2019') & sdf.cluster.str.contains(clusterkey))
            |
            (sdf.reference.str.contains('GaiaCollaboration2018_tab1a') & sdf.cluster.str.contains(clusterkey.replace('_','')))
            |
            (sdf.reference.str.contains('CantatGaudin_2018') & sdf.cluster.str.contains(clusterkey))
        )

    elif clusterkey in ["CrA", "kc19group_113"]:

        sel = (
            (sdf.reference.str.contains('Kounkel_2019') & sdf.cluster.str.contains(clusterkey))
        )

    elif clusterkey == "Hyades":

        sel = (
            (sdf.reference.str.contains('Roser_2011_Hyades') & sdf.cluster.str.contains(clusterkey))
            |
            (sdf.reference.str.contains('Gagne') & sdf.cluster.str.contains("HYA"))
            |
            (sdf.reference.str.contains('GaiaCollaboration2018_tab1a') & sdf.cluster.str.contains(clusterkey))
            |
            (sdf.reference.str.contains('Kounkel_2019') & sdf.cluster.str.contains(clusterkey))
        )

    elif clusterkey == "Orion":

        sel = (
            (sdf.reference.str.contains('Kounkel_2018_Ori'))
            |
            (sdf.reference.str.contains('Kounkel_2019') & sdf.cluster.str.contains(clusterkey))
        )

    elif clusterkey == 'VelaOB2':

        sel = sdf.cluster.str.contains('cg19velaOB2')

    elif clusterkey == 'ScoOB2':

        sel = (
            (sdf.reference.str.contains('Velez_2018') & sdf.cluster.str.contains(clusterkey))
            |
            (sdf.reference.str.contains('Kounkel_2019') & sdf.cluster.str.contains('Upper_Sco'))
            |
            (sdf.reference.str.contains('Gagne') & sdf.cluster.str.contains("LCC"))
            |
            (sdf.reference.str.contains('Gagne') & sdf.cluster.str.contains("UCL"))
            |
            (sdf.reference.str.contains('Gagne') & sdf.cluster.str.contains("USCO"))
            |
            (sdf.reference.str.contains('Rizzuto') & sdf.cluster.str.contains("ScoOB2"))
        )

    return sdf[sel]


def collect_lcs(cdf, outdir):

    lcpaths = list(cdf.path)

    LCDIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS'

    outlcdir = join(outdir, 'lcs')
    if not os.path.exists(outlcdir):
        os.mkdir(outlcdir)

    for lcpath in lcpaths:

            src = join(LCDIR, lcpath)
            dst = join(outlcdir, os.path.basename(lcpath))

            if not os.path.exists(dst):
                copyfile(src,dst)
                print(f'Made {dst}')
            else:
                print(f'Found {dst}')


    return glob(join(outlcdir, 'hlsp*fits'))


def make_quicklook(task):

    lcpath = task[0]
    outdir = task[1]
    BpmRp = task[2]
    reference = task[3]

    outpath = join(
        outdir,
        os.path.basename(lcpath).replace('.fits','_quicklook.png')
    )
    if os.path.exists(outpath):
        print(f'Found {outpath}')
        return


    d = get_lc_data(lcpath, mag_aperture='IRM1', tfa_aperture='PCA1')

    source_id, time, mag, xcc, ycc, ra, dec, tmag, pca_mag = d

    plt.close('all')
    f,axs = plt.subplots(nrows=2, figsize=(14,5), sharex=True)
    axs[0].scatter(
        time, mag, c='k', marker='.'
    )
    axs[1].scatter(
        time, pca_mag, c='k', marker='.'
    )
    axs[1].set_xlabel('BJDTDB', fontsize='x-small')
    axs[0].set_ylabel('IRM1', fontsize='x-small')
    axs[1].set_ylabel('PCA1', fontsize='x-small')
    for a in axs:
        a.set_ylim(a.get_ylim()[::-1])

    if isinstance(BpmRp, float) and isinstance(reference, str):
        titlestr = f'{source_id}\nT={tmag:.1f}, Bp-Rp={BpmRp:.1f}, ref={reference[:25]:s}'
    else:
        titlestr = f'{source_id}: T={tmag:.1f}'
    axs[0].set_title(titlestr, fontsize='x-small')

    f.savefig(outpath, dpi=100, bbox_inches='tight')
    print(f'Made {outpath}')


def make_source_list(k, ver=0.6):

    clusterkey = k

    outdir = join(DATADIR, 'cluster_data', 'cdips_catalog_split')
    outname = f'OC_MG_FINAL_v{ver}_publishable_CUT_{k}.csv'
    outpath = os.path.join(outdir, outname)

    if not os.path.exists(outpath):
        df = get_cdips_pub_catalog(ver=ver)
        sdf = _get_viable_cluster_lcs(df)
        cdf = _apply_cluster_selection_function(sdf, clusterkey)
        cdf.to_csv(outpath, index=False)
        print(f'Made {outpath}')
    else:
        print(f'Found {outpath}')




def run_quicklook(k, parallel=1):

    clusterkey = k

    # 20200812: S1-S13, southern skies. CDIPS DR4.
    metadata_path = join(
        RESULTSDIR, 'lc_metadata', 'source_ids_to_cluster_merge_20200812.csv'
    )

    outdir = join(RESULTSDIR, 'specific_clusters', clusterkey)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    quicklookdir = join(outdir, 'quicklooks')
    if not os.path.exists(quicklookdir):
        os.mkdir(quicklookdir)

    df = pd.read_csv(metadata_path, sep=';')

    sdf = _get_viable_cluster_lcs(df)

    cdf = _apply_cluster_selection_function(sdf, clusterkey)

    lcpaths = collect_lcs(cdf, outdir)

    BpmRp = np.array(cdf.phot_bp_mean_mag - cdf.phot_rp_mean_mag)
    refs = list(cdf.reference)

    if not parallel:
        # ~40 quicklook plots per second in serial
        for lcpath, bpmrp, r in zip(lcpaths, BpmRp, refs):
            task = (lcpath, quicklookdir, bpmrp, r)
            make_quicklook(task)

    else:
        tasks = [(lcpath, quicklookdir, bpmrp, r)
                 for lcpath, bpmrp, r in
                 zip(lcpaths, BpmRp, refs)]

        # fire up the pool of workers
        nworkers = mp.cpu_count()
        pool = mp.Pool(nworkers, maxtasksperchild=1000)
        results = pool.map(make_quicklook, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()


def main():

    DO_QUICKLOOK = 0 # run for the Online TESS Science Meeting in Aug 2020
    DO_SOURCES_ONLY = 1 # for individual clusters

    if DO_SOURCES_ONLY:
        allowed_keys = [
            "IC_2602", "CrA", "kc19group_113", "Hyades", "VelaOB2", "ScoOB2",
            "Orion", "NGC_2516"
        ]
        for k in allowed_keys:
            make_source_list(k)


    if DO_QUICKLOOK:
        allowed_keys = [
            "IC_2602", "NGC_2516", "CrA", "kc19group_113", "Hyades",
            "VelaOB2", "ScoOB2", "Orion"
        ]
        for k in allowed_keys:
            run_quicklook(k, parallel=1)


if __name__ == "__main__":
    main()
