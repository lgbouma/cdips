import os
import matplotlib.pyplot as plt
from astropy.io import fits

def plot_simple_cdips_lc(
    lcpath,
    outdir,
    outname=None,
    vecnames='IRM1,PCA1,BGV'
):
    """
    Given a CDIPS light curve filepath, `lcpath`, write a three-panel plot of
    `vecnames` versus BJD_TDB.

    outdir is required.  If outname is None, a good name will be made.
    """
    assert isinstance(lcpath, str)
    assert isinstance(outdir, str)

    hdul = fits.open(lcpath)
    hdr = hdul[0].header
    d = hdul[1].data
    source_id = hdr['Gaia-ID']
    Tmag = hdr['TESSMAG']
    bpmrp = hdr['phot_bp_mean_Mag'] - hdr['phot_rp_mean_mag']
    sector = hdr['SECTOR']
    sectorstr = str(sector).zfill(4)

    if outname is None:
        outname = f"bpmrp{bpmrp:.2f}_GDR2_{source_id}_{sectorstr}.png"
    outpath = os.path.join(outdir, outname)

    if os.path.exists(outpath):
        print(f"Found {outpath}, skipping.")
        return 1

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(14,9))

    for ax, vecname in zip(axs, vecnames.split(',')):
        ax.scatter(
            d['TMID_BJD'], d[vecname], s=10, c='k', marker='.', linewidths=0
        )
        ax.set_ylabel(vecname)
        if vecname in ['IRM1', 'PCA1', 'TFA1']:
            ax.set_ylim(ax.get_ylim()[::-1])

    axs[-1].set_xlabel("TMID_BJD")

    titlestr = f"GDR2 {source_id}, {sectorstr}, T={Tmag:.1f}, BP-RP={bpmrp:.1f}"

    axs[0].set_title(titlestr)

    fig.savefig(
        outpath, bbox_inches='tight', dpi=300
    )

    plt.close("all")
    print(f"Wrote {outpath}")
    return 1
