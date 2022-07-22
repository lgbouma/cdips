"""
Contents:

Visualize Cutis+2020 rotation periods.
    plot_Prot_BpmRp0
Interpolation between (BP-RP)_0 and rotation period.
    PleiadesInterpModel
    PraesepeInterpModel
    NGC6811InterpModel
Convective turnover time estimation formulae
    given_Teff_get_tconv_CS11
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from cdips.paths import DATADIR
from os.path import join
from scipy.interpolate import interp1d

def plot_Prot_BpmRp0(outdir='.', ylim=None):

    reference_clusters = ['Pleiades', 'Praesepe', 'NGC 6811']

    scale = 'linear'
    Prot_source = 'Curtis20'

    BPMRPINTERPMODEL = {
        'Pleiades': PleiadesInterpModel,
        'Praesepe': PraesepeInterpModel,
        'NGC 6811': NGC6811InterpModel
    }

    plt.close('all')
    f, ax = plt.subplots(figsize=(8,6))

    for ix, c in enumerate(reference_clusters):

        cdf = _get_Curtis20_data(c.lower().replace(' ','_'))

        # data
        ax.scatter(cdf['(BP-RP)0'], cdf.Prot, s=0.5, c=f'C{ix}', label=c)

        InterpModel = BPMRPINTERPMODEL[c]

        _bpmrp0 = np.linspace(0.47, 3.30, 500)
        _prot = InterpModel(_bpmrp0)
        ax.plot(_bpmrp0, _prot, lw=1, c=f'C{ix}', alpha=0.5)

    ax.legend(fontsize='x-small', loc='best')
    ax.set_xlabel('$(G_\mathrm{BP} - G_\mathrm{RP})_0$ [mag]')
    ax.set_ylabel(f'{Prot_source} '+'P$_\mathrm{rot}$ [day]')
    if isinstance(ylim, list):
        ax.set_ylim(ylim)
    ax.set_xscale('linear'); ax.set_yscale(scale)

    s = ''
    if isinstance(ylim, list):
        s += 'ylim'+'_'.join(np.array(ylim).astype(str))

    outpath = join(
        outdir, f'BpmRp0_vs_{Prot_source}_prot_{scale}.png'
    )

    f.savefig(outpath, dpi=600, bbox_inches='tight')



def _get_Curtis20_data(cluster):
    """
    Given cluster name, return rotation periods, colors, and a few other pieces
    of information from Curtis et al., 2020.

    Args: cluster (str): one of the cluster keys below.
    """
    ALIASDICT = {
        'pleiades': 'Pleiades', # 120 Myr
        'praesepe': 'Praesepe', # 670 Myr
        'ngc_6811': 'NGC 6811', # 1 Gyr
        'ngc_752': 'NGC 752', # 1.4 Gyr
        'ngc_6819': 'NGC 6819', # 2.5 Gyr
        'ruprecht_147': 'Ruprecth 147' # 2.7 Gyr
    }

    assert cluster in list(ALIASDICT.keys()), f'cluster={cluster} N/A.'

    t = Table.read(
        join(DATADIR, 'LITERATURE_DATA', 'Curtis_2020_apjabbf58t5_mrt.txt'),
        format='cds'
    )

    df = t[t['Cluster'] == ALIASDICT[cluster]].to_pandas()

    return df


def PleiadesInterpModel(BpmRp0, bounds_error=True):

    df = pd.read_csv(
        join(DATADIR, "LITERATURE_DATA",
             'Curtis_2020_bpmrp0_prot_pleaides_upper.csv')
    )
    df = df.sort_values(by='bpmrp0')

    fn_BpmRp0_t0_prot = interp1d(
        df.bpmrp0, df.Prot, kind='quadratic',
        bounds_error=bounds_error, fill_value=np.nan
    )

    Protmod = fn_BpmRp0_t0_prot(BpmRp0)

    return Protmod


def PraesepeInterpModel(BpmRp0, bounds_error=True):

    df = pd.read_csv(
        join(DATADIR, "LITERATURE_DATA",
             'Curtis_2020_bpmrp0_prot_praesepe_upper.csv')
    )
    df = df.sort_values(by='bpmrp0')

    fn_BpmRp0_t0_prot = interp1d(
        df.bpmrp0, df.Prot, kind='quadratic',
        bounds_error=bounds_error, fill_value=np.nan
    )

    Protmod = fn_BpmRp0_t0_prot(BpmRp0)

    return Protmod


def NGC6811InterpModel(BpmRp0, bounds_error=True):

    df = pd.read_csv(
        join(DATADIR, "LITERATURE_DATA",
             'Curtis_2020_bpmrp0_prot_ngc6811_upper.csv')
    )
    df = df.sort_values(by='bpmrp0')

    fn_BpmRp0_t0_prot = interp1d(
        df.bpmrp0, df.Prot, kind='quadratic',
        bounds_error=bounds_error, fill_value=np.nan
    )

    Protmod = fn_BpmRp0_t0_prot(BpmRp0)

    return Protmod


def given_Teff_get_tconv_CS11(teff):
    """
    Eq 36 from Cranmer and Saar 2011

    Given teff, returns t_convective-turnover in units of days, for Teffs from
    3300K to 7000K.
    """
    arg = -(teff/1952.5) - (teff/6250)**18
    return 314.24 * np.exp(arg) + 0.002

if __name__ == "__main__":
    plot_Prot_BpmRp0()

