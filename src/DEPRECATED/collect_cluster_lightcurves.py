def make_cg18_cut():
    """for speed, work with only DR2 source id's and cluster names."""

    datadir = '../data/cluster_data/'
    tabpath = os.path.join(
        datadir,'CantatGaudin_2018_table2_membership_info.vot'
    )
    vot = parse(tabpath)
    tab = vot.get_first_table().to_table()

    outdf = pd.DataFrame({'source':tab['Source'],'cluster':tab['Cluster']})

    outpath = (
        '../data/cluster_data/CantatGaudin_2018_table2_cut_only_source_cluster.csv'
    )

    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))



def kharchenko13_to_gaiadr2_xmatch(df, cluster_name):
    """
    Given a dataframe with stars in a cluster (from Kharchenko+13), match
    Kharchenko stars to Gaia-DR2 through the 2MASS-best-neighbor table.

    Return a dataframe that includes the Gaia DR2 id, the angular distance, and
    the astrometric parameters.
    """

    # we dropped the u for a reason!
    tmbn = Gaia.load_table('gaiadr2.tmass_best_neighbour')

    tmassid = nparr(df['2MASS']).astype(int)

    jobstr = (
        'select source_id,original_ext_source_id,angular_distance,'
        'gaia_astrometric_params,tmass_oid from gaiadr2.tmass_best_neighbour'
        'where tmass_oid in'
    )

    outdir = '../data/cluster_data/MWSC_Gaia_xmatch/'
    outfile = os.path.join(outdir,'{}_xmatch.xml.gz'.format(cluster_name))

    job = Gaia.launch_job(
        jobstr, dump_to_file=True, output_file=outfile
    )

    import IPython; IPython.embed()

    pass


def xmatch_kharchenko13_gaiadr2_2mass(homedir='/home/luke/'):
    """
    runs crossmatch on 2mass ID to the votables that you manually upload to the
    gaia website.
    """

    Gaia.login(credentials_file=os.path.join(homedir, '.gaia_credentials'))

    tables = Gaia.load_tables(only_names=True, include_shared_tables=True)

    for t in tables:
        print(t.get_qualified_name())

    jobstr = (
	'select tmass_oid, source_id, original_ext_source_id,'
        'angular_distance,gaia_astrometric_params from '
        'gaiadr2.tmass_best_neighbour where '
        'tmass_oid in (select col2mass from user_lbouma.blanco_1)'
    )

    outdir = '../data/cluster_data/MWSC_Gaia_xmatch/'
    outfile = os.path.join(outdir,'{}_xmatch.xml.gz'.format(cluster_name))

    if not os.path.exists(outfile):
        job = Gaia.launch_job(
            jobstr, dump_to_file=True, output_file=outfile
        )
    else:
        pass

    Gaia.logout()

    vot = parse(outfile)
    tab = vot.get_first_table().to_table()

    lcbasenames = [str(bn)+'_llc.fits' for bn in nparr(tab['source_id'])]

    return lcbasenames




def get_k13_stellar_data_given_clustername(cluster_name, p_0=61):
    '''
    This function reads the Kharchenko+ 2013 "stars/*" tables for each cluster,
    and selects the stars that are "most probably cluster members, that is,
    stars with kinematic and photometric membership probabilities >61%".

    (See Kharchenko+ 2012 for definitions of these probabilities)

    args:
        cluster_name (str): matching the format from
        ../data/cluster_data/MWSC_1sigma_members/

        p_0: probability for inclusion. See Eqs in Kharchenko+ 2012. p_0=61 (not
        sure why not 68.27) is 1 sigma members by kinematic and photometric
        membership probability, also accounting for spatial step function and
        proximity within stated cluster radius.

    Returns:
        dataframe with stellar data for cluster members:
        ['RAhour', 'DEdeg', 'Bmag', 'Vmag', 'Jmag', 'Hmag', 'Ksmag', 'e_Jmag',
        'e_Hmag', 'e_Ksmag', 'pmRA', 'pmDE', 'e_pm', 'RV', 'e_RV', 'Qflg',
        'Rflg', 'Bflg', '2MASS', 'ASCC', 'SpType', 'Rcl', 'Ps', 'Pkin', 'PJKs',
        'PJH', 'MWSC', 'RAdeg']
    '''

    datadir = '../data/cluster_data/MWSC_1sigma_members/'
    clusterpath = glob(os.path.join(datadir, '*{}*'.format(cluster_name)))
    if len(clusterpath) != 1:
        raise AssertionError('cluster_name did not give unique star data file')
    clusterpath = clusterpath[0]

    tab = Table.read(clusterpath, format='ascii.ecsv')

    # Select 1-sigma cluster members by photometry & kinematics. (Note: this
    # was already done to construct the "member lists". This is just repeating
    # Kharchenko's cut for caution.
    #
    # From Kharchenko+ 2012, require that:
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
    inds &= ( ((tab['Ksmag']>7) & (tab['Qflg']=='AAA')) | (tab['Ksmag']<7))
    pm_inds = ((tab['e_pm'] < 10) & (tab['DEdeg']>-30)) | \
              ((tab['e_pm'] < 15) & (tab['DEdeg']<=-30))
    inds &= pm_inds

    members = tab[inds]
    mdf = members.to_pandas()

    return mdf, tab


def get_cg18_sources_given_clustername(cluster_name):
    """
    cluster_name
    """

    datadir = '../data/cluster_data/'
    csvpath = os.path.join(
        datadir,'CantatGaudin_2018_table2_cut_only_source_cluster.csv'
    )

    df = pd.read_csv(csvpath)

    # cluster names are strings of byte objects of strings. lol. make them
    # strings.
    df['cluster'] = list(map(
        lambda x: x.replace('b\'','').replace('\'',''), df['cluster']
    ))

    if cluster_name not in np.unique(df['cluster']):
        raise AssertionError('got cluster name not in cluster list')

    clustersourceids = df.loc[df['cluster']==cluster_name, 'source']

    return np.array(clustersourceids)




def write_k13_tab_to_vot(_tab, cluster_name):

    # save to .vot, easiest format for gaia archive
    _vot = from_table(_tab)
    outdir = '../data/cluster_data/MWSC_1sigma_members_votables/'
    outpath = os.path.join(outdir, cluster_name+'.vot')
    writeto(_vot, outpath)
    print('wrote {}'.format(outpath))



def old_main():
    # first, test this process with one cluster.

    cluster_name = 'Blanco_1'

    use_cg18 = 1
    use_k13 = 0 # deprecated
    runname = 'ISP_1-2-1186' # external lcdir

    if use_k13:
        _df, _tab = get_k13_stellar_data_given_clustername(cluster_name, p_0=61)
        write_k13_tab_to_vot(_tab, cluster_name)
        lcbasenames = xmatch_kharchenko13_gaiadr2_2mass()
        df = kharchenko13_to_gaiadr2_xmatch(_df, cluster_name)

    elif use_cg18:

        datapath = (
            '../data/cluster_data/'
            'CantatGaudin_2018_table2_cut_only_source_cluster.csv'
        )
        if not os.path.exists(datapath):
            make_cg18_cut()

        sourceids = get_cg18_sources_given_clustername(cluster_name)

    lcbasenames = [str(s)+'_llc.fits' for s in sourceids]
    lcdir = '/nfs/phtess1/ar1/TESS/FFI/LC/FULL/s0002/{}'.format(runname)
    outdir = (
        '../data/cluster_data/lightcurves/{}_{}'.
        format(cluster_name, runname)
    )
    scp_lightcurves(lcbasenames, lcdir=lcdir, outdir=outdir)

    n_from_cg18 = len(sourceids)
    n_lcs_from_me = len(glob(os.path.join(outdir, '*_llc.fits')))

    import IPython; IPython.embed()

