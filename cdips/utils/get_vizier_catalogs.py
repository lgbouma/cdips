from astroquery.vizier import Vizier

def get_k13_index():
    #
    # the ~3784 row table
    #
    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/A+A/558/A53')
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    k13_index = catalogs[1].to_pandas()
    for c in k13_index.columns:
        if c != 'N':
            k13_index[c] = k13_index[c].str.decode('utf-8')

    return k13_index

def get_k13_param_table():
    #
    # the ~3000 row table with determined parameters
    #
    cols = ['map', 'cmd', 'stars', 'Name', 'MWSC', 'Type', 'RAJ2000',
            'DEJ2000', 'r0', 'r1', 'r2', 'pmRA', 'pmDE', 'RV', 'e_RV', 'o_RV',
            'd', 'E_B-V_', 'logt', 'N1sr2', 'rc', 'rt', 'k', 'SType',
            '__Fe_H_', 'Simbad']

    v = Vizier(columns=cols)

    v.ROW_LIMIT = -1
    catalog_list = v.find_catalogs('J/A+A/558/A53')
    catalogs = v.get_catalogs(catalog_list.keys())
    k13 = catalogs[0].to_pandas()

    k13['Name'] = k13['Name'].str.decode('utf-8')
    return k13

def get_soubiran_19_rv_table():

    cols = ['ID', 'ID2', 'RA_ICRS', 'DE_ICRS', 'dmode', 'Nmemb', 'Nsele', 'RV',
            'e_RV', 's_RV', 'X', 'e_X', 'Y', 'e_Y', 'Z', 'e_Z', 'U', 'e_U',
            'V', 'e_V', 'W', 'e_W', 'Vr', 'e_Vr', 'Vphi', 'e_Vphi', 'Vz',
            'e_Vz', 'Simbad']

    v = Vizier(columns=cols)

    v.ROW_LIMIT = -1
    catalog_list = v.find_catalogs('J/A+A/619/A155')
    catalogs = v.get_catalogs(catalog_list.keys())
    df = catalogs[0].to_pandas()

    df['ID'] = df['ID'].str.decode('utf-8')

    return df
