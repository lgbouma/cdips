from glob import glob
import pandas as pd, numpy as np
import os

datadir = '/home/luke/Dropbox/proj/tessmaps/results'

def main():

    d = {}
    for snum in range(0,12+1):
        k13str = 'gd_kharchenko13_sector{}.csv'.format(snum)
        k13path = os.path.join(datadir,k13str)
        d[snum] = pd.read_csv(k13path, sep=',')

    df_1to5 = pd.concat(
        (d[0],d[1],d[2],d[3],d[4])
    )

    df_1to5 = df_1to5.drop_duplicates().reset_index().drop('index',axis=1)
    print('distance < 2kpc, clusters w/ >30 stars, age<10Gyr. Kharchenko clusters. (No moving groups)')
    print('exposure times calculated thru tessmaps\n')
    print('sector 1 thru 5 inclusive.')
    print('\t{} stars, in {} unique clusters.'.
          format(df_1to5.sum()['N1sr2'], len(df_1to5)))

    for snum in range(6, 13+1):

        print('sector {}.'.format(snum))
        print('\t{} stars, in {} unique clusters'.
              format(d[snum-1].sum()['N1sr2'], len(d[snum-1])))

    df_all = pd.concat( (d[i] for i in range(0,12+1) ) )
    df_all = df_all.drop_duplicates().reset_index().drop('index',axis=1)
    print('\ntotal unique stars over first year: {}'.format(df_all.sum()['N1sr2']))


if __name__=='__main__':
    main()
