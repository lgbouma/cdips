import os
from glob import glob
import pandas as pd, numpy as np

sector=7

if sector==6:
    date = '20190616'
elif sector==7:
    date = '20190618'

csvpath = (
    '/home/luke/Dropbox/proj/cdips/results/vetting_classifications/'
    '{}_LGB_sector{}_classifications_PC_cut.csv'.format(date,sector)
)

df = pd.read_csv(csvpath)

for n in np.array(df['Name']):

    pdfname = 'vet_'+n+'_llc.pdf'

    pdfdir = '/home/luke/local/cdips/vetting/sector-{}/pdfs/'.format(sector)

    pdfpath = os.path.join(pdfdir, pdfname)

    if os.path.exists(pdfpath):

        os.remove(pdfpath)

        print('removed {}'.format(pdfpath))

    else:

        print('didnt find & skipped {}'.format(pdfpath))
