"""
given a directory classified w/ TagSpaces tags, produce a CSV file with the
classifications

also produce CSV file with PC only classifications
"""
import os
from glob import glob
import pandas as pd

classified_dir = '/home/luke/Dropbox/proj/cdips/results/vetting_classifications/sector_6_pdfs_VETTED_20190606/'
outdir = '/home/luke/Dropbox/proj/cdips/results/vetting_classifications/'
outname = '20190611_LGB_sector6_classifications.csv'
outpath = os.path.join(outdir, outname)

pdfpaths = glob(os.path.join(classified_dir,'*pdf'))

classes = [p.split('[')[1].replace('].pdf','') for p in pdfpaths]
pdfnames = list(map(os.path.basename,
    [p.split('[')[0].replace('vet_','').replace('_llc','') for p in pdfpaths]))

df = pd.DataFrame({'Name':pdfnames, 'Tags':classes})

if not os.path.exists(outpath):
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))
else:
    print('found {}'.format(outpath))

# PC cut

outname = '20190611_LGB_sector6_classifications_PC_cut.csv'
outpath = os.path.join(outdir, outname)

outdf = df[df['Tags'].str.contains('PC', regex=False)]

if not os.path.exists(outpath):
    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))
else:
    print('found {}'.format(outpath))


# good cut

outname = '20190611_LGB_sector6_classifications_good_cut.csv'
outpath = os.path.join(outdir, outname)

outdf = df[df['Tags'].str.contains('good', regex=False)]

if not os.path.exists(outpath):
    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))
else:
    print('found {}'.format(outpath))


