"""
given a directory classified w/ TagSpaces tags, produce a CSV file with the
classifications

also produce CSV file with PC only classifications

also produce directories with cuts to send to vetting collaborators
"""
import os, shutil
from glob import glob
import pandas as pd

##########################################
# modify these
sector = 6
today = '20190616'
classified_dir = '/home/luke/Dropbox/proj/cdips/results/vetting_classifications/sector_{}_{}_LGB_DONE/'.format(sector,today)
outdir = '/home/luke/Dropbox/proj/cdips/results/vetting_classifications/'
outtocollabdir = '/home/luke/Dropbox/proj/cdips/results/vetting_classifications/{}_sector-{}_{}'
outname = '{}_LGB_sector{}_classifications.csv'.format(today,sector)
##########################################

outpath = os.path.join(outdir, outname)

pdfpaths = glob(os.path.join(classified_dir,'*pdf'))

if not len(pdfpaths) > 1:
    raise AssertionError('bad pdfpaths. no glob matches.')

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

pcoutname = outname.replace('.csv','_PC_cut.csv')
outpath = os.path.join(outdir, pcoutname)

outdf = df[df['Tags'].str.contains('PC', regex=False)]

if not os.path.exists(outpath):
    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))
else:
    print('found {}'.format(outpath))

collabdir = outtocollabdir.format(today, sector, 'PC_cut')
for n in df['Name']:
    shutil.copyfile(
        '/home/luke/local/cdips/vetting/sector-{}/pdfs/vet_'.format(sector)+str(n)+'_llc.pdf',
        os.path.join(collabdir,'vet_'+str(n)+'_llc.pdf')
    )


# good cut

goodoutname = outname.replace('.csv','_good_cut.csv')
outpath = os.path.join(outdir, goodoutname)

outdf = df[df['Tags'].str.contains('good', regex=False)]

if not os.path.exists(outpath):
    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))
else:
    print('found {}'.format(outpath))


collabdir = outtocollabdir.format(today, sector, 'interesting_cut')
for n in df['Name']:
    shutil.copyfile(
        '/home/luke/local/cdips/vetting/sector-{}/pdfs/vet_'.format(sector)+str(n)+'_llc.pdf',
        os.path.join(collabdir,'vet_'+str(n)+'_llc.pdf')
    )


