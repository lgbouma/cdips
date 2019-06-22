import os, shutil
from glob import glob
import pandas as pd, numpy as np
from functools import reduce

def main():
    isfull = 0
    iscollabsubclass = 1

    if isfull:
        given_full_classifications_organize(sector=7, today='20190618')
    if iscollabsubclass:
        given_collab_subclassifications_merge(sector=7)

def ls_to_df(classfile, classifier='LGB'):
    indf = pd.read_csv(classfile, names=['lsname'])

    pdfpaths = np.array(indf['lsname'])
    # e.g.,
    # vet_hlsp_cdips_tess_ffi_gaiatwo0002890062263458259968-0006_tess_v01_llc[gold PC].pdf

    classes = [p.split('[')[1].replace('].pdf','') for p in pdfpaths]

    pdfnames = [p.split('[')[0]+'.pdf' for p in pdfpaths]

    df = pd.DataFrame({'Name':pdfnames, classifier+'_Tags':classes})

    return df


def hartmanformat_to_df(classfile, classifier="JH"):

    with open(classfile, 'r') as f:
        lines = f.readlines()

    pdfnames = [l.split(' ')[0] for l in lines]

    classes = [' '.join(l.split(' ')[1:]).rstrip('\n') for l in lines]

    df = pd.DataFrame({'Name':pdfnames, classifier+'_Tags':classes})

    return df


def given_collab_subclassifications_merge(sector=6):
    """
    LGB or JH or JNW has done classifications. merge them, save to csv.
    """
    datadir = '/home/luke/Dropbox/proj/cdips/results/vetting_classifications/'
    if sector==6:
        classfiles = [
            os.path.join(datadir, '20190621_sector-6_PCs_LGB_class.txt'),
            os.path.join(datadir, '20190621_sector-6_PCs_JH_class.txt'),
            os.path.join(datadir, '20190621_sector-6_PCs_JNW_class.txt')
        ]
        outpath = os.path.join(datadir,
                               'sector-6_PCs_MERGED_SUBCLASSIFICATIONS.csv')
    elif sector==7:
        classfiles = [
            os.path.join(datadir, '20190621_sector-7_PCs_LGB_class.txt'),
            os.path.join(datadir, '20190621_sector-7_PCs_JH_class.txt')
        ]
        outpath = os.path.join(datadir,
                               'sector-7_PCs_MERGED_SUBCLASSIFICATIONS.csv')

    print('merging {}'.format(repr(classfiles)))

    dfs = []
    for classfile in classfiles:
        if 'LGB' in classfile:
            df = ls_to_df(classfile, classifier='LGB')
        elif 'JNW' in classfile:
            df = ls_to_df(classfile, classifier='JNW')
        elif 'JH' in classfile:
            df = hartmanformat_to_df(classfile, classifier='JH')
        dfs.append(df)

    mdf = reduce(lambda x, y: pd.merge(x, y, on='Name'), dfs)

    mdf.to_csv(outpath, sep=';', index=False)
    print('wrote {}'.format(outpath))


def given_full_classifications_organize(
    sector=7,
    today='20190618'
):
    """
    given a directory classified w/ TagSpaces tags, produce a CSV file with the
    classifications

    also produce CSV file with PC only classifications

    also produce directories with cuts to send to vetting collaborators
    """
    ##########################################
    # modify these
    classified_dir = '/home/luke/Dropbox/proj/cdips/results/vetting_classifications/sector_{}_{}_LGB_DONE/'.format(sector,today)
    outdir = '/home/luke/Dropbox/proj/cdips/results/vetting_classifications/'
    outtocollabdir = '/home/luke/Dropbox/proj/cdips/results/vetting_classifications/{}_sector-{}_{}'
    outname = '{}_LGB_sector{}_classifications.csv'.format(today,sector)
    ##########################################

    outpath = os.path.join(outdir, outname)

    pdfpaths = glob(os.path.join(classified_dir,'*pdf'))

    if not len(pdfpaths) > 1:
        raise AssertionError('bad pdfpaths. no glob matches.')
    for p in pdfpaths:
        if '[' not in p:
            raise AssertionError('got {} with no classification'.format(p))

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
    if not os.path.exists(collabdir):
        os.mkdir(collabdir)

    for n in outdf['Name']:
        shutil.copyfile(
            '/home/luke/local/cdips/vetting/sector-{}/pdfs/vet_'.format(sector)+str(n)+'_llc.pdf',
            os.path.join(collabdir,'vet_'+str(n)+'_llc.pdf')
        )

    # "good" (aka: interesting) object cut
    goodoutname = outname.replace('.csv','_good_cut.csv')
    outpath = os.path.join(outdir, goodoutname)

    outdf = df[df['Tags'].str.contains('good', regex=False)]

    if not os.path.exists(outpath):
        outdf.to_csv(outpath, index=False)
        print('made {}'.format(outpath))
    else:
        print('found {}'.format(outpath))

    collabdir = outtocollabdir.format(today, sector, 'interesting_cut')
    if not os.path.exists(collabdir):
        os.mkdir(collabdir)

    for n in outdf['Name']:
        shutil.copyfile(
            '/home/luke/local/cdips/vetting/sector-{}/pdfs/vet_'.format(sector)+str(n)+'_llc.pdf',
            os.path.join(collabdir,'vet_'+str(n)+'_llc.pdf')
        )


if __name__ == '__main__':
    main()
