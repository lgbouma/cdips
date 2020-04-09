import os, shutil, socket
from glob import glob
import pandas as pd, numpy as np
from functools import reduce

def main():
    isfull = 0
    iscollabsubclass = 0
    organize_PCs = 1

    sector = 13
    today = '20200320'

    if isfull:
        given_full_classifications_organize(sector=sector, today=today)
    if iscollabsubclass:
        given_collab_subclassifications_merge(sector=sector)
    if organize_PCs:
        given_merged_organize_PCs(sector=sector)


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
    datadir = os.path.join(
        os.path.expanduser('~'),
        'Dropbox/proj/cdips/results/vetting_classifications/'
    )
    if sector==6:
        classfiles = [
            os.path.join(datadir, '20190621_sector-6_PCs_LGB_class.txt'),
            os.path.join(datadir, '20190621_sector-6_PCs_JH_class.txt'),
            os.path.join(datadir, '20190621_sector-6_PCs_JNW_class.txt')
        ]
    elif sector==7:
        classfiles = [
            os.path.join(datadir, '20190621_sector-7_PCs_LGB_class.txt'),
            os.path.join(datadir, '20190621_sector-7_PCs_JH_class.txt')
        ]
    elif sector==8:
        classfiles = [
            os.path.join(datadir, '20191121_sector-8_PCs_LGB_class.txt'),
            os.path.join(datadir, '20191121_sector-8_PCs_JH_class.txt'),
            os.path.join(datadir, '20191121_sector-8_PCs_JNW_class.txt')
        ]
    elif sector==9:
        classfiles = [
            os.path.join(datadir, '20191101_sector-9_PCs_LGB_class.txt'),
            os.path.join(datadir, '20191101_sector-9_PCs_JH_class.txt'),
            os.path.join(datadir, '20191101_sector-9_PCs_JNW_class.txt')
        ]
    elif sector==10:
        classfiles = [
            os.path.join(datadir, '20191118_sector-10_PCs_LGB_class.txt'),
            os.path.join(datadir, '20191118_sector-10_PCs_JH_class.txt'),
            os.path.join(datadir, '20191118_sector-10_PCs_JNW_class.txt')
        ]
    elif sector==11:
        classfiles = [
            os.path.join(datadir, '20191205_sector-11_PCs_LGB_class.txt'),
            os.path.join(datadir, '20191205_sector-11_PCs_JH_class.txt'),
            os.path.join(datadir, '20191205_sector-11_PCs_JNW_class.txt')
        ]
    elif sector==12:
        classfiles = [
            os.path.join(datadir, '20200317_sector-12_PCs_LGB_class.txt'),
            os.path.join(datadir, '20200317_sector-12_PCs_JH_class.txt'),
            os.path.join(datadir, '20200317_sector-12_PCs_JNW_class.txt')
        ]
    elif sector==13:
        classfiles = [
            os.path.join(datadir, '20200320_sector-13_PCs_LGB_class.txt'),
            os.path.join(datadir, '20200320_sector-13_PCs_JH_class.txt'),
            os.path.join(datadir, '20200320_sector-13_PCs_JNW_class.txt')
        ]

    outpath = os.path.join(
        datadir, 'sector-{}_PCs_MERGED_SUBCLASSIFICATIONS.csv'.format(sector)
    )
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

    # merge all the classication dataframes on Name to one dataframe
    mdf = reduce(lambda x, y: pd.merge(x, y, on='Name'), dfs)

    mdf.to_csv(outpath, sep=';', index=False)
    print('wrote {}'.format(outpath))


def given_merged_organize_PCs(sector=None):
    """
    Using output from given_collab_subclassifications_merge, assign gold=2,
    maybe=1, junk, and look closer at anything with average rating >1 (note:
    not >=). The one exception: things classified as "not_cdips_still_good",
    which go in their own pile.
    """

    datadir = os.path.join(
        os.path.expanduser('~'),
        'Dropbox/proj/cdips/results/vetting_classifications/'
    )

    inpath = os.path.join(
        datadir, 'sector-{}_PCs_MERGED_SUBCLASSIFICATIONS.csv'.format(sector)
    )
    df = pd.read_csv(inpath, sep=';')

    tag_colnames = [c for c in df.columns if 'Tags' in c]

    # iterate over ["LGB_tags", "JH_tags", "JNW_tags"] to get scores assigned
    # by each
    for tag_colname in tag_colnames:

        newcol = tag_colname.split('_')[0]+'_score'

        classifier_isgold = np.array(
            df[tag_colname].str.lower().str.contains('gold')
        )
        classifier_ismaybe = np.array(
            df[tag_colname].str.lower().str.contains('maybe')
        )
        classifier_isjunk = np.array(
            df[tag_colname].str.lower().str.contains('junk')
        )

        df[newcol] = (
            2*classifier_isgold +
            1*classifier_ismaybe +
            0*classifier_isjunk
        )

    df['average_score'] = (
        df['LGB_score'] + df['JH_score']  + df['JNW_score']
    ) / 3

    threshold_cutoff = 1.0

    df['clears_threshold'] = (df['average_score'] > threshold_cutoff)

    #
    # nb. not_cdips_still_good will go in a special pile!
    #

    classifier_isnotcdipsstillgood = np.array(
        df["LGB_Tags"].str.lower().str.contains('not_cdips_still_good')
    )

    df['is_not_cdips_still_good'] = classifier_isnotcdipsstillgood

    outpath = os.path.join(
        datadir, 'sector-{}_PCs_MERGED_RATINGS.csv'.format(sector)
    )
    df.to_csv(outpath, sep=';', index=False)
    print('made {}'.format(outpath))


    #
    # output:
    # 1) things that clear threshold, and are CDIPS objects (not field stars)
    # 2) things that are in the "not CDIPS, still good" pile
    #

    df_clears_threshold = df[df.clears_threshold & ~df.is_not_cdips_still_good]
    df_is_not_cdips_still_good = df[df.is_not_cdips_still_good]


    #
    # 1) CDIPS OBJECTS
    #
    outpath = os.path.join(
        datadir, 'sector-{}_PCs_CLEAR_THRESHOLD.csv'.format(sector)
    )
    df_clears_threshold.to_csv(outpath, sep=';', index=False)
    print('made {}'.format(outpath))

    # now copy to new directory
    outdir = os.path.join(datadir, 'sector-{}_CLEAR_THRESHOLD'.format(sector))
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if sector==6:
        srcdir = '../results/vetting_classifications/20190617_sector-6_PC_cut'
    elif sector==7:
        srcdir = '../results/vetting_classifications/20190618_sector-7_PC_cut'
    elif sector in [8,9,10,11]:
        # NB. I "remade" these vetting plots to add the neighborhood charts
        srcdir = glob(
            '../results/vetting_classifications/2019????_sector-{}_PC_cut_remake'.
            format(sector)
        )
        assert len(srcdir) == 1
        srcdir = srcdir[0]
    elif sector in [12,13]:
        srcdir = glob(
            '../results/vetting_classifications/2020????_sector-{}_PC_cut'.
            format(sector)
        )
        assert len(srcdir) == 1
        srcdir = srcdir[0]

    for n in df_clears_threshold['Name']:
        src = os.path.join(srcdir, str(n))
        dst = os.path.join(outdir, str(n))
        if not os.path.exists(dst):
            try:
                shutil.copyfile(src, dst)
                print('copied {} -> {}'.format(src, dst))
            except FileNotFoundError:
                print('WRN! DID NOT FIND {}'.format(src))
        else:
            print('found {}'.format(dst))

    #
    # 2) NOT_CDIPS_STILL_GOOD
    #
    outpath = os.path.join(
        datadir, 'sector-{}_PCs_NOT_CDIPS_STILL_GOOD.csv'.format(sector)
    )
    df_is_not_cdips_still_good.to_csv(outpath, sep=';', index=False)
    print('made {}'.format(outpath))

    # now copy to new directory
    outdir = os.path.join(
        datadir, 'sector-{}_NOT_CDIPS_STILL_GOOD'.format(sector)
    )
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if sector in [6,7]:
        raise NotImplementedError
    elif sector in [8,9,10,11]:
        srcdir = glob(
            '../results/vetting_classifications/2019????_sector-{}_PC_cut_remake'.
            format(sector)
        )
        assert len(srcdir) == 1
        srcdir = srcdir[0]
    elif sector in [12,13]:
        srcdir = glob(
            '../results/vetting_classifications/2020????_sector-{}_PC_cut'.
            format(sector)
        )
        assert len(srcdir) == 1
        srcdir = srcdir[0]

    for n in df_is_not_cdips_still_good['Name']:
        src = os.path.join(srcdir, str(n))
        dst = os.path.join(outdir, str(n))
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            print('copied {} -> {}'.format(src, dst))
        else:
            print('found {}'.format(dst))



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
    outdir = os.path.join(
        os.path.expanduser('~'),
        'Dropbox/proj/cdips/results/vetting_classifications/'
    )
    classified_dir = os.path.join(
        outdir, 'sector_{}_{}_LGB_DONE/'.format(sector,today)
    )
    outtocollabdir = os.path.join(
        outdir, '{}_sector-{}_{}'
    )
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
            os.path.join(
                os.path.expanduser('~'),
                'local/cdips/vetting/sector-{}/pdfs/vet_'.format(sector)+str(n)+'_llc.pdf'
            ),
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
            os.path.join(
                os.path.expanduser('~'),
                'local/cdips/vetting/sector-{}/pdfs/vet_'.format(sector)+str(n)+'_llc.pdf'
            ),
            os.path.join(collabdir,'vet_'+str(n)+'_llc.pdf')
        )


if __name__ == '__main__':
    main()
