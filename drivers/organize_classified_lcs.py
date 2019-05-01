"""
vetting pdfs and pkls have been made.
they have been looked thru, and classified as one or more of:

    EB,instr,stellar,PC,weirdo,blend

using the `tagspaces` software.

Parse the resulting directory to get a nice csv with the results.
"""

import pandas as pd, numpy as np
from glob import glob
import os, re, shutil

##########################################

def make_classification_df(classified_pdf_dir, outpath):
    all_pdf_paths = glob(os.path.join(
        classified_pdf_dir, 'vet*.pdf'))
    classified_pdf_paths = glob(os.path.join(
        classified_pdf_dir, 'vet*[*.pdf'))

    if len(all_pdf_paths) != len(classified_pdf_paths):
        print('WARNING: you have {} pdfs, but only {} are classified'.
              format(len(all_pdf_paths), len(classified_pdf_dir)))

    EB_flag = np.array(['EB' in f for f in classified_pdf_paths])
    instr_flag = np.array(['instr' in f for f in classified_pdf_paths])
    stellar_flag = np.array(['stellar' in f for f in classified_pdf_paths])
    PC_flag = np.array(['PC' in f for f in classified_pdf_paths])
    weirdo_flag = np.array(['weirdo' in f for f in classified_pdf_paths])
    blend_flag = np.array(['blend' in f for f in classified_pdf_paths])

    flags = [EB_flag, instr_flag, stellar_flag, PC_flag, weirdo_flag, blend_flag]

    original_names = [os.path.basename(f.replace(re.search('\[.*\]',f).group(),''))
                      for f in classified_pdf_paths]
    classifxn_str = [re.search('\[.*\]',f).group() for f in
                      classified_pdf_paths]

    df = pd.DataFrame(
        {'names':original_names,
         'EB_flag':EB_flag,
         'instr_flag':instr_flag,
         'stellar_flag':stellar_flag,
         'PC_flag':PC_flag,
         'weirdo_flag':weirdo_flag,
         'blend_flag':blend_flag,
         'classifxn_str':classifxn_str
        }
    )

    df = df.sort_values(by='names')
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))

    outpath = outpath.replace('.csv','_PC_cut.csv')
    df[df['PC_flag']][['names','classifxn_str']].to_csv(outpath, index=False)
    print('made {}'.format(outpath))

    return df

##########################################

def main():
    who_vetted = "LGB"
    # contains pdfs of the form `vet_3125511477274746880_llc[isblend EB].pdf`
    classified_pdf_dir = '/home/luke/sector6_LGB_vetted'
    # save info csv here
    outputdir = '/home/luke/Dropbox/proj/cdips/results/vetting_classifications'
    outpath = os.path.join(
        outputdir,
        '20190430_{}_sector6_classifications.csv'.format(who_vetted)
    )

    df = make_classification_df(classified_pdf_dir, outpath)

    originaldir = '/home/luke/Dropbox/proj/cdips/results/vetting/sector-6/pdfs'
    for flag in ['EB','PC','weirdo','stellar','instr']:
        dstdir = os.path.join(
            '/home/luke/Dropbox/proj/cdips/results/'
            'vetting_classifications/sector-6_{}s'.format(flag)
        )
        if not os.path.exists(dstdir):
            os.mkdir(dstdir)
        srcs = [os.path.join(originaldir,n) for n in
                df[df['{}_flag'.format(flag)]]['names']]
        dsts = [os.path.join(dstdir,n) for n in
                df[df['{}_flag'.format(flag)]]['names']]

        for src,dst in zip(srcs, dsts):
            if not os.path.exists(dst):
                shutil.copyfile(src,dst)
                print('copy {}->{}'.format(src,dst))
            else:
                print('found {}'.format(dst))

if __name__=="__main__":
    main()
