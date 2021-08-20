# NOTE: DEPRECATED AFTER 2019/09/19 EXOFOPTESS UPLOAD

import os, socket, shutil
from glob import glob
import numpy as np, pandas as pd
from numpy import array as nparr
from cdips.utils import today_YYYYMMDD, make_tarfile_from_fpaths

from astrobase import imageutils as iu

from cdips.utils import (
    find_rvs as fr,
    get_vizier_catalogs as gvc,
    collect_cdips_lightcurves as ccl
)
from cdips.utils.pipelineutils import save_status, load_status

##########
# config #
##########

hostname = socket.gethostname()
if 'phtess1' in hostname:
    fitdir = "/home/lbouma/proj/cdips/results/fit_gold"
elif 'brik' in hostname:

    fitdir = "/home/luke/Dropbox/proj/cdips/results/fit_gold"

    exofopdir = "/home/luke/Dropbox/proj/cdips/data/exoFOP_uploads"

    hlspreportdir = os.path.join(exofopdir, 'files_to_upload',
                                 'hlsp_vetting_reports')

    nbhddir = '/home/luke/Dropbox/proj/cdips_followup/results/neighborhoods'

    exofop_upload_dir = os.path.join(
        exofopdir, 'files_to_upload',
        'to_upload_{}'.format(today_YYYYMMDD()))
    if not os.path.exists(exofop_upload_dir):
        os.mkdir(exofop_upload_dir)

    uploadnumber = 1
    while uploadnumber < 999:
        uploaddir = os.path.join(
            os.path.dirname(exofop_upload_dir),
            'lb{datestamp}-{uploadnumber}'.format(datestamp=today_YYYYMMDD(),
                                                  uploadnumber=str(uploadnumber).zfill(3))
        )
        if os.path.exists(uploaddir):
            uploadnumber +=1
            continue
        else:
            os.mkdir(uploaddir)
            print('made {}'.format(uploaddir))
            break


else:
    raise ValueError('where is fit_gold directory on {}?'.format(hostname))

########
# main #
########

def main(is_20190818_exofop_upload=1):
    """
    ----------
    Args:

        is_20190818_exofop_upload: if True, reads in the manually-written (from
        google spreadsheet) comments and source_ids, and writes those to a
        special "TO_EXOFOP" csv file.
    """

    if not is_20190818_exofop_upload:
        raise NotImplementedError

    df = pd.read_csv('/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/data/exoFOP_uploads/20190918_s6_and_s7_w_sourceid.csv',
                     sep='|')

    for ix, r in df.iterrows():

        source_id = np.int64(r['source_id'])
        ticstr = str(r['target']).replace(r'.01','')

        #
        # first, get and rename the vetting reports
        #
        reportpath = glob(os.path.join(
            hlspreportdir, '*{}*pdf'.format(source_id))
        )

        if len(reportpath) != 1:
            raise AssertionError(
                'need a vetting report for {}'.format(source_id)
            )
        reportpath = reportpath[0]

        reportname = os.path.basename(reportpath).replace('.pdf','')

        lettercode = "O" # other
        dstname = (
            '{ticstr}{lettercode}-lb{datestamp}_{reportname}.pdf'.
            format(ticstr=ticstr, lettercode=lettercode,
                   datestamp=today_YYYYMMDD(), reportname=reportname)
        )

        dstpath = os.path.join(exofop_upload_dir, dstname)

        if not os.path.exists(dstpath):
            shutil.copyfile(reportpath, dstpath)
            print('copy {} -> {}'.format(reportpath, dstpath))
        else:
            print('found {}'.format(dstpath))

        #
        # second, get the rename the neighborhood png inspection plots
        #
        nbhdpath = glob(os.path.join(
            nbhddir, '*{}*hood.png'.format(ticstr)))
        nbhdextrapath = glob(os.path.join(
            nbhddir, '*{}*hood_extra.png'.format(ticstr)))

        for p in [nbhdpath, nbhdextrapath]:

            if len(p) != 1:
                #maybe b/c Zari2018 PMS
                print('\nWARNING: DID NOT FIND NBHD REPORT FOR {}'.
                      format(ticstr))
                continue
            p = p[0]

            nbhdname = os.path.basename(p).replace('.png','')
            nbhdname = '_'.join(nbhdname.split('_')[1:])

            lettercode = "O" # other
            dstname = (
                '{ticstr}{lettercode}-lb{datestamp}_{nbhdname}.png'.
                format(ticstr=ticstr, lettercode=lettercode,
                       datestamp=today_YYYYMMDD(), nbhdname=nbhdname)
            )

            dstpath = os.path.join(exofop_upload_dir, dstname)

            if not os.path.exists(dstpath):
                shutil.copyfile(p, dstpath)
                print('copy {} -> {}'.format(p, dstpath))
            else:
                print('found {}'.format(dstpath))

    #
    # make a text file that describes each file (file name, data tag,
    # optional group name, proprietary period, and description.
    #
    files_to_upload = glob(os.path.join(exofop_upload_dir,'*'))

    fnames = [os.path.basename(x) for x in files_to_upload if not
              x.endswith('.txt')]

    description = []
    tags = []
    for f in fnames:

        ticstr_plus_01 = f.split('O-')[0] + '.01'
        sdf = df[df['target']==ticstr_plus_01]

        assert len(sdf)==1

        tags.append(str(sdf['tag'].iloc[0]))

        if 'vet_hlsp' in f:
            description.append('CDIPS pipeline report')
        elif 'neighborhood.png' in f:
            description.append('CDIPS neighborhood analysis view 1')
        elif 'neighborhood_extra.png' in f:
            description.append('CDIPS neighborhood analysis view 2')

    txt_df = pd.DataFrame({
        'fname':fnames,
        'tag':tags,
        'groupname':np.ones(len(fnames))*np.nan,
        'proprietary_period':np.zeros(len(fnames)).astype(int),
        'description':description
    })

    txtfiles = glob(os.path.join(exofop_upload_dir, '*.txt'))
    for f in txtfiles:
        os.remove(f)

    outpath = os.path.join(
        exofop_upload_dir,
        'lb{datestamp}-{uploadnumber}.txt'.format(datestamp=today_YYYYMMDD(),
                                                  uploadnumber=str(uploadnumber).zfill(3))
    )
    txt_df.to_csv(outpath, sep="|", header=False, index=False)
    print('made {}'.format(outpath))

    #
    # copy the directory to the one that we will actually use for the upload
    # and compression.
    #

    files_to_upload = glob(os.path.join(exofop_upload_dir,'*'))

    dsts = []
    for f in files_to_upload:
        src = f
        dst = os.path.join(uploaddir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            print('copied {}->{}'.format(src, dst))
        else:
            print('found {}'.format(dst))
        dsts.append(dst)

    # FIXME
    # NOTE: this is super annoying. you have to do something like
    #
    # cd ~/Dropbox/proj/cdips/data/exoFOP_uploads/files_to_upload/lb20190918-005
    #
    # tar cf lb20190918-005.tar *
    #
    # because exoFOPtess REQUIRES that tarball to directly inflate to the file
    # level


if __name__=="__main__":
    main()
