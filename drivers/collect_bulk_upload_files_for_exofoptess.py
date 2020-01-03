import os, socket, shutil
from glob import glob
import numpy as np, pandas as pd
from numpy import array as nparr
from cdips.utils import today_YYYYMMDD, make_tarfile_from_fpaths

from astrobase import imageutils as iu
from astrobase.services.identifiers import gaiadr2_to_tic

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
if 'phtess' in hostname:
    fitdir = "/home/lbouma/proj/cdips/results/fit_gold"
    exofopdir = "/home/lbouma/proj/cdips/data/exoFOP_uploads"
elif 'brik' in hostname:
    fitdir = "/home/luke/Dropbox/proj/cdips/results/fit_gold"
    exofopdir = "/home/luke/Dropbox/proj/cdips/data/exoFOP_uploads"

else:
    raise ValueError('need to define directories on {}'.format(hostname))

DATESTR = None
if DATESTR is None:
    DATESTR = today_YYYYMMDD()

hlspreportdir = os.path.join(exofopdir, 'files_to_upload',
                             'hlsp_vetting_reports', DATESTR)
if not os.path.exists(hlspreportdir):
    print('made {}'.format(hlspreportdir))
    os.mkdir(hlspreportdir)

exofop_upload_dir = os.path.join(
    exofopdir, 'files_to_upload',
    'to_upload_{}'.format(DATESTR))

if not os.path.exists(exofop_upload_dir):
    os.mkdir(exofop_upload_dir)

uploadnumber = 1
while uploadnumber < 999:
    uploaddir = os.path.join(
        os.path.dirname(exofop_upload_dir),
        'lb{datestamp}-{uploadnumber}'.format(datestamp=DATESTR,
                                              uploadnumber=str(uploadnumber).zfill(3))
    )
    if os.path.exists(uploaddir):
        uploadnumber +=1
        continue
    else:
        os.mkdir(uploaddir)
        print('made {}'.format(uploaddir))
        break

########
# main #
########

def main(uploadnamestr='sectors_8_thru_11_clear_threshold'):
    """
    ----------
    Args:

        uploadnamestr: used as unique identifying string in file names. Must
        match that from merge_for_exofoptess.

    """

    dfpath = os.path.join(
        exofopdir, "{}_{}_w_sourceid.csv".
        format(DATESTR, uploadnamestr)
    )
    df = pd.read_csv(dfpath, sep='|')

    ticstrs = []
    for ix, r in df.iterrows():

        source_id = np.int64(r['source_id'])

        if str(r['target']).startswith('TIC'):
            ticstr = str(r['target']).replace(r'.01','')
        else:
            ticid = gaiadr2_to_tic(str(source_id))
            if ticid is None:
                raise ValueError(
                    'failed to get TICID for {}'.format(source_id)
                )
            ticstr = 'TIC'+ticid

        ticstrs.append(ticstr)

        #
        # first, get and rename the vetting reports
        #
        reportpaths = glob(os.path.join(
            hlspreportdir, '*{}*pdf'.format(source_id))
        )

        if len(reportpaths) == 0:
            raise AssertionError(
                'need a vetting report for {}'.format(source_id)
            )

        for reportpath in reportpaths:

            reportname = os.path.basename(reportpath).replace('.pdf','')

            lettercode = "O" # other
            dstname = (
                '{ticstr}{lettercode}-lb{datestamp}_{reportname}.pdf'.
                format(ticstr=ticstr, lettercode=lettercode,
                       datestamp=DATESTR, reportname=reportname)
            )

            dstpath = os.path.join(exofop_upload_dir, dstname)

            if not os.path.exists(dstpath):
                shutil.copyfile(reportpath, dstpath)
                print('copy {} -> {}'.format(reportpath, dstpath))
            else:
                print('found {}'.format(dstpath))

    df['ticstrs'] = ticstrs

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

        ticstr = f.split('O-')[0]

        sdf = df[df['ticstrs']==ticstr]

        assert len(sdf)==1

        tags.append(str(sdf['tag'].iloc[0]))

        if 'vet_hlsp' in f:
            sectornum = f.split('-')[2].lstrip('0')
            description.append('CDIPS pipeline report (S{})'.format(sectornum))

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
        'lb{datestamp}-{uploadnumber}.txt'.format(datestamp=DATESTR,
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
