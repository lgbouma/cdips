import os
from glob import glob

with open('s6s7_full_lc_list.txt','r') as f:
    lines = f.readlines()

outlines = []
for l in lines:
    getpath = (
        l.split('/')[-1].replace('_tess_v01_llc.fits\n','-')
        +
        'cam{}'.format(l.split('/')[1].split('_')[0][-1])
        +
        '-'
        +
        'ccd{}'.format(l.split('/')[1].split('_')[1][-1])
        +
        '_tess_v01_llc.fits'
    )

    geturl = (
        'https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:HLSP/cdips/'
        +
        's{}/'.format(str(int(l.split('/')[0][-1])).zfill(4))
        +
        l.split('/')[1]
        +
        '/'
        +
        getpath
    )

    outlines.append(
        'curl -C - -L -o {getpath} {geturl}\n'.format(
            getpath=getpath, geturl=geturl
        )
    )

with open('s6_s7_curl_script.sh','w') as f:
    f.writelines(outlines)
