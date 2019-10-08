To download individual light curves, we recommend using either the [MAST
Portal](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html), or
[astroquery](https://astroquery.readthedocs.io/en/latest/mast/mast.html).

This directory contains two quick-fix solutions for those who wish to download
all the light curves from Sectors 6 and 7.

NB. the total size of the light curves once downloaded is about 50 Gb.

Solution #1: `s6_s7_curl_script.sh` downloads everything via curl.
Since the file is 40Mb, it is available via dropbox, at this link:
[https://www.dropbox.com/s/jdhautbf99tare7/s6_s7_curl_script.sh?dl=0](https://www.dropbox.com/s/jdhautbf99tare7/s6_s7_curl_script.sh?dl=0).

Solution #2: `download_sector6_and_sector7.py` is a python script you can run
to download everything.
