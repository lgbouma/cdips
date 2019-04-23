import os, pickle, subprocess, itertools

def scp_lightcurves(lcpaths,
                    outdir='../data/cluster_data/lightcurves/sector6_cam2_ccd3/'):

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for lcpath in lcpaths:

        fromstr = "lbouma@phn12:{}".format(lcpath)
        tostr = "{}/.".format(outdir)

        p = subprocess.Popen([
            "scp",
            fromstr,
            tostr,
        ])
        sts = os.waitpid(p.pid, 0)

    return 1

def main():

    filelist = [
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/2949532854543795200_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/2949621227791832064_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/2950836016338711552_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/2954014601377987584_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/3002502411324940288_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/3006706909428340736_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/3007158980506721280_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/3007171311355035136_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/3007458700504002560_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/3007459868735163904_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/3049068652908112896_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/3099227273859068416_llc_raw_tfa_bkgd.png',
    '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam2_ccd3/3099278882184614912_llc_raw_tfa_bkgd.png',
    ]

    scp_lightcurves(filelist)

if __name__=="__main__":
    main()
