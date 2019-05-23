#!/usr/bin/env bash

##########################################
# USAGE: ./collect_figures.sh
##########################################

fdir=../paper/figures/
pdir=../paper/

# placeholder for positions of clusters
cp ${fdir}GI_figs/cluster_positions_ecliptic_scicase.pdf \
   ${pdir}f1_PLACEHOLDER.pdf

# placeholder for what our pipeline does
cp ${fdir}trex_overview.pdf ${pdir}f2.pdf



lcdir=../results/projid1088_cam2_ccd2_lcs/center_lcs/
# plot of (mag,x,y,T,s,d,k,bkgd) vs time
cp ${lcdir}EPDparams_vs_time_frac1.0_4979427719678442752_llc.png ${pdir}f3.png

# plot of mag vs (t,T,x,y,s,d,k) for four stars
lcdir=../results/projid1088_cam2_ccd2_lcs/
cp ${lcdir}IRM1_vs_EPD_parameters_fourstars_norbit1.png ${pdir}f4.png


echo "collected figures -> /paper/f?.pdf -> /paper/f?.png"
