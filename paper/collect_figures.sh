#!/usr/bin/env bash

##########################################
# USAGE: ./collect_figures.sh
##########################################

fdir=../paper/figures/
pdir=../paper/

# # positions of clusters
# cp ${fdir}GI_figs/cluster_positions_ecliptic_scicase.pdf \
#    ${pdir}cluster_positions.pdf

# what our pipeline does
cp ${fdir}trex_overview.pdf ${pdir}pipelineoverview.pdf

# rms vs mag 
fdir=../results/paper_figures/
cp ${fdir}rms_vs_mag.png ${pdir}rms_vs_mag.png

# positions of lightcurves
cp ${fdir}sector6_cam[1]_ccd[1-2-3-4]cluster_field_star_positions.png \
   ${pdir}sector6_cam[1]_ccd[1-2-3-4]cluster_field_star_positions.png

# positions of lightcurves
cp ${fdir}cluster_field_star_positions.png ${pdir}cluster_field_star_positions.png

# CDF of T mags of LCs
cp ${fdir}cdf_T_mag.png ${pdir}cdf_T_mag.png

# HRDs of close and entire LC subsets
cp ${fdir}hrd_scat_all_CDIPS_LCs.png ${pdir}hrd_scat_all_CDIPS_LCs.png
cp ${fdir}hrd_scat_close_subset.png ${pdir}hrd_scat_close_subset.png

# HRDs of close and entire LC subsets
cp ${fdir}pm_scat_all_CDIPS_LCs.png ${pdir}pm_scat_all_CDIPS_LCs.png
cp ${fdir}pm_scat_close_subset.png ${pdir}pm_scat_close_subset.png

# counts of the CDIPS target star catalog
cp ${fdir}target_star_cumulative_counts.png \
   ${pdir}target_star_cumulative_counts.png

# catalog matching statistics
cp ${fdir}catalog_to_gaia_match_statistics_MWSC.png \
   ${pdir}mwscmatchstats.png

cp ${fdir}catalog_to_gaia_match_statistics_Dias14.png \
   ${pdir}dias14matchstats.png

# wcs
cp ${fdir}proj1500-s0006-cam1-ccd1-combinedphotref-onenight_spocwcs_sep_hist.png \
   ${pdir}astromresidual_hist.png

cp ${fdir}proj1500-s0006-cam1-ccd1-combinedphotref-onenight_spocwcs_quiver_meas_proj_sep.png \
   ${pdir}astromresidual_quiver.png

# pdftk burst the chosen vetting pdf
pdftk vet_hlsp_cdips_tess_ffi_gaiatwo0005541111035713815552-0007_tess_v01_llc.pdf burst output gaiatwo0005541111035713815552-0007_page%02d.pdf

##########################################

# outdated, probably won't be in paper
fdir=../results/astrometric_residual/
cp ${fdir}proj1510-s0006-cam3-ccd3-combinedphotref-onenight_apertures_on_frame_x_954t1094_y_954t1094.png \
   ${pdir}astromresidual_apertures.png

lcdir=../results/projid1088_cam2_ccd2_lcs/center_lcs/
# plot of (mag,x,y,T,s,d,k,bkgd) vs time
cp ${lcdir}EPDparams_vs_time_frac1.0_4979427719678442752_llc.png \
   ${pdir}epdparams_vs_time.png

# plot of mag vs (t,T,x,y,s,d,k) for four stars
lcdir=../results/projid1088_cam2_ccd2_lcs/
cp ${lcdir}IRM1_vs_EPD_parameters_fourstars_norbit1.png \
   ${pdir}mag_vs_epdparams.png

echo "collected figures -> /paper/f?.pdf -> /paper/f?.png"
