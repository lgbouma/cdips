#!/usr/bin/env bash

##########################################
# USAGE: ./collect_figures.sh
##########################################

fdir=../paper/figures/
pdir=../paper/

cp ${fdir}GI_figs/cluster_positions_ecliptic_scicase.pdf \
   ${pdir}f1_PLACEHOLDER.pdf

cp ${fdir}pipeline_figure.pdf ${pdir}f2.pdf

echo "collected figures -> /paper/f?.pdf"
