1. make lightcurves using pipe-trex.

2. run `trex_lc_to_mast_lc` (symlinks CDIPS matches into a good directory
   structure; then hard copies the LCs and reformats them into the MAST
   format).

3. for an initial look at the rms vs mag for the LCs, run
   `get_cdips_lc_stats`. this can also tell you how many NaNs were in the
   run.  also writes `supplemented_cdips_lc_statistics.txt`, which is used for
   the vetting pdfs.

4. to learn how many stars were expected on silicon, run
   `how_many_cdips_stars_on_silicon`

5. `do_initial_period_finding`: runs initial TLS and LS, to apply use for cuts
   and pass to reconstructive TFA.
   also writes `which_references_and_clusters_matter.txt` (to know how badly
   your results depend on the Dias 2014 catalog)
   also makes plots that show the distribution of results.

6. [optional] `skim_cream`: make BLS/SPDM two-panel checkplots for the "cream"
   determined by the TLS and LS checks above.

7. `reconstructive_tfa/RunTFASR.sh` does signal reconstruction for TFA
   lightcurves

8. `drivers/make_fficut_wget_script.py`: makes wget script to grab the FFI
   cutouts needed for centroid analysis. (There are manual steps).

9.  [optional] `skim_cream`: make BLS/SPDM two-panel checkplots for the "cream"
    determined by the TLS and LS checks above.

10. `make_vetting_multipg_pdf`: make a multipage PDF with the information needed
   to make classifications for vetting.

    * Check `logs/vetting_pdf.log` for name-matching errors. (First pass, they
      will always show up!)

11. `paper_plot_all_figures`: plots figures for paper I.
