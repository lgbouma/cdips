1. make lightcurves using pipe-trex.

2. run `trex_lc_to_mast_lc` (symlinks CDIPS matches into a good directory
   structure; then hard copies the LCs and reformats them into the MAST
   format).

3. for an initial look at the rms vs mag for the LCs, run
   `get_cdips_lc_stats`. this can also tell you how many NaNs were in the
   run.

4. to learn how many stars were expected on silicon, run
   `how_many_cdips_stars_on_silicon`

5. `do_initial_period_finding`: runs initial TLS and LS, to apply use for cuts
   and pass to reconstructive TFA.
   also writes `which_references_and_clusters_matter.txt` (to know how badly
   your results depend on the Dias 2014 catalog)

6. `skim_cream`: make BLS/SPDM two-panel checkplots for the "cream" determined
   by the TLS and LS checks above.

7. `reconstructive_tfa/RunTFASR.sh` does signal reconstruction for TFA
   lightcurves

8.  `skim_cream`: make BLS/SPDM two-panel checkplots for the "cream" determined
    by the TLS and LS checks above.

9. `make_vetting_multipg_pdf`: make a multipage PDF with the information needed
   to make classifications for vetting.
