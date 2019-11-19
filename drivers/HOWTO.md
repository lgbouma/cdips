TO MAKE THE TARGET STAR CATALOG
----------

The `/drivers/` are not used. Instead, the process is:

1. Use `/cdips/catalogbuild/homogenize_cluster_lists.py` to get everything in
   same format.  The merging is then done here as well.  (Iteratively).

2. Use `/cdips/catalogbuild/construct_unique_cluster_name_column.py` to do what
   is advertized.

TO PRODUCE CANDIDATES FROM CDIPS LCS
----------

1. make lightcurves using cdips-pipeline.

2. `lc_thru_periodfinding`: makes HLSP lightcurves and period-finds, by
   wrapping the following steps.

  * run `trex_lc_to_mast_lc` (symlinks CDIPS matches into a good directory
    structure; then hard copies the LCs and reformats them into the MAST
    format).

  * for an initial look at the rms vs mag for the LCs, run
    `get_cdips_lc_stats`. this can also tell you how many NaNs were in the
    run.  also writes `supplemented_cdips_lc_statistics.txt`, which is used for
    the vetting pdfs.

  * to learn how many stars were expected on silicon, run
    `how_many_cdips_stars_on_silicon`

  * `do_initial_period_finding`: runs initial TLS and LS, to apply use for cuts
     and pass to reconstructive TFA.
     also writes `which_references_and_clusters_matter.txt` (to know how badly
     your results depend on the Dias 2014 catalog). also makes plots that show
     the distribution of results.

3. manually fine-tune your criteria for "signal detection" & update
   `do_initial_period_finding.py`.

4. `reconstructive_tfa/RunTFASR.sh` does signal reconstruction for TFA
   lightcurves

5. `make_all_vetting_reports`: make a multipage PDF with the information needed
   to make classifications for vetting.

    * Check `logs/vetting_pdf.log` for name-matching errors. (First pass, they
      will always show up! Grep for "ERR!")


TO CHECK STAR CATALOGS AGAINST TOIS
----------
Run the scripts in `/toi_youngstar_matching/*.py`.


TO PROCESS VETTING CLASSIFICATIONS
----------

1. `classification_postprocessing`: does three stages of classification:

    1. `given_full_classifications_organize`: LGB classifies everything, sends
       PCs to team
    2. `given_collab_subclassifications_merge`: team responds w/
       classifications (csvs and txt files). they must be merged
    3. `given_merged_gold_organize_PCs`: the results of the merge must be
       organized.

2. `fit_models_to_gold`: MCMC fit Mandel-Agol transits to gold above -- these
   parameters are used for CTOIs (and probably shouldn't be blindly believed in
   publications).

3. `merge_for_exofoptess`: merge csv results from `fit_models_to_gold`

TO COMMUNICATE RESULTS
----------

1. `paper_plot_all_figures`: plots figures for paper I.

2. `paper_get_LC_and_target_star_stats`: analyze stats of all CDIPS target
   stars. also stats of all CDIPS LCs.  how many are there total?  what
   fraction come from which references?  how many are single-source claims?  how
   many are multi-source claims?  what clusters are important?
