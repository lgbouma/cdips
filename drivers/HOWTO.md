TO MAKE THE TARGET STAR CATALOG
----------

The `/drivers/` are not used. Instead, the process is:

1. Use `/cdips/catalogbuild/homogenize_cluster_lists.py` to get everything in
   same format.  The merging is then done here as well.  (Iteratively).

2. Use `/cdips/catalogbuild/construct_unique_cluster_name_column.py` to do what
   is advertized.

TO PRODUCE PLANET CANDIDATES FROM CDIPS LCS
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
   `do_initial_period_finding.py` (w/ the sector number too).

4. `reconstructive_tfa/RunTFASR.sh` does signal reconstruction for TFA
   lightcurves

5. `make_all_vetting_reports`: make a multipage PDF with the information needed
   to make classifications for vetting.

    * Run it twice, and verify you made all of them with
    `python -u make_all_vetting_reports.py &> logs/s#_vetting_report.log &`
    `python -u make_all_vetting_reports.py &> logs/s#_vetting_report_check.log &`
    `verify_vetting_reports.sh`


TO PRODUCE PERIODOGRAM CLASSIFICATION (ALL VARIABILITY, NOT ONLY PLANETS)
----------

1. Make lightcurves using cdips-pipeline. (Update the "lc_list_YYYYMMDD.txt"
   metadata file to ensure that they are findable).

2. Select the stars you wish to do the analysis for. For instance, this could
   be a set of stars in a particular open cluster.

3. Run `do_allvariable_period_finding.py`: period-finds with LS, SPDM, and TLS.

4. `make_all_vetting_reports`: make a multipage PDF with the information needed
   to make classifications for periodogram/classification vetting.


TO CHECK STAR CATALOGS AGAINST TOIS
----------
Run the scripts in `/toi_youngstar_matching/*.py`.


TO PROCESS VETTING CLASSIFICATIONS
----------

1. `classification_postprocessing`: does three stages of classification:

    1. `given_full_classifications_organize`: LGB classifies everything, sends
       PCs to team
    2. LGB gold/maybe/junk classification of PCs. Write output to e.g.,
       `ls *pdf > 20190621_sector-6_PCs_LGB_class.txt`
    3. `given_collab_subclassifications_merge`: team responds w/
       classifications (csvs and txt files). they must be merged
    4. `given_merged_gold_organize_PCs`: organize the results of the merge into
       objects that clear the classification cutoff threshold, and those that
       are not CDIPS objects, but still good.

2. `fit_models_to_gold`: MCMC fit Mandel-Agol transits to gold above -- these
   parameters are used for CTOIs (and probably shouldn't be blindly believed in
   publications).

3. `merge_for_exofoptess`: merge csv results from `fit_models_to_gold`

4. `collect_bulk_upload_files_for_exofoptess`
    1. (phtess2) /home/lbouma/proj/cdips/data/exoFOP_uploads/files_to_upload/collect_reports.sh
    2. (phtess2) Execute the main driver.
    3. (brik) /home/luke/Dropbox/proj/cdips/data/exoFOP_uploads/pull_from_phtess2.sh


TO COMMUNICATE RESULTS
----------

1. `paper_plot_all_figures`: plots figures for paper I.

2. `paper_get_LC_and_target_star_stats`: analyze stats of all CDIPS target
   stars. also stats of all CDIPS LCs.  how many are there total?  what
   fraction come from which references?  how many are single-source claims?  how
   many are multi-source claims?  what clusters are important?
