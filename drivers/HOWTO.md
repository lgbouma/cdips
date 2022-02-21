TO MAKE THE TARGET STAR CATALOG
----------

The `/drivers/` are not used. Instead, the process is:

1. Use `/cdips/catalogbuild/homogenize_cluster_lists.py` to get everything in
   same format (`cluster`, `source_id`, `age`).  The merging is then done here
   as well.  (Iteratively).

2. Use `/cdips/catalogbuild/construct_unique_cluster_name_column.py` to do what
   is advertized.

TO PRODUCE PLANET CANDIDATES FROM CDIPS LCS
----------

1. make lightcurves using cdips-pipeline.

2. `lc_to_hlspformat`: makes HLSP lightcurves, by
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

3. `do_initial_period_finding`: runs detrending, LS, and TLS.
    also writes `which_references_and_clusters_matter.txt`, to get familiar
    with the clusters in any given field. also makes plots that show the
    distribution of results, and automatically fine-tunes the criteria for
    "signal detection".

4. (DEPRECATED -- only used in v0 S1-S13 reductions)
   `reconstructive_tfa/RunTFASR.sh` does signal reconstruction for TFA
    lightcurves.

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
    2. LGB gold/maybe/junk classification of PCs. This might be partially
       already done, so do it via TagSpaces in LGB_YYYYMMDD_sector-SS_PC_cut.
       Write output to e.g., `ls *pdf > 20190621_sector-6_PCs_LGB_class.txt`

    3. `given_collab_subclassifications_merge`: team responds w/
       classifications (csvs and txt files). they must be merged

    4. `given_merged_gold_organize_PCs`: organize the results of the merge into
       objects that clear the classification cutoff threshold, and those that
       are not CDIPS objects, but still good.

2. `fit_models_to_gold`: MCMC fit Mandel-Agol transits to gold above -- these
   parameters are used for CTOIs (and probably shouldn't be blindly believed in
   publications).

3. `merge_for_exofoptess`:
    A) uses results from `fit_models_to_gold` to make a MCMC-vetting page.
    B) appends that page to the existing vetting reports.
    C) merges csv results from `fit_models_to_gold`

4. `collect_bulk_upload_files_for_exofoptess`: run it to make the hyperspecific
   tarballs needed to upload to ExoFOP.
   There will ultimately be two:
      exoFOP_uploads/files_to_upload/lb20220219-001/lb20220219-001.tar,
   which contains all the vetting pdf files and a description text file.
   And
      exoFOP_uploads/params_planet_20220219_001.txt


TO COMMUNICATE RESULTS
----------

1. `paper_plot_all_figures`: plots figures for paper I.

2. `paper_get_LC_and_target_star_stats`: analyze stats of all CDIPS target
   stars. also stats of all CDIPS LCs.  how many are there total?  what
   fraction come from which references?  how many are single-source claims?  how
   many are multi-source claims?  what clusters are important?
