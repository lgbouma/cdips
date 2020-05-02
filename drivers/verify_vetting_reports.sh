#! /bin/bash
#
# Usage: vetting reports have been made, and make_all_vetting_reports.py has
# been run-through twice, with logs sent to `logfile`.

sector=5

abovebelow=/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_periodfinding/sector-$sector/n_above_and_below_limit.txt
logfile=/home/lbouma/proj/cdips/drivers/logs/s${sector}_vetting_report_check.log

nabove=`cat $abovebelow | cut -d'|' -f1 | cut -d'=' -f2`
nvetreports=`cat $logfile | grep 'Found' | wc -l`

echo 'Expected '$nabove' reports or skipped reports.'
echo 'Found '$nvetreports
