# Usage:
# $ run_allvar_driver.sh &
# Reason:
# This program has memory bloat; this approach manually restarts it.
# The approach is pretty bad; a manual process kill (e.g., `killall bash`, or
# the manual ID) is needed afterward.
while true;
do
  echo 'NEW LOOP BEGINS' `date` >> logs/NGC_2516_20210429_postsubm_v1.log
  python -u do_allvariable_report_making.py &>> logs/NGC_2516_20210429_postsubm_v1.log ; wait
done
