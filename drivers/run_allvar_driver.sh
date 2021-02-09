# Usage:
# $ run_allvar_driver.sh &
# Reason:
# This program has memory bloat; this approach manually restarts it.
# The approach is pretty bad; a manual process kill (e.g., `killall bash`, or
# the manual ID) is needed afterward.
while true;
do
  echo 'NEW LOOP BEGINS' `date` >> logs/compstar_NGC_2516_20210209_v2.log
  python -u do_allvariable_report_making.py &>> logs/compstar_NGC_2516_20210209_v2.log ; wait
done
