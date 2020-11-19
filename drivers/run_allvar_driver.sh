# Usage:
# $ run_allvar_driver.sh &
# Reason:
# This program has memory bloat; this approach manually restarts it.
while true;
do
  python -u do_allvariable_report_making.py &> logs/ngc2516_core_plus_halo.log ; wait
done
