###############################################################################
# Purpose: Given the simpletransit fit_gold output results (png files, and pdf
# tables), create a single MCMC summary page to append to the CDIPS vetting
# reports.
#
# Author: LGB
#
# Date: Thu Feb 17 10:29:37 2022
#
# Usage: maybe as a stand-alone shell script... maybe will be ported to python.
#
# Dependencies:
# * imagemagick `convert`, https://imagemagick.org/index.php
# * pdftk
###############################################################################

#
# First, convert the PDF table to an image, and trim out the white space.
#
convert -density 300 *posteriortable.pdf -quality 100 -alpha remove tempposteriortable.png
convert tempposteriortable.png -bordercolor white -border 2x2 -trim +repage temptrimmed.png
convert temptrimmed.png -resize 1971x temptrimmedresized.png

#
# Next, generate the individual panels to be used
#

# Panel 1 (top): light curves.
convert *rawlc.png -resize 1971x temprawlc.png
convert *rawtrimlc.png -resize 1971x temprawtrimlc.png
convert temprawlc.png temprawtrimlc.png +append 1.png
rm temp*png

# Panel 2 (bottom): phase plot, posterior table, and corner plot.
convert *cornerplot.png -resize 1971x tempcorner.png
convert *phaseplot.png -resize 1971x tempphaseplot.png
convert tempphaseplot.png temptrimmedresized.png -append left.png
convert left.png tempcorner.png +append 2.png
rm temp*png

# Create the new pdf page to append!
convert 1.png 2.png -append page.png
convert page.png -units pixelsperinch -density 131.4 +repage page.pdf
rm 1.png 2.png

# Append it!
pdftk vet*pdf page.pdf output new_vet.pdf
