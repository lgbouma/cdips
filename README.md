# `cdips`
Cluster Difference Imaging Photometric Survey

## Description

This repository contains some of the code developed for the [CDIPS
project](https://arxiv.org/abs/1910.01133).  If you are interested in light
curves, they can be accessed [through
MAST](https://archive.stsci.edu/hlsp/cdips); planet-finding vetting reports are
available at [ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/) and have
documentation [available here](http://lgbouma.com/notes/).

The tools in this repo can be understood as answers to the question "what do
you do once the [cdips-pipeline](https://github.com/waqasbhatti/cdips-pipeline)
gives you light curves?".

Some possible answers include "find planets", "measure stellar rotation
periods", and "find weird variable stars".  This question is also in many cases
applicable to any time-series data, and some of the tools in this repository
are developed with that generality in mind.

Directory contents are as follows.

* `/drivers/` is the interface to make target catalogs, find planet candidates
  and variable stars, classify, vet the classifications, and make plots and
  tables for papers.  See /drivers/HOWTO.md for instructions.

* `/cdips/` is a python package with tools to make target catalogs, process
  light curves, optimize cdips-pipeline parameters, plot and analyze light
  curves.

* `/paper_I/` contains the CDIPS method paper.

* `/tests/` has testing scripts for vetting report generation, HLSP light curve
  generation, and detrending method comparison.  pytest & CI are not set up for
  this repo.

## Author

Luke Bouma

## License

MIT

## Install

`pip install cdips`

`git clone https://github.com/lgbouma/cdips`
`python setup.py install`

to verify you have the dependencies (with apologies for not making this
automatic in the installation; some of the dependencies are fairly obscure)

`cd cdips/tests`
`python test_environment.py`

One place to start if you are building an environment from scratch:

`conda env create -f environment.yml -n py38`

## Update env

`conda env export --no-builds > environment.yml`

## Relevant reading

1. [CDIPS-I: Methods](https://ui.adsabs.harvard.edu/abs/2019ApJS..245...13B/abstract)

2. [CDIPS-II: TOI-837b](https://ui.adsabs.harvard.edu/abs/2020AJ....160..239B/abstract)

3. [CDIPS-III: NGC-2516](https://ui.adsabs.harvard.edu/abs/2021AJ....162..197B/abstract)
