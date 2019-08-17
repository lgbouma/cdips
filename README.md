# `cdips`
Cluster Difference Imaging Photometric Survey

Description
----------

This repository contains documentation, code, and small chunks of data
supporting the CDIPS project. 
Most of these tools can be understood as answers to the question "what do you
do once the [cdips-pipeline](https://github.com/waqasbhatti/cdips-pipeline)
gives you light curves?" Directory contents are as follows.

```
/cdips/ python package with tools to make target catalogs, process light
curves, optimize cdips-pipeline parameters, plot and analyze light curves.

/data/ not much on github -- only small miscellanea

/doc/ greppable notes

/drivers/ interface to make target catalogs, find planet candidates and variable
stars, classify, vet the classifications, and make plots and tables for papers.

/paper_I/ CDIPS method paper.

/long_paper/ deprecated iteration of the CDIPS method paper when its scope was
"light curves + planets" rather than "light curves".

/tests/ a few pytestable ideas (but pytest & CI are not set up for this repo).
```

Author
----------
Luke Bouma

License
----------
MIT

Install
----------
`conda env create -f environment.yml -n cdips`

Update env
----------
`conda env export > environment.yml`
