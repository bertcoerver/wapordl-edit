# wapordl-edit

Clone of the original wapordl repo with zonal statistics functionality added

## IMPORTANT: THIS IS A COPY OF THE REPO MADE BY BERT COEVER OF FAO: https://bitbucket.org/cioapps/wapordl/ 

--------------

## WaPORDL

Original wapordl description: Download data from the WaPOR3 dataset as spatially aggregated timeseries or as spatial data clipped to a bounding-box or shapefile.

--------------

wapordl is a great little package that provides easy access to wapor v3 data via python. Specificlaly it supports the quick and easy downloading of rasters and time series

It can be accessed via the link below as well as directly (and much mor easily) via conda or pip:

    conda install  -c conda-forge wapordl

If your interested in carrying out zonal statistics (calculating field based statistics) you can use wapordl for that as well (since version 1.1), check out its README and look for the `identifier` keyword example.

--------------

Again all credits go to Bert Coever and the FAO for building a great package. This repo thus builds on the work of Bert Coever and the FAO.

If you do not need zonal statistics I reccomend utilising the original wapordl repo via conda as it is a much easier install

[![HitCount](https://hits.dwyl.com/operations@eleafcom/eLEAF-Github/wapordl-edit.svg?style=flat-square)](http://hits.dwyl.com/operations@eleafcom/eLEAF-Github/wapordl-edit) 

## Authors and acknowledgment
Development: Eleaf and Roeland de Koning
Original Code: FAO and Bert Coever

## License
open source


## Release Notes

#### 0.1 2024/10/31

    - bug fix unscale and unoffset missing from wapor_zonal process, added in. As long as it is inbedded in the gdla metadata data is unscaled before carrying out zonal statistics process
    - started adding release notes

##### 0.1 2024/11/25
    - speed up zonal stats for large polygons (large number of cells)
