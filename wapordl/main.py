import logging
import os
from datetime import datetime
from string import ascii_lowercase, ascii_uppercase
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import requests
import shapely
from osgeo import gdal, gdalconst, ogr
from osgeo_utils import gdal_calc
from tqdm import tqdm

from wapordl.constants import AGERA5_VARS, L2_BB, L3_BBS, WAPOR3_VARS
from wapordl.zonalstats import (
    output_df_and_geometries_to_gpkg,
    retrieve_wapor_zonal_stats_as_dataframe,
    vector_file_to_dict,
)

gdal.UseExceptions()
logging.basicConfig(encoding="utf-8", level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    import rioxarray
    import xarray as xr

    use_xarray = True
    # raise ImportError
except ImportError:
    logging.info("Consider installing `xarray` and `rioxarray` for faster unit conversions.")
    use_xarray = False


def split_period_into_years(period):
    """
    Split a given period into a list of year tuples.

    Parameters
    ----------
    period : list[str]
        A list containing two strings representing the start and end dates in "YYYY-MM-DD" format.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples, where each tuple contains the start and end dates of each year in the specified period.
    """
    # Convert strings to datetime objects
    start_date = datetime.strptime(period[0], "%Y-%m-%d")
    end_date = datetime.strptime(period[1], "%Y-%m-%d")

    # Create a list of tuples for each year
    year_periods = []
    for year in range(start_date.year, end_date.year + 1):
        start_of_year = datetime(year, 1, 1)
        end_of_year = datetime(year, 12, 31)

        # Adjust the start and end dates based on the overall period
        if year == start_date.year:
            start_of_year = start_date
        if year == end_date.year:
            end_of_year = end_date

        year_periods.append((start_of_year.strftime("%Y-%m-%d"), end_of_year.strftime("%Y-%m-%d")))

    return year_periods


def reproject_vector(fh: str, epsg=4326) -> str:
    """Create a 2D GeoJSON file with `EPSG:4326` SRS from any
    OGR compatible vector file.

    Parameters
    ----------
    fh : str
        Path to input file.
    epsg : int, optional
        target SRS, by default 4326.

    Returns
    -------
    str
        Path to output (GeoJSON) file.
    """

    ext = os.path.splitext(fh)[-1]
    out_fh = fh.replace(ext, f"_reprojected.geojson")

    options = gdal.VectorTranslateOptions(
        dstSRS=f"EPSG:{epsg}",
        format="GeoJSON",
        dim="XY",
    )
    x = gdal.VectorTranslate(out_fh, fh, options=options)
    x.FlushCache()
    x = None

    return out_fh


def check_vector(fh: str) -> tuple:
    """Check if a provided vector file is correctly formatted for wapordl.

    Parameters
    ----------
    fh : str
        Path to input file.

    Returns
    -------
    tuple
        Information about the input file, first value is EPSG code (int), second is
        driver name, third is True if coordinates are 2D.
    """
    # with ogr.Open(fh) as ds: # NOTE does not work in gdal < 3.7, so not using
    # for backward compatability with Colab.
    ds = ogr.Open(fh)

    driver = ds.GetDriver()
    layer = ds.GetLayer()
    ftr = layer.GetNextFeature()
    geom = ftr.geometry()
    is_two_d = geom.CoordinateDimension() == 2
    spatialRef = layer.GetSpatialRef()
    epsg = spatialRef.GetAuthorityCode(None)

    try:
        ds = ds.Close()
    except AttributeError as e:
        if str(e) == "'DataSource' object has no attribute 'Close'":
            ds = ds.Release()
        else:
            raise e

    return int(epsg), getattr(driver, "name", None), is_two_d


def guess_l3_region(region_shape: shapely.Polygon) -> str:
    """Given a shapely.Polygon, determines the WaPOR level-3 region code (three letters)
    with which the given shape overlaps.

    Parameters
    ----------
    region_shape : shapely.Polygon
        Shape for which to search mathing level-3 code.

    Returns
    -------
    str
        WaPOR level-3 code.

    Raises
    ------
    ValueError
        Raised if no code can be found, i.e. the given shape doesn't overlap with any level-3 bounding-box.
    """

    checks = {x: shapely.Polygon(np.array(bb)).intersects(region_shape) for x, bb in L3_BBS.items()}
    number_of_results = sum(checks.values())
    if number_of_results == 0:
        added_regions = update_L3_BBS()
        l3_bbs = {x: L3_BBS[x] for x in added_regions}
        checks = {
            x: shapely.Polygon(np.array(bb)).intersects(region_shape) for x, bb in l3_bbs.items()
        }
        number_of_results = sum(checks.values())
        if number_of_results == 0:
            raise ValueError(f"`region` can't be linked to any L3 region.")  # NOTE: TESTED

    l3_regions = [k for k, v in checks.items() if v]
    l3_region = l3_regions[0]
    if number_of_results > 1:
        logging.warning(
            f"`region` intersects with multiple L3 regions ({l3_regions}), continuing with {l3_region} only."
        )
    else:
        logging.info(f"Given `region` matches with `{l3_region}` L3 region.")

    return l3_region


def collect_responses(url: str, info=["code"]) -> list:
    """Calls GISMGR2.0 API and collects responses.

    Parameters
    ----------
    url : str
        URL to get.
    info : list, optional
        Used to filter the response, set to `None` to keep everything, by default ["code"].

    Returns
    -------
    list
        The responses.
    """
    data = {"links": [{"rel": "next", "href": url}]}
    output = list()
    while "next" in [x["rel"] for x in data["links"]]:
        url_ = [x["href"] for x in data["links"] if x["rel"] == "next"][0]
        response = requests.get(url_)
        response.raise_for_status()
        data = response.json()["response"]
        if isinstance(info, list) and "items" in data.keys():
            output += [tuple(x.get(y) for y in info) for x in data["items"]]
        elif "items" in data.keys():
            output += data["items"]
        else:
            output.append(data)
    if isinstance(info, list):
        try:
            output = sorted(output)
        except TypeError:
            output = output
    return output


def date_func(url: str, tres: str) -> dict:
    """Determines start and end dates from a string a given temporal resolution, as well
    as the number of days between the two dates.

    Parameters
    ----------
    url : str
        URL linking to a resource.
    tres : str
        One of "E" (daily), "D" (dekadal), "M" (monthly), "A" (annual).

    Returns
    -------
    dict
        Dates and related information for a resource URL.

    Raises
    ------
    ValueError
        No valid `tres` given.
    """
    if tres == "D":
        if "AGERA5" in url:
            year_acc_dekad = os.path.split(url)[-1].split("_")[-1].split(".")[0]
            year = year_acc_dekad[:4]
            acc_dekad = int(year_acc_dekad[-2:])
            month = str((acc_dekad - 1) // 3 + 1).zfill(2)
            dekad = str((acc_dekad - 1) % 3 + 1)
        else:
            year, month, dekad = os.path.split(url)[-1].split(".")[-2].split("-")
        start_day = {"D1": "01", "D2": "11", "D3": "21", "1": "01", "2": "11", "3": "21"}[dekad]
        start_date = f"{year}-{month}-{start_day}"
        end_day = {
            "D1": "10",
            "D2": "20",
            "D3": pd.Timestamp(start_date).daysinmonth,
            "1": "10",
            "2": "20",
            "3": pd.Timestamp(start_date).daysinmonth,
        }[dekad]
        end_date = f"{year}-{month}-{end_day}"
    elif tres == "M":
        if "AGERA5" in url:
            year = os.path.split(url)[-1].split("_")[-1][:4]
            month = os.path.split(url)[-1].split("_")[-1][5:7]
        else:
            year, month = os.path.split(url)[-1].split(".")[-2].split("-")
        start_date = f"{year}-{month}-01"
        end_date = f"{year}-{month}-{pd.Timestamp(start_date).days_in_month}"
    elif tres == "A":
        if "AGERA5" in url:
            year = os.path.split(url)[-1].split("_")[-1][:4]
        else:
            year = os.path.split(url)[-1].split(".")[-2]
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
    elif tres == "E":
        if "AGERA5" in url:
            year = os.path.split(url)[-1].split("_")[-1][:4]
            month = os.path.split(url)[-1].split("_")[-1][4:6]
            start_day = os.path.split(url)[-1].split("_")[-1][6:8]
        else:
            year, month, start_day = os.path.split(url)[-1].split(".")[-2].split("-")
        start_date = end_date = f"{year}-{month}-{start_day}"
    else:
        raise ValueError("Invalid temporal resolution.")  # NOTE: TESTED

    number_of_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date) + pd.Timedelta(1, "D")).days

    date_md = {
        "start_date": start_date,
        "end_date": end_date,
        "number_of_days": number_of_days,
        "temporal_resolution": {"E": "Day", "D": "Dekad", "M": "Month", "A": "Year"}[tres],
    }

    if tres == "E":
        dekad = min(3, ((int(start_day) - 1) // 10) + 1)
        days_in_dekad = {1: 10, 2: 10, 3: pd.Timestamp(start_date).daysinmonth - 20}[dekad]
        date_md["days_in_dekad"] = days_in_dekad

    return date_md


def collect_metadata(variable: str) -> dict:
    """Queries `long_name`, `units` and `source` for a given WaPOR variable code.

    Parameters
    ----------
    variable : str
        Name of variable, e.g. `L3-AETI-D`.

    Returns
    -------
    dict
        Metadata for the variable.

    Raises
    ------
    ValueError
        No valid variable name given.
    """

    if variable in AGERA5_VARS.keys():
        return AGERA5_VARS[variable]

    if variable in WAPOR3_VARS.keys():
        return WAPOR3_VARS[variable]

    if "L1" in variable:
        base_url = f"https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3/mapsets"
    elif "L2" in variable:
        base_url = f"https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3/mapsets"
    elif "L3" in variable:
        base_url = f"https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3/mosaicsets"
    else:
        raise ValueError(f"Invalid variable name {variable}.")  # NOTE: TESTED
    info = ["code", "measureCaption", "measureUnit"]
    var_codes = {
        x[0]: {"long_name": x[1], "units": x[2]} for x in collect_responses(base_url, info=info)
    }

    return var_codes[variable]


def make_dekad_dates(period: list, max_date=None) -> list:
    """Make a list of dekadal timestamps between a start and end date.

    Parameters
    ----------
    period : list
        Start and end date in between which the dekadal timestamps will be generated.
    max_date : pd.Timestamp, optional
        Choose the earliest date between the end of `period` and `max_date`, by default None.

    Returns
    -------
    list
        Dekadal timestamps between the given start and end date.
    """
    period_ = [pd.Timestamp(x) for x in period]
    if isinstance(max_date, pd.Timestamp):
        period_[1] = min(period_[1], max_date)
    syear = period_[0].year
    smonth = period_[0].month
    eyear = period_[1].year
    emonth = period_[1].month
    x1 = pd.date_range(f"{syear}-{smonth}-01", f"{eyear}-{emonth}-01", freq="MS")
    x2 = x1 + pd.Timedelta("10 days")
    x3 = x1 + pd.Timedelta("20 days")
    x = np.sort(np.concatenate((x1, x2, x3)))
    x_filtered = [pd.Timestamp(x_) for x_ in x if x_ >= period_[0] and x_ < period_[1]]
    return x_filtered


def make_monthly_dates(period: list, max_date=None) -> list:
    """Make a list of monthly timestamps between a start and end date.

    Parameters
    ----------
    period : list
        Start and end date in between which the monthly timestamps will be generated.
    max_date : pd.Timestamp, optional
        Choose the earliest date between the end of `period` and `max_date`, by default None.

    Returns
    -------
    list
        Monthly timestamps between the given start and end date.
    """
    period_ = [pd.Timestamp(x) for x in period]
    period_[0] = pd.Timestamp(f"{period_[0].year}-{period_[0].month}-01")
    if isinstance(max_date, pd.Timestamp):
        period_[1] = min(period_[1], max_date)
    x1 = pd.date_range(period_[0], period_[1], freq="MS")
    x_filtered = [pd.Timestamp(x_) for x_ in x1]
    return x_filtered


def make_annual_dates(period: list, max_date=None) -> list:
    """Make a list of annual timestamps between a start and end date.

    Parameters
    ----------
    period : list
        Start and end date in between which the annual timestamps will be generated.
    max_date : pd.Timestamp, optional
        Choose the earliest date between the end of `period` and `max_date`, by default None.

    Returns
    -------
    list
        Annual timestamps between the given start and end date.
    """
    period_ = [pd.Timestamp(x) for x in period]
    period_[0] = pd.Timestamp(f"{period_[0].year}-01-01")
    if isinstance(max_date, pd.Timestamp):
        period_[1] = min(period_[1], max_date)
    x1 = pd.date_range(period_[0], period_[1], freq="YE-JAN")
    x_filtered = [pd.Timestamp(x_) for x_ in x1]
    return x_filtered


def make_daily_dates(period: list, max_date=None) -> list:
    """Make a list of daily timestamps between a start and end date.

    Parameters
    ----------
    period : list
        Start and end date in between which the daily timestamps will be generated.
    max_date : pd.Timestamp, optional
        Choose the earliest date between the end of `period` and `max_date`, by default None.

    Returns
    -------
    list
        Daily timestamps between the given start and end date.
    """
    period_ = [pd.Timestamp(x) for x in period]
    if isinstance(max_date, pd.Timestamp):
        period_[1] = min(period_[1], max_date)
    x1 = pd.date_range(period_[0], period_[1], freq="D")
    x_filtered = [pd.Timestamp(x_) for x_ in x1]
    return x_filtered


def generate_urls_agERA5(variable: str, period=None, check_urls=True) -> tuple:
    """Find resource URLs for an agERA5 variable for a specified period.

    Parameters
    ----------
    variable : str
        Name of the variable.
    period : list, optional
        Start and end date in between which resource URLs will be searched, by default None.
    check_urls : bool, optional
        Perform additional checks to test if the found URLs are valid, by default True.

    Returns
    -------
    tuple
        Resource URLs.

    Raises
    ------
    ValueError
        Invalid variable selected.
    ValueError
        Invalid temporal resolution.

    Notes
    -----
    https://data.apps.fao.org/static/data/index.html?prefix=static%2Fdata%2Fc3s%2FAGERA5_ET0
    """
    level, var_code, tres = variable.split("-")

    if variable not in AGERA5_VARS.keys():
        raise ValueError(f"Invalid variable `{variable}`, choose one from `{AGERA5_VARS.keys()}`.")

    max_date = pd.Timestamp.now() - pd.Timedelta(days=25)
    if isinstance(period, type(None)):
        period = ["1979-01-01", max_date.strftime("%Y-%m-%d")]

    base_url = f"https://data.apps.fao.org/static/data/c3s/{level}_{var_code}_{tres}"
    urls = list()
    if tres == "E":
        base_url = base_url[:-2]
        x_filtered = make_daily_dates(period, max_date=max_date)
        for x in x_filtered:
            url = os.path.join(base_url, f"{level}_{var_code}_{x.strftime('%Y%m%d')}.tif")
            urls.append(url)
    elif tres == "D":
        x_filtered = make_dekad_dates(period, max_date=max_date)
        for x in x_filtered:
            acc_dekad = (x.month - 1) * 3 + {1: 1, 11: 2, 21: 3}[x.day]
            url = os.path.join(base_url, f"{level}_{var_code}_{x.year}D{acc_dekad:>02}.tif")
            urls.append(url)
    elif tres == "M":
        x_filtered = make_monthly_dates(period, max_date=max_date)
        for x in x_filtered:
            url = os.path.join(base_url, f"{level}_{var_code}_{x.year}M{x.month:>02}.tif")
            urls.append(url)
    elif tres == "A":
        x_filtered = make_annual_dates(period, max_date=max_date)
        for x in x_filtered:
            url = os.path.join(base_url, f"{level}_{var_code}_{x.year}.tif")
            urls.append(url)
    else:
        raise ValueError(f"Invalid temporal resolution `{tres}`.")

    if check_urls:
        for url in urls.copy():
            try:
                x = requests.get(url, stream=True)
                x.raise_for_status()
            except requests.exceptions.HTTPError:
                logging.debug(f"Invalid url detected, removing `{url}`.")
                urls.remove(url)

    return tuple(sorted(urls))


def generate_urls_v3(
    variable: str, l3_region: Optional[str] = None, period: Optional[list[str]] = None
) -> tuple:
    """Find resource URLs for an agERA5 variable for a specified period.

    Parameters
    ----------
    variable : str
        Name of the variable.
    l3_region : _type_, optional
        Three letter code specifying the level-3 region, by default None.
    period : list, optional
        Start and end date in between which resource URLs will be searched, by default None.

    Returns
    -------
    tuple
        Resource URLs.

    Raises
    ------
    ValueError
        Invalid level selected.
    """

    level, _, tres = variable.split("-")

    if (level == "L1") or (level == "L2"):
        base_url = f"https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3/mapsets"
    elif level == "L3":
        base_url = f"https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3/mosaicsets"
    else:
        raise ValueError(f"Invalid level {level}.")  # NOTE: TESTED

    mapset_url = f"{base_url}/{variable}/rasters?filter="
    if not isinstance(l3_region, type(None)):
        mapset_url += f"code:CONTAINS:{l3_region};"
    if not isinstance(period, type(None)):
        mapset_url += f"time:OVERLAPS:{period[0]}:{period[1]};"

    urls = [x[0] for x in collect_responses(mapset_url, info=["downloadUrl"])]

    return tuple(sorted(urls))


def __make_band_names__(length):
    letters = [x for x in ascii_lowercase + ascii_uppercase]
    i = 2
    while len(letters) < length:
        for letter in letters[:52]:
            letters.append(letter * i)
        i += 1
    return letters[:length]


def unit_convertor(
    urls: list, in_fn: str, out_fn: str, unit_conversion: str, warp: gdal.Dataset, coptions=[]
) -> tuple:
    """Convert the units of multiple bands in a single geoTIFF file to another timescale.

    Parameters
    ----------
    urls : list
        Contains tuples of which the first item is a dictionary with metadata information for each band found in
        `in_fn`. Length of this list should be equal to the number of bands in `in_fn`.
    in_fn : str
        Path to geotiff file.
    out_fn : str
        Path to the to-be-created geotiff file.
    unit_conversion : str
        The desired temporal component of the converted units, should be one of
        "day", "dekad", "month" or "year".
    warp : gdal.Dataset
        The dataset to be adjusted, should point to `in_fn`.
    coptions : list, optional
        Extra creation options used to create `out_fn`, by default [].

    Returns
    -------
    tuple
        The new gdal.Dataset and the path to the created file.
    """

    global use_xarray

    input_files = dict()
    input_bands = dict()
    calc = list()
    should_convert = list()
    conversion_factors = list()
    letters = __make_band_names__(len(urls))

    if "AGERA5" in urls[0][1]:
        dtype = gdalconst.GDT_Float64
    else:
        dtype = gdalconst.GDT_Int32  # NOTE unit conversion can increase the DN's,
        # causing the data to not fit inside Int16 anymore...
        # so for now just moving up to Int32. Especially necessary
        # for NPP (which has a scale-factor of 0.001).

    for i, (md, _) in enumerate(urls):
        band_number = i + 1
        letter = letters[i]
        input_files[letter] = in_fn
        input_bands[f"{letter}_band"] = band_number
        if md.get("temporal_resolution", "unknown") == "Day":
            number_of_days = md.get("days_in_dekad", "unknown")
        else:
            number_of_days = md.get("number_of_days", "unknown")
        days_in_month = pd.Timestamp(md.get("start_date", "nat")).daysinmonth
        source_unit = md.get("units", "unknown")
        source_unit_split = source_unit.split("/")
        source_unit_q = "/".join(source_unit_split[:-1])
        source_unit_time = source_unit_split[-1]
        if any(
            [
                source_unit_time not in ["day", "month", "year", "dekad"],
                number_of_days == "unknown",
                source_unit == "unknown",
                pd.isnull(days_in_month),
            ]
        ):
            calc.append(f"{letter}.astype(numpy.float64)")
            md["units"] = source_unit
            md["units_conversion_factor"] = "N/A"
            md["original_units"] = "N/A"
            should_convert.append(False)
            conversion_factors.append(1)
        else:
            conversion = {
                ("day", "day"): 1,
                ("day", "dekad"): number_of_days,
                ("day", "month"): days_in_month,
                ("day", "year"): 365,
                ("dekad", "day"): 1 / number_of_days,
                ("dekad", "month"): 3,
                ("dekad", "year"): 36,
                ("dekad", "dekad"): 1,
                ("month", "day"): 1 / days_in_month,
                ("month", "dekad"): 1 / 3,
                ("month", "month"): 1,
                ("month", "year"): 12,
                ("year", "dekad"): 1 / 36,
                ("year", "day"): 1 / 365,
                ("year", "month"): 1 / 12,
                ("year", "year"): 1,
            }[(source_unit_time, unit_conversion)]
            calc.append(f"{letter}.astype(numpy.float64)*{conversion}")
            should_convert.append(True)
            conversion_factors.append(conversion)
            md["units"] = f"{source_unit_q}/{unit_conversion}"
            md["units_conversion_factor"] = conversion
            md["original_units"] = source_unit

    logging.debug(f"\ninput_files: {input_files}\ninput_bands: {input_bands}\ncalc: {calc}")

    conversion_is_one = [x["units_conversion_factor"] == 1.0 for x, _ in urls]

    # NOTE See todo just below.
    scales = [warp.GetRasterBand(i + 1).GetScale() for i in range(warp.RasterCount)]
    offsets = [warp.GetRasterBand(i + 1).GetOffset() for i in range(warp.RasterCount)]

    logging.debug(f"\nSCALES: {scales}\nOFFSETS: {offsets}")

    if all(should_convert) and not all(conversion_is_one):

        logging.info(
            f"Converting units from [{source_unit}] to [{source_unit_q}/{unit_conversion}] (use_xarray = {use_xarray})."
        )

        ndv = warp.GetRasterBand(1).GetNoDataValue()
        if use_xarray:
            ds = xr.open_dataset(in_fn, mask_and_scale=False, decode_coords="all")
            xr_conv = xr.DataArray(conversion_factors, coords={"band": ds["band"]})
            ndv_ = ds["band_data"].attrs["_FillValue"]

            da = xr.where(ds["band_data"] == ndv_, ndv_, ds["band_data"] * xr_conv)
            da = np.round(da, 0)

            ds_out = da.to_dataset("band")
            for i, (scale, (md, _)) in enumerate(zip(scales, urls)):
                ds_out[i + 1].attrs = md
                ds_out[i + 1] = ds_out[i + 1].rio.write_nodata(ndv)
                ds_out[i + 1].attrs["scale_factor"] = scale

            ds_out = ds_out.rio.write_crs(ds.rio.crs)
            ds_out.rio.to_raster(out_fn, compress="LZW", dtype={5: "int32", 7: "float64"}[dtype])
            filen = out_fn
        else:
            warp = gdal_calc.Calc(
                calc=calc,
                outfile=out_fn,
                overwrite=True,
                creation_options=coptions,
                quiet=True,
                type=dtype,
                NoDataValue=ndv,
                **input_files,
                **input_bands,
            )
            # TODO make bug report on GDAL for gdal_calc removing scale/offset factors
            for i, (scale, offset) in enumerate(zip(scales, offsets)):
                warp.GetRasterBand(i + 1).SetScale(scale)
                warp.GetRasterBand(i + 1).SetOffset(offset)
            warp.FlushCache()
            filen = out_fn
    else:
        if all(conversion_is_one):
            logging.info(f"Units are already as requested, no conversion needed.")
        else:
            logging.warning(
                f"Couldn't succesfully determine unit conversion factors, keeping original units."
            )
        for i, (md, _) in enumerate(urls):
            if md["units_conversion_factor"] != "N/A":
                md["units"] = md["original_units"]
                md["units_conversion_factor"] = f"N/A"
                md["original_units"] = "N/A"
        filen = in_fn

    return warp, filen


def l3_codes() -> dict:
    """Create an overview of the available WaPOR level-3 region codes.

    Returns
    -------
    dict
        keys are three letter region codes, values are the long names of the region.
    """
    mapset_url = "https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3/mosaicsets/L3-T-A/rasters?filter="
    x = collect_responses(mapset_url, info=["grid"])
    valids = {x_[0]["tile"]["code"]: x_[0]["tile"]["description"] for x_ in x}
    return valids


def l3_bounding_boxes(variable="L3-T-A", l3_region=None) -> dict:
    """Determine the bounding-boxes of the WaPOR level-3 regions.

    Parameters
    ----------
    variable : str, optional
        Name of the variable used to check the bounding-box, by default "L3-T-A".
    l3_region : str, optional
        Name of the level-3 region to check, when `None` will check all available level-3 regions, by default None.

    Returns
    -------
    dict
        keys are three letter region codes, values are the coordinates of the bounding-boxes.
    """
    urls = generate_urls_v3(variable, l3_region=l3_region, period=["2020-01-01", "2021-02-01"])
    l3_bbs = {}
    for region_code, url in zip([os.path.split(x)[-1].split(".")[-3] for x in urls], urls):
        info = gdal.Info("/vsicurl/" + url, format="json")
        bb = info["wgs84Extent"]["coordinates"][0]
        l3_bbs[region_code] = bb
    return l3_bbs


def update_L3_BBS():
    logging.info("Updating L3 bounding-boxes.")
    all_l3_regions = l3_codes()
    new_regions = set(all_l3_regions.keys()).difference(set(L3_BBS.keys()))
    added_regions = list()
    for l3_region in new_regions:
        new_bb = l3_bounding_boxes(l3_region=l3_region).get(l3_region, None)
        if not isinstance(new_bb, type(None)):
            added_regions.append(l3_region)
            L3_BBS[l3_region] = new_bb
    return added_regions


def cog_dl(
    urls: list,
    out_fn: str,
    overview: Optional[str] = None,
    warp_kwargs={},
    vrt_options={"separate": True},
    unit_conversion: Optional[str] = None,
) -> tuple:
    """Download multiple COGs into the bands of a single geotif or netcdf file.

    Parameters
    ----------
    urls : list
        URLs of the different COGs to be downloaded.
    out_fn : str
        Path to the output file.
    overview : str, optional
        Select which overview from the COGs to use, by default None.
    warp_kwargs : dict, optional
        Additional gdal.Warp keyword arguments, by default {}.
    vrt_options : dict, optional
        Additional options passed to gdal.BuildVRT, by default {"separate": True}.
    unit_conversion : str, optional
        Apply a unit conversion on the created file, can be one of None, "day", "dekad",
        "month" or "year", by default None.

    Returns
    -------
    tuple
        Paths to the created geotiff file and the (intermediate) vrt file.

    Raises
    ------
    ValueError
        Invalid output extension selected.
    """

    out_ext = os.path.splitext(out_fn)[-1]
    valid_ext = {".nc": "netCDF", ".tif": "GTiff"}
    valid_cos = {".nc": ["COMPRESS=DEFLATE", "FORMAT=NC4C"], ".tif": ["COMPRESS=LZW"]}
    if not bool(np.isin(out_ext, list(valid_ext.keys()))):
        raise ValueError(
            f"Please use one of {list(valid_ext.keys())} as extension for `out_fn`, not {out_ext}"
        )  # NOTE: TESTED
    vrt_fn = out_fn.replace(out_ext, ".vrt")

    ## Build VRT with all the required data.
    vrt_options_ = gdal.BuildVRTOptions(**vrt_options)
    prepend = {False: "/vsicurl/", True: "/vsigzip//vsicurl/"}
    vrt = gdal.BuildVRT(vrt_fn, [prepend[".gz" in x[1]] + x[1] for x in urls], options=vrt_options_)
    vrt.FlushCache()

    n_urls = len(urls)

    # Create waitbar.
    waitbar = tqdm(
        desc=f"Downloading {n_urls} COGs", leave=False, total=100, bar_format="{l_bar}{bar}|"
    )

    # Define callback function for waitbar progress.
    def _callback_func(info, *args):
        waitbar.update(info * 100 - waitbar.n)

    ## Download the data.
    warp_options = gdal.WarpOptions(
        format=valid_ext[out_ext],
        cropToCutline=True,
        overviewLevel=overview,
        multithread=True,
        targetAlignedPixels=True,
        creationOptions=valid_cos[out_ext],
        callback=_callback_func,
        **warp_kwargs,
    )
    warp = gdal.Warp(out_fn, vrt_fn, options=warp_options)
    warp.FlushCache()  # NOTE do not remove this.
    waitbar.close()
    nbands = warp.RasterCount

    if nbands == n_urls and unit_conversion is not None:
        out_fn_new = out_fn.replace(out_ext, f"_converted{out_ext}")
        out_fn_old = out_fn
        warp, out_fn = unit_convertor(
            urls, out_fn, out_fn_new, unit_conversion, warp, coptions=valid_cos[out_ext]
        )
    else:
        out_fn_old = ""

    if nbands == n_urls:
        for i, (md, _) in enumerate(urls):
            if not isinstance(md, type(None)):
                band = warp.GetRasterBand(i + 1)
                band.SetMetadata(md)

    warp.FlushCache()

    if os.path.isfile(vrt_fn):
        try:
            os.remove(vrt_fn)
        except PermissionError:
            ...

    if os.path.isfile(out_fn_old) and os.path.isfile(out_fn_new):
        try:
            os.remove(out_fn_old)
        except PermissionError:
            ...

    return out_fn, vrt_fn


def wapor_geojson_prep(file_path: str):
    # Check if vector file is in good shape.
    epsg, driver, is_two_d = check_vector(file_path)
    if not np.all([epsg == 4326, driver == "GeoJSON", is_two_d]):
        ext_ = os.path.splitext(file_path)[-1]
        fn_ = os.path.split(file_path)[-1]
        out_fn_ = fn_.replace(ext_, "_reprojected.geojson")
        dim_ = {True: "2D", False: "3D"}[is_two_d]
        logging.warning(
            f"Reprojecting `{fn_}` [EPSG:{epsg}, {dim_}] to `{out_fn_}` [EPSG:4326, 2D]."
        )
        file_path = reproject_vector(file_path, epsg=4326)

    region_code = os.path.split(file_path)[-1].replace(".geojson", "")
    return region_code, file_path


def wapor_dl(
    variable: str,
    region: Optional[Union[str, List[float]]] = None,
    period=["2021-01-01", "2022-01-01"],
    overview: Optional[str] = None,
    unit_conversion: Optional[str] = None,
    req_stats=["minimum", "maximum", "mean"],
    folder: Optional[str] = None,
    filename: Optional[str] = None,
    skip_if_exists: bool = False,
) -> Union[str, pd.DataFrame]:
    """Download a WaPOR or agERA5 variable for a specified region and period.

    Parameters
    ----------
    region : Union[str, List[float], None]
        Defines the area of interest. Can be a three letter code to describe a WaPOR level-3 region,
        a path to a vector file or a list of 4 floats, specifying a bounding box.
    variable : str
        Name of the variable to download.
    period : list, optional
        Period for which to download data, by default ["2021-01-01", "2022-01-01"].
    overview : str, optional
        Which overview of the COGs to use, by default None.
    unit_conversion : str, optional
        Apply a unit conversion on the created file, can be one of None, "day", "dekad",
        "month" or "year", by default None.
    req_stats : list, optional
        When set to `None` the function returns a path to a created file, otherwise
        it return a pd.Dataframe with the requested statistics, by default ["minimum", "maximum", "mean"].
    folder : str, optional
        Path to a folder in which to save any (intermediate) files. If set to `None`, everything will be
        kept in memory, by default None.
    filename : str, optional
        Set a different name for the output file.
    skip_if_exists: bool, optional
        if the file to be made already exists skips it, useful for repitition of the waporzonal process

    Returns
    -------
    Union[str, pd.DataFrame]
        Return a path to a file (if `req_stats` is `None`) or a pd.Dataframe if req_stats is a list
        speciyfing statistics.
    """

    global L3_BBS

    ## Retrieve info from variable name.
    level, var_code, tres = variable.split("-")

    ## Check if region is valid.
    # L3-CODE
    if all([isinstance(region, str), len(region) == 3]):

        if not region == region.upper():
            raise ValueError(
                f"Invalid region code `{region}`, region codes have three capitalized letters."
            )

        if region not in list(L3_BBS.keys()):
            logging.info(f"Searching bounding-box for `{region}`.")
            bb = l3_bounding_boxes(l3_region=region)
            if len(bb) == 0:
                raise ValueError(f"Unkown L3 region `{region}`.")
            else:
                logging.info(f"Bounding-box found for `{region}`.")
                L3_BBS = {**L3_BBS, **bb}

        if level == "L3":
            l3_region = region[:]  # three letter code to filter L3 datasets in GISMGR2.
            region = None  # used for clipping, can be None, list(bb) or path/to/file.geojson.
            region_code = l3_region[:]  # string to name the region in filenames etc.
            region_shape = None  # polygon used to check if there is data for the region.
        else:
            l3_region = None
            region_shape = shapely.Polygon(np.array(L3_BBS[region]))
            region_code = region[:]
            region = list(region_shape.bounds)
    # GEOJSON
    elif isinstance(region, str):
        if not os.path.isfile(region):
            raise ValueError(f"Geojson file not found.")  # NOTE: TESTED
        else:
            region_code, region = wapor_geojson_prep(region)
            # Open the geojson.
            with open(region, "r", encoding="utf-8") as f:
                region_shape = shapely.from_geojson(f.read())

        l3_region = None
    # BB
    elif isinstance(region, list):
        if not all([region[2] > region[0], region[3] > region[1]]):
            raise ValueError(f"Invalid bounding box.")  # NOTE: TESTED
        else:
            region_code = "bb"
            region_shape = shapely.Polygon(
                [
                    (region[0], region[1]),
                    (region[2], region[1]),
                    (region[2], region[3]),
                    (region[0], region[3]),
                    (region[0], region[1]),
                ]
            )
        l3_region = None
    else:
        raise ValueError(f"Invalid value for region ({region}).")  # NOTE: TESTED

    ## Check l3_region code.
    if level == "L3" and isinstance(l3_region, type(None)):
        l3_region = guess_l3_region(region_shape)
        region_code += f".{l3_region}"

    ## Check the dates in period.
    if not isinstance(period, type(None)):
        period = [pd.Timestamp(x) for x in period]
        if period[0] > period[1]:
            raise ValueError(f"Invalid period.")  # NOTE: TESTED
        period = [x.strftime("%Y-%m-%d") for x in period]

    ## Collect urls for requested variable.
    if "AGERA5" in variable:
        urls = generate_urls_agERA5(variable, period=period)
    else:
        urls = generate_urls_v3(variable, l3_region=l3_region, period=period)

    if len(urls) == 0:
        raise ValueError("No files found for selected region, variable and period.")  # NOTE: TESTED

    ## Determine date for each url.
    md = collect_metadata(variable)
    md["overview"] = overview
    md_urls = [({**date_func(url, tres), **md}, url) for url in urls]

    logging.info(f"Found {len(md_urls)} files for {variable}.")

    ## Determine required output resolution.
    # NOTE maybe move this to external function (assumes info the same for all urls)
    info_url = md_urls[0][1]
    info_url = {False: "/vsicurl/", True: "/vsigzip//vsicurl/"}[".gz" in info_url] + info_url
    info = gdal.Info(info_url, format="json")

    if overview is None:
        overview = -1

    xres, yres = info["geoTransform"][1::4]
    warp_kwargs = {
        "xRes": abs(xres) * 2 ** (overview + 1),
        "yRes": abs(yres) * 2 ** (overview + 1),
    }

    if isinstance(region, list):
        warp_kwargs["outputBounds"] = region
        warp_kwargs["outputBoundsSRS"] = "epsg:4326"
    elif isinstance(region, str):
        warp_kwargs["cutlineDSName"] = region
    else:
        ...

    ## Check if region overlaps with datasets bounding-box.
    if not isinstance(region_shape, type(None)) and level != "AGERA5":
        if level == "L2":
            data_bb = shapely.from_geojson(L2_BB)
        else:
            data_bb = shapely.Polygon(np.array(info["wgs84Extent"]["coordinates"])[0])

        if not data_bb.intersects(region_shape):
            info_lbl1 = region_code if region_code != "bb" else str(region)
            info_lbl2 = variable if isinstance(l3_region, type(None)) else f"{variable}.{l3_region}"
            raise ValueError(
                f"Selected region ({info_lbl1}) has no overlap with the datasets ({info_lbl2}) bounding-box."
            )

    ## Get scale and offset factor.
    scale = info["bands"][0].get("scale", 1)
    offset = info["bands"][0].get("offset", 0)

    ## Check offset factor.
    if offset != 0:
        logging.warning("Offset factor is not zero, statistics might be wrong.")

    if folder:
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if not isinstance(filename, type(None)):
            warp_fn = os.path.join(folder, f"{filename}.tif")
        else:
            warp_fn = os.path.join(
                folder,
                f"{region_code}_{variable}_{'_'.join(period)}_{overview}_{unit_conversion}.tif",
            )
    else:
        warp_fn = f"/vsimem/{pd.Timestamp.now()}_{region_code}_{variable}_{overview}_{unit_conversion}.tif"

    if skip_if_exists and os.path.exists(warp_fn):
        logging.info(f"skip if exists set to true and existing file found, skipping: {warp_fn}")
        return warp_fn

    warp_fn, vrt_fn = cog_dl(
        md_urls,
        warp_fn,
        overview=overview,
        warp_kwargs=warp_kwargs,
        unit_conversion=unit_conversion,
    )

    ## Collect the stats into a pd.Dataframe if necessary.
    if req_stats is not None:
        stats = gdal.Info(warp_fn, format="json", stats=True)
        data = {
            statistic: [x.get(statistic, np.nan) for x in stats["bands"]] for statistic in req_stats
        }
        data = pd.DataFrame(data) * scale
        data["start_date"] = [
            pd.Timestamp(x.get("metadata", {}).get("", {}).get("start_date", "nat"))
            for x in stats["bands"]
        ]
        data["end_date"] = [
            pd.Timestamp(x.get("metadata", {}).get("", {}).get("end_date", "nat"))
            for x in stats["bands"]
        ]
        data["number_of_days"] = [
            pd.Timedelta(
                float(x.get("metadata", {}).get("", {}).get("number_of_days", np.nan)), "days"
            )
            for x in stats["bands"]
        ]
        out_md = {
            k: v
            for k, v in md_urls[0][0].items()
            if k in ["long_name", "units", "overview", "original_units"]
        }
        data.attrs = out_md
    else:
        data = warp_fn

    ## Unlink memory files.
    if "/vsimem/" in vrt_fn:
        _ = gdal.Unlink(vrt_fn)
    if "/vsimem/" in warp_fn:
        _ = gdal.Unlink(warp_fn)

    return data


def wapor_map(
    variable: str,
    period: list,
    folder: str,
    region: Optional[Union[str, List[float]]] = None,
    unit_conversion: Optional[str] = None,
    overview: Optional[str] = None,
    extension=".tif",
    separate_unscale=False,
    filename: Optional[str] = None,
) -> str:
    """Download a map of a WaPOR3 or agERA5 variable for a specified region and period.

    Parameters
    ----------
    region : Union[str, List[float]]
        Defines the area of interest. Can be a three letter code to describe a WaPOR level-3 region,
        a path to a vector file or a list of 4 floats, specifying a bounding box.
    variable : str
        Name of the variable to download.
    period : list
        Period for which to download data.
    folder : str
        Folder into which to download the data.
    unit_conversion : str, optional
        Apply a unit conversion on the created file, can be one of None, "day", "dekad",
        "month" or "year", by default None.
    overview : str, optional
        Which overview of the COGs to use, by default None.
    extension : str, optional
        One of ".tif" or ".nc", controls output format, by default ".tif".
    separate_unscale : bool, optional
        Set to `True` to create single band geotif files instead of a single geotif with multiple bands,
        does not do anything when extension is set to ".nc" , by default False.
    filename : str, optional
        Set a different name for the output file.

    Returns
    -------
    str
        Path to output file.
    """

    ## Check if raw-data will be downloaded.
    if overview is not None:
        logging.warning("Downloading an overview instead of original data.")

    ## Check if a valid path to download into has been defined.
    if not os.path.isdir(folder):
        os.makedirs(folder)

    valid_units = [None, "dekad", "day", "month", "year"]
    if not unit_conversion in valid_units:
        raise ValueError(
            f"Please select one of {valid_units} instead of {unit_conversion}."
        )  # NOTE: TESTED

    ## Call wapor_dl to create a GeoTIFF.
    fp = wapor_dl(
        region=region,
        variable=variable,
        folder=folder,
        period=period,
        overview=overview,
        unit_conversion=unit_conversion,
        req_stats=None,
        filename=filename,
    )

    if extension == ".tif" and separate_unscale:
        logging.info("Splitting single GeoTIFF into multiple unscaled files.")
        folder = os.path.split(fp)[0]
        ds = gdal.Open(fp)
        number_of_bands = ds.RasterCount
        fps = list()
        for band_number in range(1, number_of_bands + 1):
            band = ds.GetRasterBand(band_number)
            md = band.GetMetadata()
            options = gdal.TranslateOptions(
                unscale=True,
                outputType=gdalconst.GDT_Float64,
                bandList=[band_number],
                creationOptions=["COMPRESS=LZW"],
            )
            output_file = fp.replace(".tif", f"_{md['start_date']}.tif")
            x = gdal.Translate(output_file, fp, options=options)
            x.FlushCache()
            fps.append(output_file)
        ds.FlushCache()
        ds = None
        try:
            os.remove(fp)
        except PermissionError:
            ...
        return fps
    elif extension != ".tif":
        if separate_unscale:
            logging.warning(
                f"The `separate_unscale` option only works with `.tif` extension, not with `{extension}`."
            )
        logging.info(f"Converting from `.tif` to `{extension}`.")
        toptions = {".nc": {"creationOptions": ["COMPRESS=DEFLATE", "FORMAT=NC4C"]}}
        options = gdal.TranslateOptions(**toptions.get(extension, {}))
        new_fp = fp.replace(".tif", extension)
        ds = gdal.Translate(new_fp, fp, options=options)
        ds.FlushCache()
        try:
            os.remove(fp)
        except PermissionError:
            ...
        return new_fp
    else:
        return fp


def wapor_ts(
    variable: str,
    period: list,
    overview: Union[str, int],
    region: Union[str, List[float], None],
    unit_conversion: Optional[str] = None,
    req_stats=["minimum", "maximum", "mean"],
) -> pd.DataFrame:
    """Download a timeseries of a WaPOR3 or agERA5 variable for a specified region and period.

    Parameters
    ----------
    region : Union[str, List[float], None]
        Defines the area of interest. Can be a three letter code to describe a WaPOR level-3 region,
        a path to a vector file or a list of 4 floats, specifying a bounding box.
    variable : str
        Name of the variable to download.
    period : list
        Period for which to download data.
    overview : Union[str, int]
        Which overview of the COGs to use, by default None.
    unit_conversion : str, optional
        Apply a unit conversion on the created file, can be one of None, "day", "dekad",
        "month" or "year", by default None.
    req_stats : list, optional
        Specify which statistics to include in the output, by default ["minimum", "maximum", "mean"].

    Returns
    -------
    pd.DataFrame
        Timeseries output.
    """

    valid_units = [None, "dekad", "day", "month", "year"]
    if not unit_conversion in valid_units:
        raise ValueError(
            f"Please select one of {valid_units} instead of {unit_conversion}."
        )  # NOTE: TESTED

    ## Check if valid statistics have been selected.
    if not isinstance(req_stats, list):
        raise ValueError("Please specify a list of required statistics.")  # NOTE: TESTED
    valid_stats = np.isin(req_stats, ["minimum", "maximum", "mean"])
    req_stats = np.array(req_stats)[valid_stats].tolist()
    if len(req_stats) == 0:
        raise ValueError(
            f"Please select at least one valid statistic from {valid_stats}."
        )  # NOTE: TESTED
    if False in valid_stats:
        logging.warning(f"Invalid statistics detected, continuing with `{', '.join(req_stats)}`.")

    ## Call wapor_dl to create a timeseries.
    df = wapor_dl(
        region=region,
        variable=variable,
        period=period,
        overview=overview,
        req_stats=req_stats,
        unit_conversion=unit_conversion,
        folder=None,
    )

    return df


def wapor_zonal(
    target_polygons: str,
    folder: str,
    id_column: str,
    variables: list[str],
    period: list,
    overview: Optional[Union[str, int]] = None,
    unit_conversion: Optional[str] = None,
    req_stats=["mean", "std", "maximum", "minimum"],
    skip_if_exists: bool = False,
    split_by_year: bool = False,
    output_gpkg: bool = False,
) -> Union[pd.DataFrame, None]:
    """Download zonal statistics for a set of WaPOR3 or agERA5 variables for a given period.

    Parameters
    ----------
    target_polygons : Union[str, List[float], None]
        Defines the areas/polygons of interest. a path to a vector file
    id_column: str
        column name in the target_polygons file to identify polygons with
    variable : str
        Name of the variable to download.
    period : list
        Period for which to download data.
    folder : str
        Folder into which to download the data.
    overview : Union[str, int], optional
        Which overview of the COGs to use, by default None.
    unit_conversion : str, optional
        Apply a unit conversion on the created file, can be one of  "day", "dekad",
        "month" or "year", by default None.
    stats : list, optional
        Specify which statistics to include in the output, by default ["std", "mean"].
    skip_if_exists: bool, optional
        if the file to be made already exists skips it, useful for repitition of the waporzonal process
    output_gpkg: bool , optional
        if true output a gpkg of the zonal stats retrieved, by default false
    split_by_year: bool , optional
        if true splits the retrieved data by year during retrieval, by default false

    Returns
    -------
    Union[pd.DataFrame, None]
        zonal statistics timeseries output if succesful otherwise None.
    """

    valid_units = [None, "dekad", "day", "month", "year"]
    if not unit_conversion in valid_units:
        raise ValueError(
            f"Please select one of {valid_units} instead of {unit_conversion}."
        )  # NOTE: TESTED

    ## Check if valid statistics have been selected.
    if not isinstance(req_stats, list):
        raise ValueError("Please specify a list of required statistics.")  # NOTE: TESTED
    valid_stats = np.isin(req_stats, ["mean", "std", "maximum", "minimum"])
    req_stats = np.array(req_stats)[valid_stats].tolist()
    if len(req_stats) == 0:
        raise ValueError(
            f"Please select at least one valid statistic from {valid_stats}."
        )  # NOTE: TESTED
    if False in valid_stats:
        logging.warning(f"Invalid statistics detected, continuing with `{', '.join(req_stats)}`.")

    region_code, target_polygons = wapor_geojson_prep(file_path=target_polygons)

    geometries = vector_file_to_dict(vector_file_path=target_polygons, column=id_column)

    if split_by_year:
        periods = split_period_into_years(period=period)
    else:
        periods = [period]

    variable_dfs = []

    for _variable in variables:
        yearly_variable_dfs = []
        for _period in periods:
            ## Call wapor_dl to create a raster stack
            fp = wapor_dl(
                region=target_polygons,
                variable=_variable,
                period=_period,
                folder=folder,
                overview=overview,
                req_stats=None,
                unit_conversion=unit_conversion,
                skip_if_exists=skip_if_exists,
            )
            logging.info(f"{_variable} downloaded for {_period}, carrying out zonal statistics")

            product_df = retrieve_wapor_zonal_stats_as_dataframe(
                extraction_tiff=fp,
                polygons=geometries,
                calculations=req_stats,
                _id=id_column,
                product=_variable,
            )

            yearly_variable_dfs.append(product_df)

            product_df.to_csv(
                os.path.join(folder, f"{region_code}_{_variable}_{'_'.join(_period)}.csv")
            )

        if yearly_variable_dfs:
            variable_zonal_stats_df = pd.concat(yearly_variable_dfs, axis=0, ignore_index=True)
        else:
            variable_zonal_stats_df = None

        variable_dfs.append(variable_zonal_stats_df)

        variable_zonal_stats_df.to_csv(os.path.join(folder, f"{region_code}_{_variable}.csv"))

    if variable_dfs:
        zonal_stats_df = pd.concat(variable_dfs, axis=0, ignore_index=True)
    else:
        zonal_stats_df = None

    zonal_stats_df.to_csv(os.path.join(folder, f"{region_code}.csv"))

    if output_gpkg:
        output_zonal_gpkg = os.path.join(folder, f"{region_code}.gpkg")
        output_df_and_geometries_to_gpkg(
            vector_file_path=target_polygons,
            df=zonal_stats_df,
            id_col=id_column,
            output_path=output_zonal_gpkg,
            group_columns=["product", "date"],
            value_columns=req_stats,
        )

    return zonal_stats_df
