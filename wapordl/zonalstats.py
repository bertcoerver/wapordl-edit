import math
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from osgeo import gdal, gdalconst, ogr, osr
from pydantic import BaseModel


class GeotransformPlus(BaseModel):
    """
    Pydantic model for storing GDAL geotransform information and dimensions.

    Attributes
    ----------
    geotransform : tuple
        A tuple of six floats representing the GDAL geotransform.
            (0) upper-left x-coordinate
            (1) pixel width (xres)
            (2) row rotation
            (3) upper-left y-coordinate
            (4) column rotation
            (5) negative pixel height (yres)
    width : int
        The number of pixels in the x dimension.
    height : int
        The number of pixels in the y dimension.
    """

    geotransform: Tuple[float, float, float, float, float, float]
    width: int
    height: int


####################################
def retrieve_wapor_zonal_stats_as_dataframe(
    polygons: dict[str, ogr.Geometry],
    extraction_tiff: str,
    product: str,
    _id: str,
    calculations: list[str] = ["mean", "std", "maximum", "minimum"],
) -> pd.DataFrame:

    with gdal.Open(extraction_tiff, gdalconst.GA_ReadOnly) as temp_data:
        _dates = get_end_dates_from_bands(temp_data)
    zonal_stats_values = time_series_extraction_polygon(
        extraction_tiff=extraction_tiff, polygons=polygons, calculations=calculations
    )

    df_columns = ["product", _id, "date"]
    df_columns.extend(calculations)

    data = {col: [] for col in df_columns}

    # Iterate through zonal_stats_values dictionary
    for main_id, sub_dict in zonal_stats_values.items():
        if isinstance(sub_dict, dict):
            num_entries = len(next(iter(sub_dict.values())))  # Number of entries in sub_dict

            # Add fixed values once per dict entry
            data[_id].extend([main_id] * num_entries)
            data["product"].extend([product] * num_entries)

            # Repeat _dates list to match the number of sub-dict entries
            data["date"].extend(_dates[:num_entries])

            # Add sub_dict values for columns that exist in df_columns
            for key, values in sub_dict.items():
                if key in df_columns:
                    data[key].extend(values)

    # Construct the DataFrame
    zonal_stats_df = pd.DataFrame(data, columns=df_columns) if data[_id] else None

    return zonal_stats_df


####################################
def time_series_extraction_polygon(
    polygons: dict[str, ogr.Geometry],
    extraction_tiff: str,
    calculations: list[str] = ["mean", "std", "maximum", "minimum"],
    lower_limit: Optional[int] = None,
) -> dict[str, Union[dict[str, list[float]], str]]:

    stack_dataset = gdal.Open(extraction_tiff, gdalconst.GA_ReadOnly)

    geotransform_plus = GeotransformPlus(
        geotransform=stack_dataset.GetGeoTransform(),
        width=int(stack_dataset.RasterXSize),
        height=int(stack_dataset.RasterYSize),
    )

    if stack_dataset is None:
        raise FileNotFoundError(f"Could not open stack GeoTIFF: {extraction_tiff}")

    final_results: Dict[str, Union[list[float], str]] = {}
    for _id, polygon in polygons.items():
        result = extract_polygon_weighted_zonal_stats_series_from_raster_stack(
            stack_dataset=stack_dataset,
            stack_geotransform_plus=geotransform_plus,
            polygon=polygon,
            polygon_id=_id,
            calculations=calculations,
            lower_limit=lower_limit,
        )
        final_results[result[0]] = result[1]

    return final_results


####################################
def generate_geometry_mask_array(
    geometry: ogr.Geometry,
    geotransform: Tuple[float, float, float, float, float, float],
    array_coord_bbox: tuple[float, float, float, float],
    array_pixel_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Generate a mask array only for the portion of the raster overlapping the geometry.

    Parameters
    ----------
    geometry : ogr.Geometry
        The input geometry object (polygon).
    geotransform : tuple
        Original geotransform from the raster (6 elements).
    array_coord_bbox : tuple[float, float, float, float]
        Coordinates of the bounding box (xmin, ymin, xmax, ymax) in spatial reference.
    array_pixel_bbox : tuple[int, int, int, int]
        Bounding box in pixel space ((min_row, min_col, max_row, max_col)).

    Returns
    -------
    mask : np.ndarray
        Mask array where 1 indicates geometry presence and 0 indicates absence,
        restricted to the bounding box.
    """

    # Extract pixel bbox
    min_row, min_col, max_row, max_col = array_pixel_bbox

    # Define the size of the mask based on the bbox
    mask_height = max_row - min_row
    mask_width = max_col - min_col
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    if mask_height > 1:
        pass

    # Calculate bbox geotransform based on the array_coord_bbox and geotransform
    xmin, ymin, xmax, ymax = array_coord_bbox
    bbox_geotransform = (
        xmin,
        geotransform[1],
        geotransform[2],
        ymax,
        geotransform[4],
        geotransform[5],
    )

    # Create an empty raster (target) just for the bbox
    mem_driver = gdal.GetDriverByName("MEM")
    target_ds = mem_driver.Create("", mask_width, mask_height, 1, gdal.GDT_Byte)

    # Set the adjusted geotransform for the bbox
    target_ds.SetGeoTransform(bbox_geotransform)

    # Create an in-memory OGR layer
    vector_mem_driver = ogr.GetDriverByName("Memory")
    vector_mem_ds = vector_mem_driver.CreateDataSource("wrk")
    vector_mem_layer = vector_mem_ds.CreateLayer("wrk", None, ogr.wkbPolygon)
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326)
    vector_mem_layer.SetProjection(spatial_ref.ExportToWkt())

    # Add the geometry to the layer
    feature_defn = vector_mem_layer.GetLayerDefn()
    feature = ogr.Feature(feature_defn)
    feature.SetGeometry(geometry)
    vector_mem_layer.CreateFeature(feature)

    # Rasterize the geometry onto the target raster
    gdal.RasterizeLayer(
        target_ds, [1], vector_mem_layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"]
    )

    # Read the rasterized mask into the numpy array
    mask[:, :] = target_ds.GetRasterBand(1).ReadAsArray()

    return mask


####################################
def extract_polygon_weighted_zonal_stats_series_from_raster_stack(
    stack_dataset: gdal.Dataset,
    stack_geotransform_plus: GeotransformPlus,
    polygon: ogr.Geometry,
    polygon_id: str,
    calculations: list[str] = ["mean", "std"],
    lower_limit: Optional[int] = None,
) -> tuple[str, Union[dict[str, list[float]], str]]:
    """
    Extract weighted mean or std values from a stack GeoTIFF per band for the cells that fall
    within the provided polygon.

    old_name: extract_polygon_weighted_zonal_stats_series_from_raster_stack

    Parameters:
        stack_dataset: Opened stack GeoTIFF dataset.
        stack_geotransform (tuple[float, float, float, float, float, float]): geotransform of
        the stack, passed seperately to avoid multiple I/O ops
        stack_width (int): width of the stack, passed seperately to avoid multiple I/O ops
        stack_height (int): height of the stack, passed seperately to avoid multiple I/O ops
        polygon ogr.Geometry: ogr geometry of the target polygon
        (xmin, ymin, xmax, ymax).
        calculations: list[str] = ["mean", "std"],
        calculations to apply to the weighted stack data (default is mean and std (np.ma.average)).
        polygon_id (str): Identifier for the polygon.
        mask_dataset (Optional[gdal.Dataset]): if provided masks during the process.
        assumes 0,1 mask, with 1 == data to mask
        lower_limit (Union[int,None]): if provided sets all values in the array below this
        value to np.nan
        multiproc (bool): if ture carries out assumes multiproc and locks the tiff while
        accessing it

    Returns:
        tuple[str, Union[tuple[str, list[float]], str]]: target identifier and list
        of calculation and result tuples for each band.

    """
    try:
        minx, miny, maxx, maxy = retrieve_geometry_bbox(geom=polygon)
        check_bbox_within_raster(
            bbox=(minx, miny, maxx, maxy),
            geotransform_plus=stack_geotransform_plus,
            _id=polygon_id,
        )

        array_coord_bbox, array_pixel_bbox = calculate_array_bbox(
            target_bbox=(minx, miny, maxx, maxy), gt=stack_geotransform_plus.geotransform
        )

        start_row, start_col, end_row, end_col = array_pixel_bbox

        # Read data for all bands within the bounding box
        stack_data = stack_dataset.ReadAsArray(
            start_col, start_row, end_col - start_col, end_row - start_row
        ).astype(np.float32)

        __, rows, cols = stack_data.shape

        weight_array = calculate_polygon_weights(
            array_bbox=array_coord_bbox, target_polygon=polygon, num_rows=rows, num_cols=cols
        )

        # Mask NaN values
        if lower_limit is not None:
            stack_data[stack_data < lower_limit] = np.nan

        nodata_value = stack_dataset.GetRasterBand(1).GetNoDataValue()

        if nodata_value is None:
            raise AssertionError(
                "nodata value not detected in the raster, neccescary for correct masking"
            )
        stack_data[stack_data == nodata_value] = np.nan

        scales, offsets = get_dataset_scale_and_offset(_dataset=stack_dataset)

        stack_data = stack_data * np.array(scales).reshape(len(scales), 1, 1) + np.array(
            offsets
        ).reshape(len(offsets), 1, 1)

        mask = np.isnan(stack_data)
        stack_data = np.ma.masked_array(stack_data, mask=mask)
        weight_stack = np.repeat(weight_array[np.newaxis, ...], stack_data.shape[0], axis=0)

        weight_stack[~np.isfinite(stack_data)] = 0

        # Calculate weighted mean for each band
        results = {}
        for _calc in calculations:
            if _calc == "mean":
                _stat = "mean"
                applied_func_result = np.ma.average(a=stack_data, axis=(1, 2), weights=weight_stack)
            elif _calc == "std":
                _stat = "std"
                applied_func_result = weighted_std(
                    data=stack_data, weights=weight_stack, axis=(1, 2)
                )
            elif _calc == "maximum":
                _stat = "maximum"
                applied_func_result = np.nanmax(stack_data, axis=(1, 2))
            elif _calc == "minimum":
                _stat = "minimum"
                applied_func_result = np.nanmin(stack_data, axis=(1, 2))

            else:
                raise KeyError("calculation must be one mean, std, maximum, minimum")

            applied_func_result_list = applied_func_result.filled(np.nan).tolist()

            results[_stat] = applied_func_result_list
        # Close the stack dataset
        stack_data = None
        weight_stack = None
        weight_array = None
        applied_func_result = None

    except Exception as exc:
        results = str(exc)
        stack_data = None
        weight_stack = None
        weight_array = None
        applied_func_result = None

    return polygon_id, results


####################################
def calculate_polygon_weights(
    array_bbox: tuple[float, float, float, float],
    target_polygon: ogr.Geometry,
    num_rows: int,
    num_cols: int,
):
    """
    Calculate weights for each cell in an array based on the overlap of each cell with the
    target rectangle (bounding box).

    Parameters:
        array_bbox (tuple[float, float, float, float]): Coordinates of the array bounding box
        in the format (xmin_arr, ymin_arr, xmax_arr, ymax_arr).
        target_polygon (ogr.Geometry): ogr polygon geometry
        num_rows (int): Number of rows in the array.
        num_cols (int): Number of columns in the array.

    Returns:
        np.ndarray: Array of weights for each cell in the array.
    """
    # Extract bounding box coordinates
    xmin_arr, ymin_arr, xmax_arr, ymax_arr = array_bbox

    # Calculate resolution
    x_res = (xmax_arr - xmin_arr) / num_cols
    y_res = (ymin_arr - ymax_arr) / num_rows

    # Initialize weights array
    weights = np.zeros((num_rows, num_cols))

    # Calculate cell area
    cell_area = x_res * abs(y_res)

    # Iterate over each cell in the array
    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate cell coordinates
            cell_xmin = xmin_arr + j * x_res
            cell_xmax = xmin_arr + (j + 1) * x_res
            cell_ymax = ymax_arr + i * y_res
            cell_ymin = ymax_arr + (i + 1) * y_res

            cell_geom = create_bbox_polygon(
                minx=cell_xmin, miny=cell_ymin, maxx=cell_xmax, maxy=cell_ymax
            )

            # Calculate overlap area between cell and target_polygon
            target_polygon_intersection = cell_geom.Intersection(target_polygon)

            if target_polygon_intersection:
                cell_weight = target_polygon_intersection.Area() / cell_area

            else:
                cell_weight = 0

            # Calculate weight for the cell
            weights[i, j] = cell_weight

    return weights


#################################
def create_bbox_polygon(
    minx: float,
    maxy: float,
    maxx: float,
    miny: float,
) -> ogr.Geometry:
    """create a ogr geom for a bbox extent

    Parameters
    ----------
    minx : float
        left (minx) most x coordinate
    maxy : float
        upper most y coordinate
    maxx : float
        right most x coordinate
    miny : float
        lower most y coordinate

    Returns
    -------
    ogr.wkbPolygon
        bbox polgyon geom
    """
    coords = [
        (minx, maxy),
        (maxx, maxy),
        (maxx, miny),
        (minx, miny),
        (minx, maxy),
    ]

    poly = create_polygon(coords)

    return poly


########################################
def create_polygon(coords: list) -> ogr.Geometry:
    """create a ogr polygon object from coords
    Parameters
    ----------
    coords : list
        coords as a list of tuple one x and y per tuple

    Returns
    -------
    polygon
        ogr polgyon object
    """
    if isinstance(coords[0], tuple):
        poly = ogr.Geometry(ogr.wkbPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for coord in coords:
            ring.AddPoint(coord[0], coord[1])

        ring.CloseRings()
        poly.AddGeometry(ring)

    elif isinstance(coords[0], list) and isinstance(coords[0][0], tuple):
        poly = ogr.Geometry(ogr.wkbMultiPolygon)
        for _list in coords:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for coord in _list:
                ring.AddPoint(coord[0], coord[1])

            ring.CloseRings()
            poly.AddGeometry(ring)
    else:
        raise AttributeError(
            "create polygon expects a list of tuples for a single ring/geom polygon"
            " or list of lists for a multi ring/geom polygon"
        )

    return poly


####################################
def weighted_std(data: np.ndarray, weights: np.ndarray, axis: tuple[int, int] = (1, 2)):
    average = np.ma.average(data, axis=axis, weights=weights)
    average = average.reshape(data.shape[0], 1, 1)
    variance = np.ma.average((data - average) ** 2, axis=axis, weights=weights)
    return np.sqrt(variance)


############################
def retrieve_geometry_bbox(geom: ogr.Geometry) -> tuple[float, float, float, float]:
    minx, maxx, miny, maxy = geom.GetEnvelope()
    geom = None
    return minx, miny, maxx, maxy


####################################
def check_bbox_within_raster(
    bbox: tuple[float, float, float, float],
    geotransform_plus: GeotransformPlus,
    _id: Union[int, str] = "id not provided",
) -> int:
    """
    Check if a bounding box falls within the extent of a raster.

    Parameters:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in the format
        (xmin, ymin, xmax, ymax).
        gt (tuple[float, float, float, float, float, float]): GDAL raster geotransform.
        _id (Union[int, str]): Identifier for the bounding box to report if there is an
        error default is id not provided

    Raises:
        AssertionError: If the bounding box falls outside the raster bounds.

    Returns:
        None
    """
    xmin, ymin, xmax, ymax = bbox

    # Calculate pixel indices corresponding to bounding box
    leftx = int((xmin - geotransform_plus.geotransform[0]) / geotransform_plus.geotransform[1])
    lowery = int((ymin - geotransform_plus.geotransform[3]) / geotransform_plus.geotransform[5])
    rightx = int((xmax - geotransform_plus.geotransform[0]) / geotransform_plus.geotransform[1])
    uppery = int((ymax - geotransform_plus.geotransform[3]) / geotransform_plus.geotransform[5])

    # Check if bounding box is within raster bounds
    if any(digit < 0 or digit > geotransform_plus.height for digit in [lowery, uppery]):
        raise AssertionError(
            f"Bounding box: {_id} falls outside the raster bounds in the y-direction, "
            f"bbox: {bbox}, cell lowery:{lowery} uppery: {uppery}, "
            f"raster height: {geotransform_plus.height}"
        )
    if any(digit < 0 or digit > geotransform_plus.width for digit in [rightx, leftx]):
        raise AssertionError(
            f"Bounding box: {_id} falls outside the raster bounds in the x-direction, bbox: {bbox},"
            f" cell rightx:{rightx} leftx: {leftx}, raster width: {geotransform_plus.width}"
        )

    return 0


####################################
def calculate_array_bbox(
    target_bbox: tuple[float, float, float, float],
    gt: tuple[float, float, float, float, float, float],
) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]]:
    """
    Calculate the bounding box of the array that covers the target bounding box and extends
    to the nearest cell boundaries.

    Parameters:
        target_bbox (tuple[float, float, float, float]): Coordinates of the target bounding box
        in the format (xmin_tgt, ymin_tgt, xmax_tgt, ymax_tgt).
        gt (tuple[float, float, float, float, float, float]): GDAL raster geotransform.

    Returns:
        tuple[tuple[float, float, float, float], tuple[int, int, int, int]]:
            Coordinates of the array bounding box and pixel indices corresponding to the
            target bbox.
    """
    # Extract target bounding box coordinates
    xmin_tgt, ymin_tgt, xmax_tgt, ymax_tgt = target_bbox

    # extract raster geotransform info
    x_origin = gt[0]
    y_origin = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]

    # Calculate array bbox coordinates
    cell_width = abs(pixel_width)
    cell_height = abs(pixel_height)

    # Calculate the cell indices corresponding to the target bbox
    start_col = int((xmin_tgt - x_origin) / cell_width)
    end_col = math.ceil((xmax_tgt - x_origin) / cell_width)
    start_row = int((y_origin - ymax_tgt) / cell_height)
    end_row = math.ceil((y_origin - ymin_tgt) / cell_height)

    # Calculate array bbox coordinates based on cell indices
    xmin_arr = x_origin + start_col * cell_width
    ymin_arr = y_origin - end_row * cell_height
    xmax_arr = x_origin + end_col * cell_width
    ymax_arr = y_origin - start_row * cell_height

    coord_bbox = xmin_arr, ymin_arr, xmax_arr, ymax_arr

    pixel_bbox = start_row, start_col, end_row, end_col

    return coord_bbox, pixel_bbox


#################################
def vector_file_to_dict(vector_file_path: str, column: str = "") -> dict[str, ogr.Geometry]:
    """
    Retrieve the geometries in a vector file as a dictionary with the str
    ID column values as keys, or an incremental key if no column is provided.

    Parameters
    ----------
    vector_file_path : str
        Path to the vector file from which to retrieve geometries.
    column : str, optional
        Column to retrieve IDs from. Creates incremental ID if not given. Default is ''.

    Returns
    -------
    dict[str, ogr.Geometry]
        Dictionary of geometries with an str ID as the key.

    Raises
    ------
    KeyError
        If the specified column is not found in the vector file.

    Notes
    -----
    The geometries are cloned to ensure that modifications to the returned geometries
    do not affect the original dataset.
    """
    geom_dict = {}

    # check for fid
    in_ds = gdal.OpenEx(vector_file_path, gdal.GA_Update)

    in_layer = in_ds.GetLayer()

    in_layer_defn = in_layer.GetLayerDefn()

    if column:
        fields = [
            (in_layer_defn.GetFieldDefn(i).GetNameRef())
            for i in range(0, in_layer_defn.GetFieldCount())
        ]

        if column not in fields:
            raise KeyError(
                f"column: {column} not found in vector: {vector_file_path} , "
                f"please provide an existing column from those detected: {fields}"
            )

    count = 1
    _feature = in_layer.GetNextFeature()
    while _feature:
        _geometry = _feature.GetGeometryRef()

        if column:
            _id = _feature.GetField(column)
        else:
            _id = count

        geom_dict[str(_id)] = _geometry.Clone()
        _feature = in_layer.GetNextFeature()
        count += 1

    in_ds = in_layer = None

    return geom_dict


#################################
def get_end_dates_from_bands(_dataset: gdal.Dataset):
    """
    Get the 'end_date' metadata variable from each band in a GDAL raster dataset.

    Parameters
    ----------
    _dataset : gdal.Dataset
        gdal dataset object.

    Returns
    -------
    list
        A list of 'end_date' values order corresponding to each band in the dataset.
    """
    # Initialize list to store end_date values
    end_dates = []

    # Iterate through each band in the dataset
    for band_index in range(_dataset.RasterCount):
        band = _dataset.GetRasterBand(band_index + 1)  # GDAL band index is 1-based
        metadata = band.GetMetadata()
        end_date = metadata.get("end_date", None)
        end_dates.append(end_date)

    return end_dates


#################################
def get_dataset_scale_and_offset(_dataset: gdal.Dataset):
    """
    Get the scale and offset from each band in a GDAL raster dataset if applicable.

    Parameters
    ----------
    _dataset : gdal.Dataset
        gdal dataset object.

    Returns
    -------
    tuple[list, list]
        A tuple of saclaes and offsets lists order corresponding to each band in the dataset.
    """
    # Initialize list to store end_date values
    scales = []
    offsets = []

    # Iterate through each band in the dataset
    for band_index in range(_dataset.RasterCount):
        band = _dataset.GetRasterBand(band_index + 1)  # GDAL band index is 1-based
        _scale = band.GetScale()
        if not _scale:
            _scale = 1.0
        scales.append(_scale)
        _offset = band.GetOffset()
        if not _offset:
            _offset = 0.0
        offsets.append(_offset)

    return scales, offsets


#################################
def detect_ogr_type(series):
    """
    Detect the appropriate OGR field type for a pandas Series.

    Parameters
    ----------
    series : pd.Series
        The pandas Series whose data type is to be mapped to an OGR type.

    Returns
    -------
    ogr_type : int
        The OGR data type constant for the series.
    """
    if pd.api.types.is_integer_dtype(series):
        return ogr.OFTInteger
    elif pd.api.types.is_float_dtype(series):
        return ogr.OFTReal
    elif pd.api.types.is_bool_dtype(series):
        return ogr.OFTInteger  # Booleans can be stored as integers
    elif pd.api.types.is_datetime64_any_dtype(series):
        return ogr.OFTDateTime
    else:
        return ogr.OFTString  # Default to string for other types


#################################
def output_df_and_geometries_to_gpkg(
    vector_file_path: str,
    df: pd.DataFrame,
    id_col: str,
    output_path: str,
    group_columns: list[str] = ["product", "date"],
    value_columns: list[str] = ["mean", "std"],
):
    """
    This function retrieves geometries from the specified vector file, merges them with
    attributes from the provided DataFrame based on a shared ID column, and saves the
    resulting combined dataset as a new GeoPackage file. It pivots the DataFrame to ensure
    that statistical values (mean, std, etc.) are uniquely identified by the combination of
    specified grouping columns (e.g., product and date).

    Parameters
    ----------
    vector_file_path : str
        Path to the input vector file containing geometries.
    df : pd.DataFrame
        DataFrame with data to merge, including a shared ID column.
    group_columns : list[str], optional
        List of column names to group by when creating the unique ID. Default is ["product", "date"].
    value_columns : list[str], optional
        List of statistical columns to aggregate and pivot. Default is ["mean", "std"].
    """
    # Retrieve geometries as a dictionary using vector_file_to_dict
    geom_dict = vector_file_to_dict(vector_file_path, id_col)

    geom_dict = {_key: _value.ExportToWkt() for _key, _value in geom_dict.items()}

    # Merge DataFrame with geometry dictionary
    geom_df = pd.DataFrame.from_dict(geom_dict, orient="index", columns=["geometry"])
    geom_df.index.name = id_col
    geom_df.reset_index(inplace=True)

    df["unique_id"] = df[group_columns].agg("_".join, axis=1)

    # Step 2: Pivot the DataFrame
    pivoted_df = df.pivot_table(
        index=id_col,
        columns="unique_id",
        values=value_columns,
        aggfunc="first",  # Use 'first' to keep the first occurrence
    )

    # Step 3: Flatten MultiIndex columns
    pivoted_df.columns = ["_".join(col).strip() for col in pivoted_df.columns.values]

    # Step 4: Reset index to make ID a column
    pivoted_df.reset_index(inplace=True)

    merged_df = geom_df.merge(pivoted_df, on=id_col, how="left")

    in_ds = gdal.OpenEx(vector_file_path)
    in_layer = in_ds.GetLayer()
    # Define the output vector file
    driver = ogr.GetDriverByName("GPKG")
    out_ds = driver.CreateDataSource(output_path)
    out_layer = out_ds.CreateLayer(
        "merged_layer", in_layer.GetSpatialRef(), geom_type=ogr.wkbMultiPolygon
    )

    # Add fields from the DataFrame to the output layer
    for col in merged_df.columns:
        if col != "geometry":
            ogr_type = detect_ogr_type(merged_df[col])
            out_layer.CreateField(ogr.FieldDefn(col, ogr_type))

    # Add geometries and attributes to the output layer
    for _, row in merged_df.iterrows():
        feature = ogr.Feature(out_layer.GetLayerDefn())
        for col, value in row.items():
            if col == "geometry":
                feature.SetGeometry(ogr.CreateGeometryFromWkt(value))
            else:
                feature.SetField(col, value)
        out_layer.CreateFeature(feature)
        feature = None

    out_ds = None  # Close the dataset

    return output_path
