''' In this file all the functions that have something to do with file formats and displaying outputs'''

import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def edit_bounds(bounds, buffer, shrink=False):
    '''
    Expands or shrinks bounding box coordinates by a buffer amount.

    Parameters:
        bounds (tuple): Bounding box as (min_x, min_y, max_x, max_y).
        buffer (float): Amount to expand or shrink the bounding box.
        shrink (bool):  If True, shrink the bounds by buffer; else expand (default False).

    Returns:
        tuple: Modified bounding box as (min_x, min_y, max_x, max_y).
    '''
    min_x, min_y, max_x, max_y = bounds

    if shrink:
        return (
            min_x + buffer,
            min_y + buffer,
            max_x - buffer,
            max_y - buffer
        )
    else:
        return (
            min_x - buffer,
            min_y - buffer,
            max_x + buffer,
            max_y + buffer
        )

def write_output(dataset, crs, output, transform, name, change_nodata=False):
    '''
    Writes a numpy array to a GeoTIFF file using rasterio.

    Parameters:
        dataset        :        Rasterio or laspy dataset (for metadata).
        crs            :        Coordinate Reference System for the output raster.
        output (np.ndarray):    Output numpy array grid to write.
        transform      :        Affine transform mapping pixel to spatial coordinates.
        name (str)     :        Output filename (including path).
        change_nodata (bool):   If True, use nodata value -9999; else use dataset's nodata.

    Returns:
        None
    '''
    output_file = name

    output = np.squeeze(output)
    # Set the nodata value: use -9999 if nodata_value is True or dataset does not have nodata.
    if change_nodata:
        nodata_value = -9999
    else:
        try:
            # TO DO: CHANGE THIS TO JUST INPUTTING A NODATA VALUE, NO NEED FOR THE WHOLE DATASET IN THIS FUNCTION
            nodata_value = dataset.nodata
            if nodata_value is None:
                raise AttributeError("No no data value found in dataset.")
        except AttributeError as e:
            print(f"Warning: {e}. Defaulting to -9999.")
            nodata_value = -9999

    # output the dataset
    with rasterio.open(output_file, 'w',
                       driver='GTiff',
                       height=output.shape[0],  # Assuming output is (rows, cols)
                       width=output.shape[1],
                       count=1,
                       dtype=np.float32,
                       crs=crs,
                       nodata=nodata_value,
                       transform=transform) as dst:
        dst.write(output, 1)
    print("File written to '%s'" % output_file)
    
def write_output_landcover(output, transform, crs, name, nodata=None, dtype=None):
    """
    Writes a numpy array to a GeoTIFF file using rasterio.

    Parameters:
        output (np.ndarray): Output array (rows, cols) or squeezable to that.
        transform (Affine):  Affine transform.
        crs:                 CRS (anything rasterio accepts).
        name (str):          Output filename (including path).
        nodata:              Nodata value to store in the file (recommended for categorical rasters).
        dtype:               Output dtype (defaults to output.dtype).
    """
    output = np.squeeze(output)
    if output.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got shape {output.shape}")

    if dtype is None:
        dtype = output.dtype

    with rasterio.open(
        name,
        "w",
        driver="GTiff",
        height=output.shape[0],
        width=output.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(output.astype(dtype, copy=False), 1)

def plot_geojson(geojson_list, crs="EPSG:28992", color='blue', edgecolor='black', labels=False):

    geoms = [shape(g['geometry']) for g in geojson_list]
    polygon_ids = [g.get('polygon_id', None) for g in geojson_list]

    gdf = gpd.GeoDataFrame({'polygon_id': polygon_ids, 'geometry': geoms}, crs=crs)

    ax = gdf.plot(color=color, edgecolor=edgecolor, figsize=(7, 7))

    if labels:
        for _, row in gdf.iterrows():
            centroid = row.geometry.centroid
            label = str(row['polygon_id']) if row['polygon_id'] is not None else ''
            ax.text(centroid.x, centroid.y, label, fontsize=8, ha='center', va='center', color='white')

    plt.title("Polygon Geometries")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def show_raster_array(array, raster_type='generic', title='Raster', vmin=None, vmax=None):
    
    cmap_dict = {
        'height': 'viridis',
        'temperature': 'turbo',
        'generic': 'gray',
        'diverging': 'bwr'
    }

    cmap = cmap_dict.get(raster_type, 'gray')

    plt.figure(figsize=(7, 7))
    img = plt.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(img, label='Pixel Value')
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.grid(False)
    plt.show()

def visualize_landcover_raster(raster_array):
    """
    Visualize the landcover raster.

    Parameters:
        raster_array (ndarray):     Array of the landcover raster.
    Returns:
        None, shows a Matplotlib plot for the raster array
    """
    cmap = ListedColormap(["purple", "grey", "black", "brown", "tan", "yellow", "green", "tan", "cyan"])
    categories = [-9999, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    norm = BoundaryNorm(categories, cmap.N)
    plt.figure(figsize=(6, 6))
    img = plt.imshow(raster_array, cmap=cmap, norm=norm, interpolation='nearest')
    cbar = plt.colorbar(img, ticks=categories)
    cbar.set_label("Land Cover Type")
    plt.title("Land Cover")
    plt.show()