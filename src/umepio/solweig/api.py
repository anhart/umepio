# from .pipeline import generate_solweig_inputs
# from .config import SolweigConfig

# __all__ = ["generate_solweig_inputs", "SolweigConfig"]
# collect all the other functions into the actual callable things here
from rasterio.enums import Resampling
import rasterio
from pathlib import Path
from umepio.solweig.processing.buildings import Buildings
from umepio.solweig.io_utils import plot_geojson, show_raster_array, visualize_landcover_raster, write_output_landcover
from umepio.solweig.processing.dems import DEMS
from umepio.solweig.processing.chm import CHM
from umepio.solweig.processing.landcover import LandCover

__all__ = [
    "load_buildings",
    "load_dems",
    "load_chm",
    "load_landcover",
    "generate_solweig_inputs"
]

def load_buildings(bbox, output_folder, plot=False, wfs_url="https://data.3dbag.nl/api/BAG3D/wfs", layer_name="BAG3D:lod13", gpkg_name="buildings", output_layer_name="buildings"):
    """
    Load all buildings within a bounding box.

    Parameters:
        bbox (tuple):               Bounding box (min_x, min_y, max_x, max_y).
        wfs_url (str):              URL for the WFS service. Default is 3dbag.nl API.
        layer_name (str):           Name of the WFS layer to query. Default is "BAG3D:lod13".
        gpkg_name (str):            Name of the GeoPackage output file (without extension).
        output_folder (str):        Folder to save the downloaded data.
        output_layer_name (str):    Layer name to save within the GeoPackage.
        plot (bool):             If True, plot the loaded buildings using plot_geojson.

    Returns:
        list:   List of dicts with 'geometry' and 'parcel_id'.
    """
    # Initialize the Buildings class
    buildings_obj = Buildings(bbox, wfs_url, layer_name, gpkg_name, output_folder, output_layer_name)

    # Optionally plot
    if plot:
        plot_geojson(buildings_obj.building_geometries, color="pink")
    
    # Return the building geometries
    return buildings_obj.building_geometries

def load_dems(bbox, building_data, output_folder, plot=False, resolution=0.5, bridge=False, resampling=Resampling.cubic_spline):
    """
    Generate Digital Elevation Models (DTM and DSM) for a bounding box and given buildings.

    Parameters:
        bbox (tuple):               (min_x, min_y, max_x, max_y)
        building_data (list or None): List of building geometries (dicts with 'geometry' and 'parcel_id'). If None, only DTM will be generated.
        resolution (float):         Desired output raster resolution (default 0.5 m)
        bridge (bool):              Include 'overbruggingsdeel' data in DSM (default False)
        resampling (rasterio.enums.Resampling):   Resampling method for raster operations.
        output_dir (str):           Folder to save output DEM files (default "output")

    Returns:
            dtm (np.ndarray):       Digital Terrain Model
            dsm (np.ndarray or None): Digital Surface Model (includes buildings), None if building_data is None
            transform (Affine):     Affine transform for the rasters
    """
    dems = DEMS(bbox, building_data, resolution, bridge, resampling, output_folder)
    
    if plot:
        show_raster_array(dems.dtm, raster_type='height', title='DTM')
        show_raster_array(dems.dsm, raster_type='height', title='DSM')

    return dems.dtm, dems.dsm, dems.transform

def load_chm(bbox, dtm, dsm, trunk_height, output_folder_chm, plot=False, output_folder_las='temp', resolution=0.5, merged_output='pointcloud.las', ndvi_threshold=0.05):
    """
    Generate Canopy Height Model (CHM) from DTM and DSM for a given bounding box.

    Parameters:
        bbox (tuple):              (min_x, min_y, max_x, max_y)
        dtm (str or np.ndarray):   Digital Terrain Model (file path or array)
        dsm (str or np.ndarray):   Digital Surface Model (file path or array)
        trunk_height (float):      Height to subtract for tree trunks
        output_folder_chm (str):   Folder to save CHM output
        plot (bool):               Whether to plot CHM (default False)
        output_folder_las (str):   Folder to save merged LAS point cloud (default 'temp')
        resolution (float):        Output raster resolution (default 0.5)
        merged_output (str):       Filename for the merged LAS file (default 'pointcloud.las')
        ndvi_threshold (float):    NDVI threshold for vegetation filtering (default 0.05)

    Returns:
        np.ndarray:               Generated Canopy Height Model (CHM)
    """
    chm = CHM(bbox, dtm, dsm, trunk_height, output_folder_chm, output_folder_las, resolution, merged_output, ndvi_threshold)
    
    if plot:
        show_raster_array(chm.chm, raster_type='height', title='CHM')

    return chm.chm

def load_landcover(bbox, output_folder, plot=False, crs="http://www.opengis.net/def/crs/EPSG/0/28992", use_bgt=True, main_roadtype=0, resolution=0.5, building_data=None, dataset=None,
                 dataset_path=None, buildings_path=None, layer=None, nodata_fill=0, roads_on_top=True, landcover_path_bgt="landcover_bgt.json",  landcover_top="landcover_top.json"):
    """
    Generate a land cover classification raster for a bounding box and save it to disk.

    This function builds a LandCover processor (BGT or Top10NL-based), rasterizes land cover
    features, and writes the resulting land cover array as a GeoTIFF in `output_folder`.

    Parameters:
        bbox (tuple):             (min_x, min_y, max_x, max_y) in the given CRS (default EPSG:28992).
        output_folder (str):      Folder to save the output GeoTIFF.
        plot (bool):              If True, visualize the land cover raster (default False).
        crs (str):                CRS for requests/alignment (default EPSG:28992).
        use_bgt (bool):           If True use BGT as source; otherwise use Top10NL (default True).
        main_roadtype (int):      Landcover code for hardened roads when using Top10NL (default 0).
        resolution (float):       Output raster resolution in meters (default 0.5).
        building_data (list|None):Optional preloaded building geometries.
        dataset (rasterio.DatasetReader|None): Optional reference dtm raster dataset for alignment.
        dataset_path (str|None):  Optional path to reference dtm raster for alignment.
        buildings_path (str|None):Optional path to building vector data (e.g. GPKG).
        layer (str|None):    Layer name if `buildings_path` is a multi-layer dataset.
        nodata_fill (int):        Nodata value written to the raster (default 0).
        roads_on_top (bool):      If True, burn roads after other classes (default True).
        landcover_path_bgt (str): Path to BGT landcover mapping JSON.
        landcover_top (str):      Path to Top10NL landcover mapping JSON.

    Returns:
        lc_array (np.ndarray):    2D land cover classification raster.
    """
    
    lc = LandCover(
            bbox=bbox,
            crs=crs,
            use_bgt=use_bgt,
            main_roadtype=main_roadtype,
            resolution=resolution,
            building_data=building_data,
            dataset=dataset,
            dataset_path=dataset_path,
            buildings_path=buildings_path,
            layer=layer,
            nodata_fill=nodata_fill,
            roads_on_top=roads_on_top,
            landcover_path_bgt=landcover_path_bgt,
            landcover_top=landcover_top,
            )
    
    out_path = str(Path(output_folder) / "landcover.tif")
    write_output_landcover(output=lc.array,transform=lc.transform, crs=crs, name=out_path, nodata=nodata_fill,  dtype=lc.array.dtype)
    
    if plot:
        visualize_landcover_raster(lc.array)

    return lc.array

def generate_solweig_inputs(
    bbox,
    output_folder,
    *,
    plot=False,
    # buildings
    wfs_url="https://data.3dbag.nl/api/BAG3D/wfs",
    layer_name="BAG3D:lod13",
    gpkg_name="buildings",
    output_layer_name="buildings",
    # dems
    resolution=0.5,
    bridge=False,
    resampling=Resampling.cubic_spline,
    # chm
    trunk_height=25,
    output_folder_las="temp",
    merged_output="pointcloud.las",
    ndvi_threshold=0.05,
    # landcover
    crs="http://www.opengis.net/def/crs/EPSG/0/28992",
    use_bgt=True,
    main_roadtype=0,
    nodata_fill=0,
    roads_on_top=True,
    landcover_path_bgt="landcover_bgt.json",
    landcover_top="landcover_top.json",
):
    """
    Run the full SOLWEIG input generation pipeline and write outputs to `output_folder`.

    Steps:
      1) Buildings (vector)
      2) DEMs: processed_dtm.tif + processed_dsm.tif
      3) CHM: canopy height model raster
      4) Landcover: landcover.tif aligned to DTM

    Returns:
      dict with arrays and file paths.
    """
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Buildings
    buildings = load_buildings(
        bbox=bbox,
        output_folder=str(out_dir),
        plot=plot,
        wfs_url=wfs_url,
        layer_name=layer_name,
        gpkg_name=gpkg_name,
        output_layer_name=output_layer_name,
    )

    # 2) DEMs (your DEMS class typically writes GeoTIFFs internally)
    dtm, dsm, transform = load_dems(
        bbox=bbox,
        building_data=buildings,
        output_folder=str(out_dir),
        plot=plot,
        resolution=resolution,
        bridge=bridge,
        resampling=resampling,
    )


    # 3) CHM
    chm = load_chm(
        bbox=bbox,
        dtm=dtm,              
        dsm=dsm,             
        trunk_height=trunk_height,
        output_folder_chm=str(out_dir),
        plot=plot,
        output_folder_las=output_folder_las,
        resolution=resolution,
        merged_output=merged_output,
        ndvi_threshold=ndvi_threshold,
    )

    # 4) Landcover (align to DTM using dataset_path)
    landcover = load_landcover(
        bbox=bbox,
        output_folder=str(out_dir),
        plot=plot,
        crs=crs,
        use_bgt=use_bgt,
        main_roadtype=main_roadtype,
        resolution=resolution,
        building_data=buildings,
        dataset_path=f"{output_folder}/processed_dtm.tif",
        nodata_fill=nodata_fill,
        roads_on_top=roads_on_top,
        landcover_path_bgt=landcover_path_bgt,
        landcover_top=landcover_top,
    )

    return {
        "buildings": buildings,
        "dtm": dtm,
        "dsm": dsm,
        "chm": chm,
        "landcover": landcover,
        "paths": {
            "dtm": str(out_dir / "processed_dtm.tif"),
            "dsm": str(out_dir / "processed_dsm.tif"),
            "landcover": str(out_dir / "landcover.tif"),
            "chm": str(out_dir / "CHM.tif"),
        },
    }

