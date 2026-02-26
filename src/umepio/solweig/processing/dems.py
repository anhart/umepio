from umepio.solweig.io_utils import *
import requests
import os
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from scipy.interpolate import NearestNDInterpolator
import startinpy
from rasterio import Affine
from shapely.geometry import shape
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject


from umepio.solweig.processing.buildings import Buildings

class DEMS:
    '''
    Class for handling Digital Elevation Models (DEM) including DTM and DSM,
    fetching AHN data via WCS, filling missing data, resampling, cropping,
    and integrating building footprints for urban terrain modeling.

    Attributes:
        buffer (float):                           Buffer size in meters for bbox expansion.
        bbox (tuple):                             Bounding box coordinates (xmin, ymin, xmax, ymax).
        bufferbbox (tuple):                       Buffered bounding box expanded by buffer.
        building_data (list):                     List of building geometries and attributes.
        resolution (float):                       Desired output raster resolution in meters.
        user_building_data (list):                User-provided building data.
        output_dir (str):                         Directory to save output files.
        bridge (bool):                            Whether to include 'overbruggingsdeel' data in the DSM.
        resampling (rasterio.enums.Resampling):   Resampling method for raster operations.
        crs (CRS):                                Coordinate reference system, default EPSG:28992.
        dtm (np.ndarray):                     Digital Terrain Model raster data.
        dsm (np.ndarray):                     Digital Surface Model raster data.
        transform (Affine):                       Affine transform for the rasters.
        og_dtm (np.ndarray):                  Original DTM before modifications.
        og_dsm (np.ndarray):                  Original DSM before modifications.
        is3D (bool):                              Flag indicating if DSM is 3D.
    '''
    def __init__(self, bbox, building_data, resolution=0.5, bridge=False, resampling=Resampling.cubic_spline, output_dir="output"):
        '''
        Initialize the DEM builder object.

        Parameters:
            bbox (tuple):                           Bounding box coordinates (xmin, ymin, xmax, ymax).
            building_data (list):                   Building geometries and data.
            resolution (float):                     Desired output resolution in meters (default 0.5).
            bridge (bool):                          Whether to include  'overbruggingsdeel' geometries (default False).
            resampling (rasterio.enums.Resampling): Resampling method (default cubic_spline).
            output_dir (str):                       Directory for output files (default "output").

        Returns:
            None
        '''
        self.buffer = 2
        self.bbox = bbox
        self.bufferbbox = edit_bounds(bbox, self.buffer)
        self.building_data = building_data
        self.resolution = resolution
        self.user_building_data = []
        self.output_dir = output_dir
        self.bridge = bridge
        self.resampling = resampling
        self.crs = (CRS.from_epsg(28992))
        self.dtm, self.dsm, self.transform = self.create_dem(bbox)
        self.og_dtm, self.og_dsm = self.dtm, self.dsm
        self.is3D = False

    @staticmethod
    def fetch_ahn_wcs(bufferbbox, output_file, nodata_value=-9999, coverage="dtm_05m", wcs_resolution=0.5):
        '''
        Fetch AHN WCS data for a given buffered bounding box and save as GeoTIFF.

        Parameters:
            bufferbbox (tuple):     Buffered bounding box (xmin, ymin, xmax, ymax).
            output_file (str):      Output filepath for the GeoTIFF (default "output/dtm.tif").
            coverage (str):         Coverage layer name, e.g. "dtm_05m" or "dsm_05m" (default "dtm_05m").
            wcs_resolution (float): Resolution of WCS data in meters (default 0.5).

        Returns:
            tuple or None: (rasterio dataset object, numpy array of raster data) if successful, else None.
        '''
        # Calculate width and height from bbox and resolution
        width = int((bufferbbox[2] - bufferbbox[0]) / wcs_resolution)
        height = int((bufferbbox[3] - bufferbbox[1]) / wcs_resolution)

        # WCS Service URL
        WCS_URL = "https://service.pdok.nl/rws/ahn/wcs/v1_0"

        # Construct query parameters
        params = {
            "SERVICE": "WCS",
            "VERSION": "1.0.0",
            "REQUEST": "GetCoverage",
            "FORMAT": "GEOTIFF",
            "COVERAGE": coverage,
            "BBOX": f"{bufferbbox[0]},{bufferbbox[1]},{bufferbbox[2]},{bufferbbox[3]}",
            "CRS": "EPSG:28992",
            "RESPONSE_CRS": "EPSG:28992",
            "WIDTH": str(width),
            "HEIGHT": str(height)
        }

        response = requests.get(WCS_URL, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=60) # Increased timeout

        if response.status_code == 200:

            # Ensure the output directory exists and if not 
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            #write the geotiff to the file    
            with open(output_file, "wb") as f:
                f.write(response.content)

            # Read raster + nodata
            with rasterio.open(output_file) as dataset:
                array = dataset.read(1)
                old_nodata = dataset.nodata

            # Replace nodata values in the array (only if old nodata exists)
            if old_nodata is not None:
                array[array == old_nodata] = nodata_value

            # Write back + update nodata metadata
            with rasterio.open(output_file, "r+") as dst:
                dst.write(array, 1)
                dst.nodata = nodata_value

            return dst, array

        else:
            print(f"Failed to fetch AHN data: HTTP {response.status_code}")
            return None
        

    @staticmethod
    def extract_center_cells(geo_array, no_data=-9999):
        '''
        Extract the values of each cell in the input data and save these with the x and y (row and col)
        indices. Thereby, make sure that the corners of the dataset are filled for a full coverage triangulation
        in the next step.

        Parameters:
            geo_array (np.ndarray):         Raster data array.
            no_data (int):                  No data value to identify invalid cells (default -9999).

        Returns:
            list:                           List of [x, y, z] cell values with corners interpolated if no data.
        '''
        # Get the indices of the rows and columns
        rows, cols = np.indices(geo_array.shape)

        # Identify corner coordinates
        corners = {
            "top_left": (0, 0),
            "top_right": (0, geo_array.shape[1] - 1),
            "bottom_left": (geo_array.shape[0] - 1, 0),
            "bottom_right": (geo_array.shape[0] - 1, geo_array.shape[1] - 1)
        }

        # Mask for valid center cells (non-no_data)
        valid_center_cells = (geo_array != no_data)

        # Extract x, y, z values for valid cells
        x_valid = cols[valid_center_cells]
        y_valid = rows[valid_center_cells]
        z_valid = geo_array[valid_center_cells]

        # Create interpolator from valid points
        interpolator = NearestNDInterpolator(list(zip(x_valid, y_valid)), z_valid)

        # Check each corner for no data and interpolate if necessary
        for corner_name, (row, col) in corners.items():
            if geo_array[row, col] == no_data:
                # Interpolate the nearest valid value
                geo_array[row, col] = interpolator((col, row))

        # Extract non-no_data and center cells again after filling corners
        valid_center_cells = (geo_array != no_data)

        # Extract final x, y, z values after filling corners
        x_filled = cols[valid_center_cells]
        y_filled = rows[valid_center_cells]
        z_filled = geo_array[valid_center_cells]

        # Prepare final list of [x, y, z]
        xyz_filled = []
        for x_i, y_i, z_i in zip(x_filled, y_filled, z_filled):
            xyz_filled.append([x_i, y_i, z_i])

        return xyz_filled

    def crop_to_bbox(self, array, transform):
        '''
        Crop a buffered raster array to the original bounding box.

        Parameters:
            array (np.ndarray): Raster data array with buffer.
            transform (Affine): Affine transform matrix of input array.

        Returns
        -------
        cropped_array (np.ndarray):
            Cropped raster array.
        new_transform (Affine):
            New Affine transform matrix for cropped raster.
        '''

        # Compute the window from the full buffered transform, for the smaller (target) bbox
        crop_pixels = int(self.buffer / self.resolution)

        # Crop array: remove buffer from all sides
        print(crop_pixels)
        cropped_array = array[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
        print(cropped_array.shape)

        # Adjust transform: move origin by number of removed pixels
        new_transform = transform * Affine.translation(crop_pixels, crop_pixels)

        return cropped_array, new_transform

    def resample_raster(self, input_array, input_transform, input_crs, output_resolution):
        '''
        Resample a raster to a different resolution.

        Parameters:
            input_array (np.ndarray): Input raster data.
            input_transform (Affine): Affine transform of input raster.
            input_crs (CRS): Coordinate Reference System of input raster.
            output_resolution (float): Desired output resolution in meters.

        Returns
        -------
        resampled_array (np.ndarray):
            Resampled raster array.
        new_transform (Affine):
            New Affine transform matrix for resampled raster.
        '''
        height, width = input_array.shape
        new_width = int((width * input_transform.a) / output_resolution)
        new_height = int((height * -input_transform.e) / output_resolution)

        new_transform = rasterio.transform.from_origin(
            input_transform.c, input_transform.f, output_resolution, output_resolution
        )

        resampled_array = np.empty((new_height, new_width), dtype=input_array.dtype)

        reproject(
            source=input_array,
            destination=resampled_array,
            src_transform=input_transform,
            src_crs=input_crs,
            dst_transform=new_transform,
            dst_crs=input_crs,
            resampling=self.resampling
        )

        return resampled_array, new_transform

    def fill_raster(self, geo_array, nodata_value, transform):
        '''
        Fill no-data values in a raster using Laplace interpolation.

        Parameters:
            geo_array (np.ndarray):     Cropped raster data array.
            nodata_value (int):         No-data value to replace NaNs after interpolation.
            transform (Affine):         Affine transform matrix of the raster.

        Returns:
            new_data(np.ndarray):       Filled raster array with no-data values replaced.
        '''

        # creating delaunay
        points = self.extract_center_cells(geo_array, no_data=nodata_value)
        dt = startinpy.DT()
        dt.insert(points, "BBox")

        # for interpolation, grid of all column and row positions, excluding the first and last rows/cols
        cols, rows = np.meshgrid(
            np.arange(0, geo_array.shape[1]),
            np.arange(0, geo_array.shape[0])
        )

        # flatten the grid to get a list of all (col, row) locations
        locs = np.column_stack((cols.ravel(), rows.ravel()))
        interpolated_values = dt.interpolate({"method": "Laplace"}, locs)

        # reshape interpolated grid back to original
        interpolated_grid = np.reshape(interpolated_values, (geo_array.shape[0], geo_array.shape[1]))

        # fill new_data with interpolated values
        new_data= interpolated_grid
        new_data = np.where(np.isnan(new_data), nodata_value, new_data)

        return new_data

    def replace_buildings(self, filled_dtm, dsm_buildings, buildings_geometries, transform, bridge):
        '''
        Replace filled DTM values with DSM building heights where buildings exist.

        Parameters:
            filled_dtm (np.ndarray):        Filled, cropped DTM array.
            dsm_buildings (np.ndarray):     Filled, cropped DSM array with buildings.
            buildings_geometries (list):    List of building geometries (dict or GeoJSON features).
            transform (Affine):             Affine transform matrix of the rasters.
            bridge (bool):                  Whether to include 'overbrugginsdeel' geometries.

        Returns:
            final_dsm (np.ndarray):         Final DSM array combining ground and building heights.
        '''
        geometries = [shape(building['geometry']) for building in buildings_geometries if 'geometry' in building]
        bridging_geometries = []
        if bridge is True:
            bridge_crs = "http://www.opengis.net/def/crs/EPSG/0/28992"
            url = f"https://api.pdok.nl/lv/bgt/ogc/v1/collections/overbruggingsdeel/items?bbox={self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}&bbox-crs={bridge_crs}&crs={bridge_crs}&limit=1000&f=json"
            response = requests.get(url)
            if response.status_code == 200:
                bridging_data = response.json()
                if "features" in bridging_data:  # Ensure data contains geometries
                    bridging_geometries = [shape(feature['geometry']) for feature in bridging_data["features"] if
                                           'geometry' in feature]
            else:
                print(f"Error fetching bridges: {response.status_code}, {response.text}")

        # Ensure mask has same shape as filled_dtm
        all_geometries = bridging_geometries + geometries
        building_mask = geometry_mask(all_geometries, transform=transform, invert=False, out_shape=filled_dtm.shape)

        # Get shape differences
        dtm_shape = filled_dtm.shape
        dsm_shape = dsm_buildings.shape

        if dtm_shape != dsm_shape:
            # Compute the cropping offsets
            row_diff = dsm_shape[0] - dtm_shape[0]
            col_diff = dsm_shape[1] - dtm_shape[1]

            # Ensure even cropping from all sides (center alignment)
            row_start = row_diff // 2
            col_start = col_diff // 2
            row_end = row_start + dtm_shape[0]
            col_end = col_start + dtm_shape[1]

            # Crop dsm_buildings to match filled_dtm
            dsm_buildings = dsm_buildings[row_start:row_end, col_start:col_end]

        # Apply the mask
        final_dsm = np.where(building_mask, filled_dtm, dsm_buildings)
        return final_dsm

    def create_dem(self, bbox):
        '''
        Create Digital Elevation Model (DEM) from AHN data with optional building and overbrugginsdeel data.

        Parameters:
            bbox (tuple):       Bounding box coordinates (xmin, ymin, xmax, ymax).

        Returns
        -------
        cropped_dtm (np.ndarray):
            Filled, cropped DTM array.
        cropped_dsm (np.ndarray):
            Cropped DSM array with buildings and building heights, optional output.
        transform (Affine):
            Affine transform matrix of the rasters.
         '''

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # --- Fetch DTM ---
        dtm_dst, dtm_array = self.fetch_ahn_wcs(
            self.bufferbbox, output_file=f"{self.output_dir}/dtm_fetched.tif", nodata_value=-9999, coverage="dtm_05m", wcs_resolution=0.5
        )
        transform = dtm_dst.transform
        filled_dtm  = self.fill_raster(dtm_array, dtm_dst.nodata, transform)

        # --- Fetch DSM if buildings are used ---
        if self.building_data:
            dsm_dst, dsm_array = self.fetch_ahn_wcs(
                self.bufferbbox, output_file=f"{self.output_dir}/dsm_fetched.tif", nodata_value=-9999, coverage="dsm_05m", wcs_resolution=0.5
            )
            filled_dsm = self.fill_raster(dsm_array, dsm_dst.nodata, transform)
            final_dsm = self.replace_buildings(
                filled_dtm, filled_dsm, self.building_data, transform, self.bridge
            )
        else:
            final_dsm = filled_dtm
        # --- Resample if needed ---
        if self.resolution != 0.5:
            filled_dtm, resamp_transform = self.resample_raster(
                filled_dtm, transform, dtm_dst.crs, self.resolution
            )

            if final_dsm is not None:
                final_dsm, _ = self.resample_raster(
                    final_dsm, transform, dtm_dst.crs, self.resolution
                )

            transform = resamp_transform

        # --- Crop the arrays to the bounding box after interpolation ---
        cropped_dtm, transform = self.crop_to_bbox(filled_dtm, transform)

        if final_dsm is not None:
            cropped_dsm, _ = self.crop_to_bbox(final_dsm, transform)
    
        # --- Write outputs ---
        write_output(dtm_dst, self.crs, cropped_dtm, transform, f"{self.output_dir}/processed_dtm.tif")

        if final_dsm is not None:
            write_output(dtm_dst, self.crs, cropped_dsm, transform, f"{self.output_dir}/processed_dsm.tif")

        return cropped_dtm, cropped_dsm if final_dsm is not None else cropped_dtm, transform
    