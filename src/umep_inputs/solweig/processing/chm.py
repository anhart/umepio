from umep_inputs.solweig.io_utils import *
import requests
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from rasterio.features import  shapes
import startinpy
from rasterio import Affine
from shapely.geometry import shape,box
from rasterio.crs import CRS
from pathlib import Path
import laspy
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter, label
from importlib.resources import files, as_file

class CHM:
    '''
    Class for creating and managing Canopy Height Models (CHMs) from LiDAR data.
    This class handles downloading and merging LAS/LAZ tiles, filtering vegetation points
    based on classification and NDVI, rasterizing vegetation using interpolation, applying
    smoothing filters, and generating final CHM raster outputs.

    Attributes:
        bbox (tuple):               Bounding box coordinates (min_x, min_y, max_x, max_y) defining the area of interest.
        bufferedbbox (tuple):       Buffered bounding box extended by a fixed margin.
        crs (rasterio.crs.CRS):                         Coordinate reference system used (default EPSG:28992).
        dtm (numpy.ndarray or rasterio object):         Digital Terrain Model raster data.
        dsm (numpy.ndarray or rasterio object):         Digital Surface Model raster data.
        output_folder_chm (str):                        Folder path where CHM outputs are saved.
        gdf (geopandas.GeoDataFrame):                   GeoDataFrame containing tile lookup information.
        chm (numpy.ndarray):                            Initial Canopy Height Model raster array.
        tree_polygons (geopandas.GeoDataFrame):         Polygons representing tree footprints.
        transform (affine.Affine):                      Affine transform for raster coordinates.
        trunk_array (numpy.ndarray):                    Array representing estimated trunk heights.
        original_chm (numpy.ndarray):                   Copy of the original CHM before processing.
        og_polygons (geopandas.GeoDataFrame):           Copy of original tree polygons.
        original_trunk (numpy.ndarray):                 Copy of original trunk array.
    '''
    def __init__(self, bbox, dtm, dsm, trunk_height, output_folder_chm='output', output_folder_las='temp', resolution=0.5, merged_output='pointcloud.las', ndvi_threshold=0.05):
        '''
        Initialize the CHM class with bounding box, DTM, DSM, trunk height and folder paths.

        Parameters:
            bbox (tuple):                                     Bounding box as (min_x, min_y, max_x, max_y).
            dtm (numpy.ndarray or rasterio dataset):          Digital Terrain Model raster.
            dsm (numpy.ndarray or rasterio dataset):          Digital Surface Model raster.
            trunk_height (float):                             Trunk height as a percentage of total tree height (e.g., 20 for 20%).
            output_folder_las (str):                          Folder path for output LAS files.
            input_folder (str):                               Folder path for input files.
            output_folder_chm (str):                          Folder path for CHM-specific output.
            resolution (float, optional):                     Resolution for raster grid cells. Defaults to 0.5.
            merged_output (str, optional):                    Filename for merged LAS output. Defaults to 'pointcloud.las'.
            ndvi_threshold (float, optional):                 NDVI index to use for vegetation extraction.
        '''
        self.bbox = bbox
        self.bufferedbbox = edit_bounds(bbox, 2)
        self.crs = (CRS.from_epsg(28992))
        self.dtm = dtm
        self.dsm = dsm
        self.tree_mask = None
        self.output_folder_chm = output_folder_chm
        self.gdf = self.read_lookup()
        self.chm, self.tree_polygons, self.transform = self.init_chm(bbox, output_folder=output_folder_las, input_folder=output_folder_las, merged_output=merged_output,  ndvi_threshold=ndvi_threshold, resolution=resolution)
        self.trunk_array = self.chm * (trunk_height / 100 )
        self.original_chm, self.og_polygons, self.original_trunk = self.chm, self.tree_polygons, self.trunk_array
    
    def read_lookup(self):
        '''
        Read the lookup file (AHN_lookup.geojson) and return it as a GeoDataFrame.
        Includes basic error handling for missing or inaccessible files.
        '''
        try:
            resource = files("umep_inputs").joinpath("geotiles/AHN_lookup.geojson")
            with as_file(resource) as path:
                self.gdf = gpd.read_file(path)
        except Exception as e:
            print(f"Error reading AHN_lookup.geojson: {e}")
            self.gdf = None  # You can handle this accordingly

    def save_las(self, merged_las, veg_points, output_name="veg_points.las"):
        '''
        Save filtered vegetation points as a LAS file.

        Parameters:
            merged_las (laspy.LasData):     Original merged LAS data with header info.
            veg_points (laspy.LasData):     Filtered LAS points representing vegetation.
            output_name (str, optional):    Output filename. Defaults to "veg_points.las".

        Creates the output folder if it does not exist and writes the LAS file.
        '''
        # Create a new LasData object with the same header and filtered points
        vegetation_las = laspy.LasData(merged_las.header)
        vegetation_las.points = veg_points.points.copy()

        # Save to file
        output_path = os.path.join(self.output_folder_chm, output_name)
        os.makedirs(self.output_folder_chm, exist_ok=True)
        vegetation_las.write(output_path)
        print(f"Saved vegetation points to {output_path}")

    def find_tiles(self, x_min, y_min, x_max, y_max):
        '''
        Find geotile names overlapping the specified bounding box.

        Parameters:
            x_min, y_min, x_max, y_max (float):       Coordinates defining the bounding box.

        Returns:
            List[str]:                           List of geotile names that intersect with the bounding box.
        '''
        query_geom = box(x_min, y_min, x_max, y_max)
        matches = self.gdf.sindex.query(
            query_geom)  # predicate="overlaps": tricky i want to still get something if it is all contained in one
        return self.gdf.iloc[matches]["GT_AHNSUB"].tolist()

    @staticmethod
    def filter_points_within_bounds(las_data, bounds):
        '''
        Filter LAS points that lie within the given bounding box.

        Parameters:
            las_data (laspy.LasData):   Input LAS point cloud.
            bounds (tuple):             Bounding box as (x_min, y_min, x_max, y_max).

        Returns:
            laspy.LasData:              Filtered LAS data containing only points within bounds.
        '''
        x_min, y_min, x_max, y_max = bounds
        mask = (
                (las_data.x >= x_min) & (las_data.x <= x_max) &
                (las_data.y >= y_min) & (las_data.y <= y_max)
        )
        return las_data[mask]

    # @staticmethod
    def extract_vegetation_points(self, LasData, ndvi_threshold=0.1, pre_filter=False):
        '''
        Extract vegetation points based on classification and NDVI threshold.

        Parameters:
        - LasData (laspy.LasData):          Input LAS point cloud data.
        - ndvi_threshold (float, optional): NDVI cutoff for vegetation points. Defaults to 0.1.
        - pre_filter (bool, optional):      If True, filter out vegetation points below 1.5m above lowest vegetation point. Defaults to False.

        Returns:
        - veg_points (laspy.LasData):       LAS data filtered to vegetation points based on NDVI and optional height filtering.
        '''

        # Filter points based on classification (vegetation-related classes), note: vegetation classes are empty in AHN4
        possible_vegetation_points = LasData[(LasData.classification == 1) |  # Unclassified
                                             (LasData.classification == 3) |  # Low vegetation
                                             (LasData.classification == 4) |  # Medium vegetation
                                             (LasData.classification == 5)]  # High vegetation

        # Calculate NDVI
        red = possible_vegetation_points.red
        nir = possible_vegetation_points.nir
        ndvi = (nir.astype(float) - red) / (nir + red)

        # Filter the points whose NDVI is greater than the threshold
        veg_points = possible_vegetation_points[ndvi > ndvi_threshold]

        # Option: already filter away the points with a height below 1.5m from the lowest veg point, introduced because
        # of one very large tile (25GN2_24.LAZ)
        if pre_filter:
            heights = veg_points.z
            min_height = heights.min()

            # Filter out points with heights between the minimum height and 1.5 meters
            filtered_veg_points = veg_points[(heights <= min_height) | (heights > 1.5)]
            return filtered_veg_points

        self.save_las(LasData, veg_points)

        return veg_points

    @staticmethod
    def raster_center_coords(min_x, max_x, min_y, max_y, resolution):
        '''
        Compute center coordinates of each cell in a raster grid.

        Parameters:
            min_x, max_x, min_y, max_y (float):  Bounding box coordinates.
            resolution (float):                 Cell size; assumed square cells.

        Returns
        -------
        grid_center_x (np.ndarray)
            X cell center coordinates.
        grid_center_y (np.ndarray)
            Y cell center coordinates.
        '''
        # create coordinates for the x and y border of every cell.
        x_coords = np.arange(min_x, max_x, resolution)  # x coordinates expand from left to right.
        y_coords = np.arange(max_y, min_y, -resolution)  # y coordinates reduce from top to bottom.

        # create center point coordinates for evey cell.
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        grid_center_x = grid_x + resolution / 2
        grid_center_y = grid_y - resolution / 2
        return grid_center_x, grid_center_y

    @staticmethod
    def median_filter_chm(chm_array, nodata_value=-9999, size=3):
        '''
        Apply a median filter to smooth the CHM, preserving NoData areas.

        Parameters:
            chm_array (np.ndarray):         CHM raster array.
            nodata_value (float, optional): NoData value in the array. Defaults to -9999.
            size (int, optional):           Median filter size (window). Defaults to 3.

        Returns:
            smoothed_chm (np.ndarray):      Smoothed CHM array with NoData preserved.
        '''
        # Create a mask for valid data
        valid_mask = chm_array != nodata_value

        # Pad the data with nodata_value
        pad_width = size // 2
        padded_chm = np.pad(chm_array, pad_width, mode='constant', constant_values=nodata_value)

        # Apply median filter to padded data
        filtered_padded = median_filter(padded_chm.astype(np.float32), size=size) # median_filter(padded_chm.astype(np.float32), size=size)

        # Remove padding
        smoothed_chm = filtered_padded[pad_width:-pad_width, pad_width:-pad_width]

        # Only keep valid data in smoothed result
        smoothed_chm[~valid_mask] = nodata_value

        return smoothed_chm

    def interpolation_vegetation(self, veg_points, resolution, no_data_value=-9999):
        '''
        Create a vegetation raster by interpolating vegetation points using Laplace interpolation.

        Parameters:
            veg_points (laspy.LasData):       Vegetation points to interpolate.
            resolution (float):               Desired raster resolution.
            no_data_value (int, optional):    Value to assign NoData cells. Defaults to -9999.

        Returns
        -------
        interpolated_grid (np.ndarray):
            Raster grid with interpolated vegetation heights.
        grid_center_xy (tuple of np.ndarray):
            Grid center coordinates (x, y).
        '''
        # bounding box extents minus 0.5 resolution of AHN dataset
        min_x, min_y, max_x, max_y = self.bbox
        # Define size of the region
        x_length = max_x - min_x
        y_length = max_y - min_y

        # Number of rows and columns
        cols = round(x_length / resolution)
        rows = round(y_length / resolution)

        mask = (
            (veg_points.x >= min_x) & (veg_points.x <= max_x) &
            (veg_points.y >= min_y) & (veg_points.y <= max_y)
        )
        veg_points = veg_points[mask]

        # Initialize raster grid
        veg_raster = np.full((rows, cols), no_data_value, dtype=np.float32)

        # Calculate center coords for each grid cell
        grid_center_xy = self.raster_center_coords(min_x, max_x, min_y, max_y, resolution)

        if veg_points.x.shape[0] == 0:
            print("There are no vegetation points in the current area.")
            veg_raster = np.full((rows, cols), -200, dtype=np.float32)
            return veg_raster, grid_center_xy

        # create the delaunay triangulation
        dt = startinpy.DT()
        dt.insert(veg_points.xyz, "BBox")

        # Flatten the grid to get a list of all center coords
        locs = np.column_stack((grid_center_xy[0].ravel(), grid_center_xy[1].ravel()))

        vegetation_points = np.column_stack((veg_points.x, veg_points.y))
        tree = cKDTree(vegetation_points)

        # Find the distance to the nearest vegetation point for each grid cell
        distances, _ = tree.query(locs, k=1)

        distance_threshold = 1
        # masking cells that exceed threshold
        within_threshold_mask = distances <= distance_threshold
        # Interpolation only for those near
        valid_locs = locs[within_threshold_mask]

        # laplace interpolation
        interpolated_values = dt.interpolate({"method": "Laplace"}, valid_locs)

        # reshape interpolated grid back to og
        interpolated_grid = np.full_like(veg_raster, no_data_value, dtype=np.float32)  # Start with no_data
        interpolated_grid.ravel()[within_threshold_mask] = interpolated_values

        return interpolated_grid, grid_center_xy

    def download_las_tiles(self, matching_tiles, output_folder):
        '''
        Download AHN5 or AHN4 LAZ tiles based on a list of matching tile names.

        Parameters:
            matching_tiles (list of str):       List of tile identifiers (e.g., '31FN2_01') to be downloaded.
            output_folder (str):                Directory where downloaded LAZ files will be saved.

        Returns:
            None
        '''
        base_url_ahn5 = "https://geotiles.citg.tudelft.nl/AHN5_T"
        base_url_ahn4 = "https://geotiles.citg.tudelft.nl/AHN4_T"
        os.makedirs(output_folder, exist_ok=True)

        for full_tile_name in matching_tiles:
            # Extract tile name and sub-tile number
            if '_' in full_tile_name:
                tile_name, sub_tile = full_tile_name.split('_')
            else:
                print(f"Skipping invalid tile entry: {full_tile_name}")
                continue

            sub_tile_str = f"_{int(sub_tile):02}"
            filename = f"{tile_name}{sub_tile_str}.LAZ"
            file_path = os.path.join(output_folder, filename)

            # Skip if already downloaded
            if os.path.exists(file_path):
                print(f"File {file_path} already exists, skipping download.")
                continue

            # Try AHN5
            url = f"{base_url_ahn5}/{filename}"
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded from AHN5 and saved {file_path}")
                continue
            except requests.exceptions.RequestException as e:
                print(f"AHN5 download failed for {filename}: {e}")

            # AHN4 fallback
            url = f"{base_url_ahn4}/{filename}"
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded from AHN4 and saved {file_path}")
            except requests.exceptions.RequestException as e:
                print(f"AHN4 download also failed for {filename}: {e}")

    # def merge_las_files(self, laz_files, bounds, merged_output):
    #     '''
    #     Merge and crop multiple LAZ files into a single LAS file within the specified bounds.

    #     Parameters:
    #         laz_files (list of str):        Paths to the input LAZ files.
    #         bounds (tuple):                 Bounding box (xmin, ymin, xmax, ymax) to crop point clouds.
    #         merged_output (str or Path):    File path to write the merged output LAS file.

    #     Returns:
    #         laspy.LasData:                  Merged and cropped point cloud data.
    #     '''

    #     merged_output = Path(merged_output)

    #     las_merged = None
    #     all_points = []
    #     merged_scales = None
    #     merged_offset = None

    #     if merged_output.exists():
    #         with laspy.open(merged_output) as las:
    #             las_merged = las.read()
    #         return las_merged

    #     for file in laz_files:
    #         with laspy.open(file) as las:
    #             las_data = las.read()
    #             cropped_las = self.filter_points_within_bounds(las_data, bounds)

    #             if las_merged is None:
    #                 # Initialize merged LAS file using the first input file
    #                 las_merged = laspy.LasData(las_data.header)
    #                 las_merged.points = cropped_las.points

    #                 merged_scales = las_merged.header.scales
    #                 merged_offset = las_merged.header.offset
    #             else:

    #                 scale = las_data.header.scales
    #                 offset = las_data.header.offsets
    #                 # Convert integer coordinates to real-world values & Transform into merged coordinate system
    #                 new_x = ((cropped_las.X * scale[0] + offset[0]) - merged_offset[0]) / merged_scales[0]
    #                 new_y = ((cropped_las.Y * scale[1] + offset[1]) - merged_offset[1]) / merged_scales[1]
    #                 new_z = ((cropped_las.Z * scale[2] + offset[2]) - merged_offset[2]) / merged_scales[2]

    #                 # Copy points and update X, Y, Z
    #                 new_points = cropped_las.points
    #                 new_points["X"] = new_x.astype(np.int32)
    #                 new_points["Y"] = new_y.astype(np.int32)
    #                 new_points["Z"] = new_z.astype(np.int32)

    #                 all_points.append(new_points.array)

    #                 # Final merge step
    #     if las_merged is not None:
    #         if all_points:
    #             all_points.append(las_merged.points.array)

    #             merged_array = np.concatenate(all_points, axis=0)
    #             las_merged.points = laspy.ScaleAwarePointRecord(merged_array, las_merged.header.point_format,
    #                                                             las_merged.header.scales, las_merged.header.offsets)

    #         las_merged.write(str(merged_output))

    #     return las_merged

    def merge_las_files(self, laz_files, bounds, merged_output):
        """
        Merge and crop multiple LAZ files into a single LAS/LAZ file within the specified bounds.

        Parameters:
            laz_files (list[str|Path]): Paths to input LAZ/LAS files.
            bounds (tuple): (xmin, ymin, xmax, ymax) crop bounds in real-world coordinates.
            merged_output (str|Path): Output file path.

        Returns:
            laspy.LasData | None: merged LAS data, or None if no points.
        """
        merged_output = Path(merged_output)

        # If already exists, just read and return
        if merged_output.exists():
            with laspy.open(merged_output) as las:
                return las.read()

        base_header = None
        base_scales = None
        base_offsets = None
        base_point_format = None

        arrays = []

        for file in laz_files:
            with laspy.open(file) as las:
                las_data = las.read()
                cropped = self.filter_points_within_bounds(las_data, bounds)

                if len(cropped.points) == 0:
                    continue

                # Initialize base from first non-empty tile
                if base_header is None:
                    base_header = las_data.header.copy()
                    base_scales = base_header.scales
                    base_offsets = base_header.offsets
                    base_point_format = base_header.point_format

                # Copy the underlying point record array (raw ints + other dims)
                pts = cropped.points.array.copy()

                # If scale/offset differ, re-encode X/Y/Z into base integer grid
                s = las_data.header.scales
                o = las_data.header.offsets
                if (not np.allclose(s, base_scales)) or (not np.allclose(o, base_offsets)):
                    # IMPORTANT:
                    # cropped.X/Y/Z in laspy are typically *real coordinates* (scaled)
                    # So use: (real - base_offset) / base_scale => base integer
                    pts["X"] = np.rint((cropped.X - base_offsets[0]) / base_scales[0]).astype(np.int32)
                    pts["Y"] = np.rint((cropped.Y - base_offsets[1]) / base_scales[1]).astype(np.int32)
                    pts["Z"] = np.rint((cropped.Z - base_offsets[2]) / base_scales[2]).astype(np.int32)
                    # If in your environment cropped.X is NOT real but already integer,
                    # replace the three lines above with the "real calc" version:
                    # x_real = cropped.X * s[0] + o[0]  (etc)

                arrays.append(pts)

        if not arrays:
            return None

        merged_array = np.concatenate(arrays, axis=0)

        # Build output LAS with base header and assign points
        las_merged = laspy.LasData(base_header)

        # Ensure point format matches base (in case source files differ slightly)
        las_merged.points = laspy.ScaleAwarePointRecord(
            merged_array,
            base_point_format,
            base_scales,
            base_offsets,
        )

        # Update mins/maxs, counts, etc.
        las_merged.update_header()

        las_merged.write(merged_output)
        return las_merged

    @staticmethod
    def chm_finish(chm_array, dtm_array,
                   dsm_array, min_height=2, max_height=40):
        '''
        Finalize CHM by removing terrain and filtering by vegetation height.

        Parameters:
            chm_array (np.ndarray):     Initial canopy height model array.
            dtm_array (np.ndarray):     Digital terrain model array.
            dsm_array (np.ndarray):     Digital surface model array.
            min_height (float):         Minimum height threshold to keep vegetation (default = 2).
            max_height (float):         Maximum height threshold to keep vegetation (default = 40).

        Returns:
            np.ndarray: Processed CHM with invalid or noisy values removed.
        '''

        result_array = chm_array - dtm_array
        result_array[(chm_array - dsm_array) < 0.0] = 0
        result_array[(result_array < min_height) | (result_array > max_height)] = 0
        result_array[np.isnan(result_array)] = 0

        return result_array

    def chm_creation(self, LasData, vegetation_data, output_filename, resolution=0.5, smooth=False, nodata_value=-9999,
                     filter_size=3):
        '''
        Create and optionally smooth a CHM from vegetation data, then save it as a GeoTIFF and extract tree polygons.

        Parameters:
            LasData (laspy.LasData):    LAS metadata for writing the output raster.
            vegetation_data (tuple):    Tuple of (veg_raster, grid_centers) for CHM generation.
            output_filename (str):      Path to save the output CHM raster.
            resolution (float):         Spatial resolution of the raster (default = 0.5).
            smooth (bool):              Whether to apply a median filter to the CHM (default = False).
            nodata_value (float):       Value to assign to NoData cells in the raster (default = -9999).
            filter_size (int):          Size of median filter kernel (default = 3).

        Returns:
            tuple: (chm_array, polygons, transform) where polygons are tree regions as GeoJSON-like dicts.
        '''

        veg_raster = vegetation_data[0]
        grid_centers = vegetation_data[1]
        top_left_x = grid_centers[0][0, 0] - resolution / 2
        top_left_y = grid_centers[1][0, 0] + resolution / 2

        transform = Affine.translation(top_left_x, top_left_y) * Affine.scale(resolution, -resolution)

        if smooth:
            veg_raster = self.median_filter_chm(veg_raster, nodata_value=nodata_value, size=filter_size)


        veg_raster = self.chm_finish(veg_raster, self.dtm, self.dsm)

        write_output(LasData, self.crs, veg_raster, transform, output_filename, True)

        # create the polygons
        labeled_array, num_clusters = label(veg_raster > 0)
        shapes_gen = shapes(labeled_array.astype(np.uint8), mask=(labeled_array > 0), transform=transform)
        polygons = [
            {"geometry": shape(geom), "polygon_id": int(value)}
            for geom, value in shapes_gen if value > 0
        ]

        return veg_raster, polygons, transform

    def init_chm(self, bbox, output_folder="output", input_folder="temp",  merged_output="pointcloud.las",  smooth_chm=True, resolution=0.5, ndvi_threshold=0.05, filter_size=3):
        '''
        Initialize and generate a CHM by downloading, merging, filtering, and interpolating LiDAR data.

        Parameters:
            bbox (tuple):           Bounding box (xmin, ymin, xmax, ymax) for the area of interest.
            output_folder (str):    Directory for saving output files (default = 'output').
            input_folder (str):     Directory where LAZ files are stored or downloaded (default = 'temp').
            merged_output (str):    Path to save the merged LAS point cloud (default = 'output/pointcloud.las').
            smooth_chm (bool):      Whether to smooth the CHM using a median filter (default = True).
            resolution (float):     Output raster resolution (default = 0.5).
            ndvi_threshold (float): NDVI threshold for filtering vegetation points (default = 0.05).
            filter_size (int):      Size of median filter kernel (default = 3).

        Returns:
            tuple: (chm_array, polygons, transform) or (None, None, None) if process fails.
        '''
        if not os.path.exists(input_folder):
            os.makedirs(input_folder)

        matching_tiles = self.find_tiles(*self.bufferedbbox)
        print("Tiles covering the area:", matching_tiles)

        existing_tiles = {
            os.path.splitext(file)[0] for file in os.listdir(input_folder) if file.endswith(".LAZ")
        }

        missing_tiles = [tile for tile in matching_tiles if tile not in existing_tiles]

        if missing_tiles:
            print("Missing tiles:", missing_tiles)
            self.download_las_tiles(missing_tiles, input_folder)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        laz_files = [
            os.path.join(input_folder, file)
            for file in os.listdir(input_folder)
            if file.endswith(".LAZ") and os.path.splitext(file)[0] in matching_tiles
        ]

        if not laz_files:
            print("No relevant LAZ files found in the input folder or its subfolders.")
            return None, None, None
        las_data = self.merge_las_files(laz_files, self.bufferedbbox, f"{input_folder}/{merged_output}")

        if las_data is None:
            print("No valid points found in the given boundary.")
            return None, None, None
        las_data.write("test_output/pointcloud.las")
        # Extract vegetation points
        veg_points = self.extract_vegetation_points(las_data, ndvi_threshold=ndvi_threshold, pre_filter=False)

        vegetation_data = self.interpolation_vegetation(veg_points, resolution)
        output_filename = os.path.join(self.output_folder_chm, f"CHM.TIF")

        # Create the CHM and save it
        chm, polygons, transform = self.chm_creation(las_data, vegetation_data, output_filename, resolution=resolution, smooth=smooth_chm, nodata_value=-9999,
                     filter_size=filter_size)

        return chm, polygons, transform

    