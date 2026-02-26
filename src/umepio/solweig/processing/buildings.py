'''All necessary functions to load building footprints and geometries, adjusted from SOLWEIG_SOLFD by Jessica Monahan'''
from umepio.solweig.io_utils import *
from io import BytesIO
import requests
from shapely.geometry import mapping
import gzip
import os
import pandas as pd
import geopandas as gpd


class Buildings:
    '''
    Manage 3D building data within a bounding box by downloading, loading,
    and modifying building geometries from a WFS service.

    Attributes:
        bbox (tuple):                   Bounding box (min_x, min_y, max_x, max_y) for the area of interest.
        bufferbbox (tuple):             Buffered bounding box expanded by 2 units.
        wfs_url (str):                  URL of the WFS service to download building data.
        layer_name (str):               WFS layer name to query.
        data (GeoDataFrame):            Downloaded building data.
        building_geometries (list):     List of building geometries with parcel IDs.
        is3D (bool):                    Flag indicating if 3D building data is used.
    '''

    def __init__(self, bbox, wfs_url="https://data.3dbag.nl/api/BAG3D/wfs", layer_name="BAG3D:lod13", gpkg_name="buildings", output_folder = "output", output_layer_name="buildings"):
        '''
        Initialize the Buildings object by setting bounding boxes, downloading,
        and loading building data.

        Parameters:
            bbox (tuple):               Bounding box (min_x, min_y, max_x, max_y).
            wfs_url (str):              URL for the WFS service. Default is 3dbag.nl API.
            layer_name (str):           Name of the WFS layer to query. Default is "BAG3D:lod13".
            gpkg_name (str):            Name of the GeoPackage output file (without extension).
            output_folder (str):        Folder to save the downloaded data.
            output_layer_name (str):    Layer name to save within the GeoPackage.
        '''
        self.bbox = bbox
        self.bufferbbox = edit_bounds(bbox, 2)
        self.wfs_url = wfs_url
        self.layer_name = layer_name
        self.data = self.download_wfs_data(gpkg_name, output_folder, output_layer_name)
        self.building_geometries = self.load_buildings(self.data)
        self.is3D = False

    def download_wfs_data(self, gpkg_name, output_folder, layer_name):
        '''
        Download building features from the WFS service within the buffered bounding box.
        Saves the data as a GeoPackage file.

        Parameters:
            gpkg_name (str):        Filename for the GeoPackage (without extension).
            output_folder (str):    Folder to save the GeoPackage.
            layer_name (str):       Layer name to use inside the GeoPackage.

        Returns:
            GeoDataFrame:           Downloaded building features concatenated, or None if no features were downloaded.
        '''
        all_features = []
        start_index = 0
        count = 10000

        while True:
            params = {
                "SERVICE": "WFS",
                "REQUEST": "GetFeature",
                "VERSION": "2.0.0",
                "TYPENAMES": self.layer_name,
                "SRSNAME": "urn:ogc:def:crs:EPSG::28992",
                "BBOX": f"{self.bufferbbox[0]},{self.bufferbbox[1]},{self.bufferbbox[2]},{self.bufferbbox[3]},urn:ogc:def:crs:EPSG::28992",
                "COUNT": count,
                "STARTINDEX": start_index
            }
            headers = {"User-Agent": "Mozilla/5.0 QGIS/33411/Windows 11 Version 2009"}
            response = requests.get(self.wfs_url, params=params, headers=headers)

            if response.status_code == 200:
                if response.headers.get('Content-Encoding', '').lower() == 'gzip' and response.content[
                                                                                      :2] == b'\x1f\x8b':
                    data = gzip.decompress(response.content)
                else:
                    data = response.content

                with BytesIO(data) as f:
                    gdf = gpd.read_file(f)
                all_features.append(gdf)
                if len(gdf) < count:
                    break
                start_index += count
            else:
                print(f"Failed to download WFS data. Status code: {response.status_code}")
                print(f"Error message: {response.text}")
                return gpd.GeoDataFrame()

        if all_features:
            full_gdf = gpd.GeoDataFrame(pd.concat(all_features, ignore_index=True))
            os.makedirs(output_folder, exist_ok=True)
            output_gpkg = os.path.join(output_folder, f"{gpkg_name}.gpkg")
            full_gdf.to_file(output_gpkg, layer=layer_name, driver="GPKG")
            print(f"GPKG with buildings saved to folder: {output_folder}")
            return full_gdf
        else:
            print("No features were downloaded.")
            return None
        
    @staticmethod    
    def load_buildings(buildings_gdf, buildings_path=None, layer=None):
        '''
        Load building geometries from a GeoDataFrame or from a file.

        Parameters:
            buildings_gdf (GeoDataFrame or None):   Building data GeoDataFrame.
            buildings_path (str or None):           Path to building file to load if GeoDataFrame is None.
            layer (str or None):                    Layer name to read from file if applicable.

        Returns:
            list:   List of dicts with 'geometry' (GeoJSON mapping) and 'parcel_id'. None if no data could be loaded.
        '''
        if buildings_gdf is None:
            if buildings_path is not None:
                buildings_gdf = gpd.read_file(buildings_path, layer=layer)
            else: return None

        return [{"geometry": mapping(geom), "parcel_id": identificatie} for geom, identificatie in
                zip(buildings_gdf.geometry, buildings_gdf["identificatie"])]


